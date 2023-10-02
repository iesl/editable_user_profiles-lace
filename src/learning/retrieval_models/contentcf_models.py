"""
Content based recommenders similar to the Ask the GRU paper.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from transformers import AutoModel


class ContentCF(nn.Module):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Learn a single representation per user like in collaborative filtering.
    """
    # Set externally before initialization.
    num_users = None
    
    def __init__(self, model_hparams):
        """
        :param model_hparams: dict(string:int); model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        self.user_embeddings = nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.bert_encoding_dim)
        loss_margin = model_hparams.get('loss_margin', 1.0)
        self.criterion = nn.TripletMarginLoss(margin=loss_margin, p=2, reduction='sum')
    
    def caching_score(self, query_user_idx, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of user rep as cand_reps.
        - Compute scores and return.
        query_encode_ret_dict: int; idexes the user_embeddings.
        cand_encode_ret_dict: list({'doc_reps': numpy.array})
        """
        batch_size = len(cand_encode_ret_dicts)
        # Flatten the query abstracts sentences into a single doc with many sentences
        repeated_user_idxs = torch.LongTensor([query_user_idx] * batch_size)
        # Pack candidate representations.
        cand_doc_reps = [d['doc_reps'] for d in cand_encode_ret_dicts]
        cand_doc_reps = torch.FloatTensor(np.vstack(cand_doc_reps))
        # Move to gpu.
        repeated_user_idxs = Variable(repeated_user_idxs)
        cand_doc_reps = Variable(cand_doc_reps)
        if torch.cuda.is_available():
            repeated_user_idxs = repeated_user_idxs.cuda()
            cand_doc_reps = cand_doc_reps.cuda()
        
        # Compute scores as at train time.
        user_reps = self.user_embeddings(repeated_user_idxs)
        user2cand_sims = -1 * torch.cdist(user_reps.unsqueeze(1), cand_doc_reps.unsqueeze(1))
        user2cand_sims = user2cand_sims.squeeze()
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            batch_scores = user2cand_sims.cpu().data.numpy()
        else:
            batch_scores = user2cand_sims.data.numpy()
        
        # Return both the same thing in both because the nearest and printing code expects it.
        ret_dict = {
            'batch_scores': batch_scores,
            'pair_scores': batch_scores
        }
        return ret_dict
    
    def caching_encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        abs_bert_batch = batch_dict['abs_bert_batch']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        doc_cls_reps = self.partial_forward(bert_batch=abs_bert_batch)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            doc_cls_reps = doc_cls_reps.cpu().data.numpy()
        else:
            doc_cls_reps = doc_cls_reps.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i in range(doc_cls_reps.shape[0]):
            # 1 x encoding_dim
            upsr = doc_cls_reps[i, :]
            # return: # num_sents x encoding_dim
            batch_reps.append({'doc_reps': upsr})
        return batch_reps
    
    def encode(self, batch_dict):
        """
        Function used at test time when encoding only sentences.
        This mimics the encode function of disent_models.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        # Not sure where this will be used - may need small modifications based on that.
        raise NotImplementedError
        abs_bert_batch = batch_dict['bert_batch']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        doc_cls_reps = self.partial_forward(bert_batch=abs_bert_batch)
        
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            doc_cls_reps = doc_cls_reps.cpu().data.numpy()
        else:
            doc_cls_reps = doc_cls_reps.data.numpy()
        
        # Return a list of reps instead of reps collated as one np array.
        doc_reps = []
        for i in enumerate(doc_cls_reps.shape[0]):
            # 1 x encoding_dim
            upsr = doc_cls_reps[i, :]
            # return: # num_sents x encoding_dim
            doc_reps.append(upsr)
        ret_dict = {
            'doc_reps': doc_reps
        }
        return ret_dict
    
    def forward(self, batch_dict):
        batch_losses = self.forward_rank(batch_dict['batch_rank'])
        loss_dict = {
            'rankl': batch_losses
        }
        return loss_dict
    
    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
        {
            'user_idxs': torch.LongTensor; Indexes for the user representations.
            'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'abs_lens': list(int); Number of sentences in query abs.
        }
        :return: loss_val; torch Variable.
        """
        # Get the abstract representations from the model.
        pos_cls_reps = self.partial_forward(bert_batch=batch_rank['pos_bert_batch'])
        # Get the user representations from the embedding matrix.
        user_idxs = Variable(batch_rank['user_idxs'])
        if torch.cuda.is_available():
            user_idxs = user_idxs.cuda()
        user_reps = self.user_embeddings(user_idxs)

        if 'neg_bert_batch' in batch_rank:
            nbert_batch = batch_rank['neg_bert_batch']
            neg_cls_reps = self.partial_forward(bert_batch=nbert_batch)
            
            loss_val = self.criterion(user_reps, pos_cls_reps, neg_cls_reps)
            return loss_val
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            random_idxs = torch.randperm(pos_cls_reps.size()[0])
            neg_cls_reps = pos_cls_reps[random_idxs]
            
            loss_val = self.criterion(user_reps, pos_cls_reps, neg_cls_reps)
            
        return loss_val

    def partial_forward(self, bert_batch):
        """
        Pass a batch of sentences through BERT and get contextual sentence reps.
        :return:
            sent_reps: batch_size x num_sents x encoding_dim
        """
        # batch_size x encoding_dim
        doc_cls_reps = self.doc_reps_bert(bert_batch=bert_batch)
        if len(doc_cls_reps.size()) == 1:
            doc_cls_reps = doc_cls_reps.unsqueeze(0)
        return doc_cls_reps

    def doc_reps_bert(self, bert_batch):
        """
        Pass the concated abstract through BERT, and read off CLS
        rep to get the document vector.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use
            for getting BERT representations. The sentence mapped to BERT vocab and
            appropriately padded.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
        """
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        # Read of CLS token as document representation: (batch_size, hidden_size)
        cls_doc_reps = model_outputs.last_hidden_state[:, 0, :]
        cls_doc_reps = cls_doc_reps.squeeze()
        return cls_doc_reps
    