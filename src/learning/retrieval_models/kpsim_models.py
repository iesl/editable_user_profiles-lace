"""
Models which learn keyphrase representations.
Mostly a bunch of wrappers for raw bert models which are finetuned.
"""
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from transformers import AutoModel


class KeyPhraseSentBERTWrapper(nn.Module):
    """
    Pass sentence through encoder and minimize triple loss
    with inbatch negatives.
    """
    def __init__(self, model_hparams):
        """
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768
        self.sent_context_encoder = AutoModel.from_pretrained(model_hparams['sent-base-pt-layer'])
        # Use a frozen sentence context encoder if asked; mostly it will be frozen.
        if not model_hparams['sent_fine_tune']:
            for param in self.sent_context_encoder.base_model.parameters():
                param.requires_grad = False
        self.kp_encoder = AutoModel.from_pretrained(model_hparams['kp-base-pt-layer'])
        self.criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
    
    def forward(self, batch_dict):
        batch_bpr = self.forward_rank(batch_dict['batch_rank'])
        loss_dict = {
            'rankl': batch_bpr
        }
        return loss_dict
    
    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
            {
                'query_bert_batch': dict(); BERT inputs for the keyphrases.
                'pos_bert_batch': dict(); BERT inputs for the independent
                    sentences that keyphrases occur in.
            }
        :return: loss_val; torch Variable.
        """
        qbert_batch = batch_rank['query_bert_batch']
        pbert_batch = batch_rank['pos_bert_batch']
        # Get the representations from the model.
        q_sent_reps = self.sent_reps_bert(bert_model=self.kp_encoder, bert_batch=qbert_batch)
        p_context_reps = self.sent_reps_bert(bert_model=self.sent_context_encoder, bert_batch=pbert_batch)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch = batch_rank['neg_bert_batch']
            n_context_reps = self.sent_reps_bert(bert_model=self.sent_context_encoder, bert_batch=nbert_batch)
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            n_context_reps = p_context_reps[torch.randperm(p_context_reps.size()[0])]
        loss_val = self.criterion(q_sent_reps, p_context_reps, n_context_reps)
        return loss_val
    
    def sent_reps_bert(self, bert_model, bert_batch):
        """
        Get representation for the string passed via bert by averaging
        token embeddings; string can be a sentence or a phrase.
        :param bert_model: torch.nn.Module subclass. A bert model.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens');
            items to use for getting BERT representations. The sentence mapped to
            BERT vocab and appropriately padded.
        :return:
            sent_reps: FloatTensor [batch_size x bert_encoding_dim]
        """
        seq_lens = bert_batch['seq_lens']
        batch_size, max_seq_len = len(seq_lens), max(seq_lens)
        sent_mask = np.zeros((batch_size, max_seq_len, self.bert_encoding_dim))
        # Build a mask for
        for i, seq_len in enumerate(seq_lens):
            sent_mask[i, :seq_len, :] = 1.0
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        sent_mask = Variable(torch.FloatTensor(sent_mask))
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
            sent_mask = sent_mask.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = bert_model(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        sent_tokens = final_hidden_state * sent_mask
        # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
        sent_reps = torch.sum(sent_tokens, dim=1)/torch.count_nonzero(
            sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
        return sent_reps


class KeyPhraseAspireWrapper(nn.Module):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Maximize similarity of keyphrase to encoding
    """
    
    def __init__(self, model_hparams):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.sent_context_encoder = AutoModel.from_pretrained(model_hparams['consent-base-pt-layer'])
        # If fine tune is False then freeze the bert params.
        if not model_hparams['sent_fine_tune']:
            for param in self.sent_context_encoder.base_model.parameters():
                param.requires_grad = False
        self.kp_encoder = AutoModel.from_pretrained(model_hparams['kp-base-pt-layer'])
        self.criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')

    def forward(self, batch_dict):
        batch_loss = self.forward_rank(batch_dict['batch_rank'])
        loss_dict = {
            'rankl': batch_loss
        }
        return loss_dict
    
    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
        {
            'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'pos_abs_lens': list(int);
            'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from positive abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'pos_senttoki': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
            'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'neg_abs_lens': list(int);
            'neg_senttoki': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
        }
        :return: loss_val; torch Variable.
        """
        qbert_batch = batch_rank['query_bert_batch']
        pbert_batch, pabs_lens = batch_rank['pos_bert_batch'], batch_rank['pos_abs_lens']
        pos_senttoki = batch_rank['pos_senttok_idxs']
        q_repeat_idxs, pos_sent_idxs = batch_rank['query_repeat_idxs'], batch_rank['pos_sent_idxs']
        # Get the representations from the model.
        q_kp_reps = self.sent_reps_bert(bert_model=self.kp_encoder, bert_batch=qbert_batch)
        _, p_sent_reps = self.partial_forward(bert_batch=pbert_batch, abs_lens=pabs_lens, sent_tok_idxs=pos_senttoki)
        # Create repeated copies of the kp and get the positive sentences.
        q_repeat_idxs, pos_sent_idxs = Variable(q_repeat_idxs), Variable(pos_sent_idxs)
        if torch.cuda.is_available():
            q_repeat_idxs = q_repeat_idxs.cuda()
            pos_sent_idxs = pos_sent_idxs.cuda()
        q_repkp_reps = torch.index_select(q_kp_reps, dim=0, index=q_repeat_idxs)
        batch_size, max_sents, encoding_dim = p_sent_reps.size()
        pos_con_sent_reps = torch.index_select(p_sent_reps.view(batch_size*max_sents, encoding_dim), dim=0,
                                               index=pos_sent_idxs)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch, nabs_lens = batch_rank['neg_bert_batch'], batch_rank['neg_abs_lens']
            neg_senttoki = batch_rank['neg_senttok_idxs']
            neg_sent_idxs = batch_rank['neg_sent_idxs']
            _, n_sent_reps = self.partial_forward(bert_batch=nbert_batch, abs_lens=nabs_lens, sent_tok_idxs=neg_senttoki)
            neg_sent_idxs = Variable(neg_sent_idxs)
            if torch.cuda.is_available():
                neg_sent_idxs = neg_sent_idxs.cuda()
            batch_size, max_sents, encoding_dim = n_sent_reps.size()
            neg_con_sent_reps = torch.index_select(n_sent_reps.view(batch_size*max_sents, encoding_dim), dim=0,
                                                   index=neg_sent_idxs)
            
            loss_val = self.criterion(q_repkp_reps, pos_con_sent_reps, neg_con_sent_reps)
            return loss_val
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            random_idxs = torch.randperm(pos_con_sent_reps.size()[0])
            neg_con_sent_reps = pos_con_sent_reps[random_idxs]

            loss_val = self.criterion(q_repkp_reps, pos_con_sent_reps, neg_con_sent_reps)
            return loss_val
    
    def partial_forward(self, bert_batch, abs_lens, sent_tok_idxs):
        """
        Pass a batch of sentences through BERT and read off sentence
        representations based on SEP idxs.
        :return:
            sent_reps: batch_size x q_max_sents x encoding_dim
        """
        # batch_size x num_sents x encoding_dim
        doc_cls_reps, sent_reps = self.con_sent_reps_bert(bert_batch=bert_batch, num_sents=abs_lens,
                                                          batch_senttok_idxs=sent_tok_idxs)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        if len(doc_cls_reps.size()) == 1:
            doc_cls_reps = doc_cls_reps.unsqueeze(0)
        # batch_size x q_max_sents x encoding_dim;
        return doc_cls_reps, sent_reps
    
    def con_sent_reps_bert(self, bert_batch, batch_senttok_idxs, num_sents):
        """
        Pass the concated abstract through BERT, and average token reps to get sentence reps.
        -- NO weighted combine across layers.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
            representations. The sentence mapped to BERT vocab and appropriately padded.
        :param batch_senttok_idxs: list(list(list(int))); batch_size([num_sents_per_abs[num_tokens_in_sent]])
        :param num_sents: list(int); number of sentences in each example in the batch passed.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
            sent_reps: FloatTensor [batch_size x num_sents x bert_encoding_dim]
        """
        seq_lens = bert_batch['seq_lens']
        batch_size, max_seq_len = len(seq_lens), max(seq_lens)
        max_sents = max(num_sents)
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.sent_context_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        # Read of CLS token as document representation.
        doc_cls_reps = final_hidden_state[:, 0, :]
        doc_cls_reps = doc_cls_reps.squeeze()
        # Average token reps for every sentence to get sentence representations.
        # Build the first sent for all batch examples, second sent ... and so on in each iteration below.
        sent_reps = []
        for sent_i in range(max_sents):
            cur_sent_mask = np.zeros((batch_size, max_seq_len, self.bert_encoding_dim))
            # Build a mask for the ith sentence for all the abstracts of the batch.
            for batch_abs_i in range(batch_size):
                abs_sent_idxs = batch_senttok_idxs[batch_abs_i]
                try:
                    sent_i_tok_idxs = abs_sent_idxs[sent_i]
                except IndexError:  # This happens in the case where the abstract has fewer than max sents.
                    sent_i_tok_idxs = []
                cur_sent_mask[batch_abs_i, sent_i_tok_idxs, :] = 1.0
            sent_mask = Variable(torch.FloatTensor(cur_sent_mask))
            if torch.cuda.is_available():
                sent_mask = sent_mask.cuda()
            # batch_size x seq_len x encoding_dim
            sent_tokens = final_hidden_state * sent_mask
            # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
            cur_sent_reps = torch.sum(sent_tokens, dim=1)/ \
                            torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
            sent_reps.append(cur_sent_reps.unsqueeze(dim=1))
        # batch_size x max_sents x encoding_dim
        sent_reps = torch.cat(sent_reps, dim=1)
        return doc_cls_reps, sent_reps

    def sent_reps_bert(self, bert_model, bert_batch):
        """
        Get representation for the string passed via bert by averaging
        token embeddings; string can be a sentence or a phrase.
        :param bert_model: torch.nn.Module subclass. A bert model.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens');
            items to use for getting BERT representations. The sentence mapped to
            BERT vocab and appropriately padded.
        :return:
            sent_reps: FloatTensor [batch_size x bert_encoding_dim]
        """
        seq_lens = bert_batch['seq_lens']
        batch_size, max_seq_len = len(seq_lens), max(seq_lens)
        sent_mask = np.zeros((batch_size, max_seq_len, self.bert_encoding_dim))
        # Build a mask for
        for i, seq_len in enumerate(seq_lens):
            sent_mask[i, :seq_len, :] = 1.0
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        sent_mask = Variable(torch.FloatTensor(sent_mask))
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
            sent_mask = sent_mask.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = bert_model(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        sent_tokens = final_hidden_state * sent_mask
        # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
        sent_reps = torch.sum(sent_tokens, dim=1)/torch.count_nonzero(
            sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
        return sent_reps