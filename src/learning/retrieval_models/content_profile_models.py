"""
Models which learn facet disentangled representations of paper abstracts.
"""
import collections
import logging
import os, codecs, json
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
from transformers import AutoModel

from . import pair_distances as pair_dist
from .pair_distances import rep_len_tup
from ..models_common import generic_layers as gl


class MySPECTER(nn.Module):
    """
    Pass abstract through SciBERT all in one shot, read off cls token and use
    it to compute similarities. This is an unfaceted model and is meant to
    be similar to SPECTER in all aspects:
    - triplet loss function
    - only final layer cls bert representation
    - no SEP tokens in between abstract sentences
    """
    
    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        self.bert_layer_weights = gl.SoftmaxMixLayers(in_features=self.bert_layer_count, out_features=1, bias=False)
        self.criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')

    def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Compute scores and return.
        query_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array})
        cand_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array})
        """
        # Pack representations as padded gpu tensors.
        query_cls_reps = [d['doc_cls_reps'] for d in query_encode_ret_dict]
        num_query_abs = len(query_cls_reps)
        query_cls_reps = np.vstack(query_cls_reps)
        cand_cls_reps = [d['doc_cls_reps'] for d in cand_encode_ret_dicts]
        batch_size = len(cand_cls_reps)
        flat_query_cls_reps = np.zeros((batch_size, num_query_abs, self.bert_encoding_dim))
        for bi in range(batch_size):
            flat_query_cls_reps[bi, :num_query_abs, :] = query_cls_reps
        flat_query_cls_reps, cand_cls_reps = Variable(torch.FloatTensor(flat_query_cls_reps)),\
                                             Variable(torch.FloatTensor(np.vstack(cand_cls_reps)))
        if torch.cuda.is_available():
            # batch_size x num_query_abs x encoding_dim
            flat_query_cls_reps = flat_query_cls_reps.cuda()
            # batch_size x encoding_dim
            cand_cls_reps = cand_cls_reps.cuda()
        # Compute scores from all user docs to candidate docs.
        cand2user_doc_sims = -1*torch.cdist(cand_cls_reps.unsqueeze(1), flat_query_cls_reps)
        # batch_size x 1 x num_query_abs
        cand_sims, _ = torch.max(cand2user_doc_sims.squeeze(1), dim=1)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            batch_scores = cand_sims.cpu().data.numpy()
        else:
            batch_scores = cand_sims.data.numpy()
        # Return the same thing as batch_scores and pair_scores because the pp_gen_nearest class expects it.
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
        doc_bert_batch, batch_size = batch_dict['bert_batch'], len(batch_dict['bert_batch']['seq_lens'])
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        doc_cls_reps = self.partial_forward(bert_batch=doc_bert_batch)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            doc_cls_reps = doc_cls_reps.cpu().data.numpy()
        else:
            doc_cls_reps = doc_cls_reps.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i in range(batch_size):
            batch_reps.append({'doc_cls_reps': doc_cls_reps[i, :]})
        return batch_reps

    def encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch = batch_dict['bert_batch']
        # Get the representations from the model.
        doc_reps = self.partial_forward(bert_batch=doc_bert_batch)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            doc_reps = doc_reps.cpu().data.numpy()
        else:
            doc_reps = doc_reps.data.numpy()
        ret_dict = {
            'doc_reps': doc_reps,  # batch_size x encoding_dim
        }
        return ret_dict

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
                'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from positive abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
                'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                    concated sentences from query abstracts; Tokenized and int mapped
                    sentences and other inputs to BERT.
            }
        :return: loss_val; torch Variable.
        """
        qbert_batch = batch_rank['query_bert_batch']
        pbert_batch = batch_rank['pos_bert_batch']
        # Get the representations from the model.
        q_sent_reps = self.partial_forward(bert_batch=qbert_batch)
        p_context_reps = self.partial_forward(bert_batch=pbert_batch)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch = batch_rank['neg_bert_batch']
            n_context_reps = self.partial_forward(bert_batch=nbert_batch)
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            n_context_reps = p_context_reps[torch.randperm(p_context_reps.size()[0])]
        loss_val = self.criterion(q_sent_reps, p_context_reps, n_context_reps)
        return loss_val
    
    def partial_forward(self, bert_batch):
        """
        Function shared between the training and test time behaviour. Pass a batch
        of sentences through BERT and return cls representations.
        :return:
            cls_doc_reps: batch_size x encoding_dim
        """
        # batch_size x bert_encoding_dim
        cls_doc_reps = self.doc_reps_bert(bert_batch=bert_batch)
        if len(cls_doc_reps.size()) == 1:
            cls_doc_reps = cls_doc_reps.unsqueeze(0)
        return cls_doc_reps

    def doc_reps_bert(self, bert_batch):
        """
        Pass the concated abstract through BERT, and read off [SEP] token reps to get sentence reps,
        and weighted combine across layers.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
            representations. The sentence mapped to BERT vocab and appropriately padded.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
        """
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        # Weighted combine the hidden_states which is a list of [bs x max_seq_len x bert_encoding_dim]
        # with as many tensors as layers + 1 input layer.
        hs_stacked = torch.stack(model_outputs.hidden_states, dim=3)
        weighted_sum_hs = self.bert_layer_weights(hs_stacked)  # [bs x max_seq_len x bert_encoding_dim x 1]
        weighted_sum_hs = torch.squeeze(weighted_sum_hs, dim=3)
        # Read of CLS token as document representation: (batch_size, sequence_length, hidden_size)
        cls_doc_reps = weighted_sum_hs[:, 0, :]
        cls_doc_reps = cls_doc_reps.squeeze()
        return cls_doc_reps


class SPECTER(MySPECTER):
    """
    Pass abstract through SciBERT all in one shot, read off cls token and use
    it to compute similarities.
    - This is meant to mimic HF specter which does not have the linear mixing layers.
    """
    
    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        self.criterion = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
    
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


class SentAlignBiEnc(MySPECTER):
    """
    **Unused now because of SEP tokens not making sense as sentence reps.**
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
    - Compute pairwise sentence similarities for query and candidate.
    - Maximize maximum similarity of anchor and positive.
    """
    
    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['fine_tune']:
            for param in self.bert_encoder.base_model.parameters():
                param.requires_grad = False
        if model_hparams['score_aggregation'] == 'dotlse':
            self.dist_function = pair_dist.allpair_masked_dist_dotlse
        elif model_hparams['score_aggregation'] == 'dotmax':
            self.dist_function = pair_dist.allpair_masked_dist_dotmax
        elif model_hparams['score_aggregation'] == 'dottop2':
            self.dist_function = pair_dist.allpair_masked_dist_dottopk
        elif model_hparams['score_aggregation'] == 'cosinemax':
            self.dist_function = pair_dist.allpair_masked_dist_cosinemax
        else:
            raise ValueError(f'Unknown aggregation: {model_hparams["score_aggregation"]}')
        # self.bert_layer_weights = gl.SoftmaxMixLayers(in_features=self.bert_layer_count, out_features=1, bias=False)
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function,
                                                          margin=1.0, reduction='sum')
    
    def encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
        doc_query_sepsi = batch_dict['sep_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        sent_reps = self.partial_forward(bert_batch=doc_bert_batch, abs_lens=doc_abs_lens,
                                         sent_sep_idxs=doc_query_sepsi)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            sent_reps = sent_reps.cpu().data.numpy()
        else:
            sent_reps = sent_reps.data.numpy()
        unpadded_sent_reps = []
        for i, num_sents in enumerate(doc_abs_lens):
            # encoding_dim x num_sents
            upsr = sent_reps[i, :, :num_sents]
            # return: # num_sents x encoding_dim
            unpadded_sent_reps.append(upsr.transpose(1, 0))
        ret_dict = {
            'sent_reps': unpadded_sent_reps,
        }
        return ret_dict
    
    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
        {
            'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'query_abs_lens': list(int); Number of sentences in query abs.
            'query_sep_idxs': LongTensor; Indices of the sep tokens to get sent reps,
                flattened and indices adjusted to index the one dimensional token reps.
            'pos_abs_lens': list(int);
            'pos_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from positive abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'pos_sep_idxs': LongTensor; Indices of the sep tokens to get sent reps,
                flattened and indices adjusted to index the one dimensional token reps.
            'neg_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'neg_abs_lens': list(int);
            'neg_sep_idxs': LongTensor; Indices of the sep tokens to get sent reps,
                flattened and indices adjusted to index the one dimensional token reps.
        }
        :return: loss_val; torch Variable.
        """
        qbert_batch, qabs_lens = batch_rank['query_bert_batch'], batch_rank['query_abs_lens']
        pbert_batch, pabs_lens = batch_rank['pos_bert_batch'], batch_rank['pos_abs_lens']
        query_sepsi, pos_sepsi = batch_rank['query_sep_idxs'], batch_rank['pos_sep_idxs']
        # Get the representations from the model.
        q_sent_reps = self.partial_forward(bert_batch=qbert_batch, abs_lens=qabs_lens, sent_sep_idxs=query_sepsi)
        p_sent_reps = self.partial_forward(bert_batch=pbert_batch, abs_lens=pabs_lens, sent_sep_idxs=pos_sepsi)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch, nabs_lens = batch_rank['neg_bert_batch'], batch_rank['neg_abs_lens']
            neg_sepsi = batch_rank['neg_sep_idxs']
            n_sent_reps = self.partial_forward(bert_batch=nbert_batch, abs_lens=nabs_lens, sent_sep_idxs=neg_sepsi)
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            random_idxs = torch.randperm(p_sent_reps.size()[0])
            n_sent_reps = p_sent_reps[random_idxs]
            nabs_lens = [pabs_lens[i] for i in random_idxs.tolist()]
            
        # Bundle the lengths with the embeds so the similarity
        # function can use the lens for masking.
        query_sents = rep_len_tup(embed=q_sent_reps, abs_lens=qabs_lens)
        pos_sents = rep_len_tup(embed=p_sent_reps, abs_lens=pabs_lens)
        neg_sents = rep_len_tup(embed=n_sent_reps, abs_lens=nabs_lens)
        
        loss_val = self.criterion(query_sents, pos_sents, neg_sents)
        return loss_val

    def partial_forward(self, bert_batch, abs_lens, sent_sep_idxs):
        """
        Pass a batch of sentences through BERT and read off sentence
        representations based on SEP idxs.
        :return:
            sent_reps: batch_size x encoding_dim x num_sents
        """
        # batch_size x num_sents x encoding_dim
        doc_cls_reps, sent_reps = self.sent_reps_bert(bert_batch=bert_batch, num_sents=abs_lens,
                                                      flat_sep_idxs=sent_sep_idxs)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        # Similarity function expects: batch_size x encoding_dim x q_max_sents;
        return sent_reps.permute(0, 2, 1)

    def sent_reps_bert(self, bert_batch, flat_sep_idxs, num_sents):
        """
        Pass the concated abstract through BERT, and read off [SEP] token reps to get sentence reps,
        and weighted combine across layers.
        :param bert_batch: dict('tokid_tt', 'seg_tt', 'attnmask_tt', 'seq_lens'); items to use for getting BERT
            representations. The sentence mapped to BERT vocab and appropriately padded.
        :param flat_sep_idxs: LongTensor; Indices of the sep tokens to get sent reps,
            flattened and indices adjusted to index the one dimensional token reps.
        :param num_sents: list(int); number of sentences in each example in the batch passed.
        :return:
            doc_cls_reps: FloatTensor [batch_size x bert_encoding_dim]
            sent_reps: FloatTensor [batch_size x max_num_sents x bert_encoding_dim]
        """
        seq_lens = bert_batch['seq_lens']
        batch_size, max_seq_len = len(seq_lens), max(seq_lens)
        max_sents = max(num_sents)
        pad_mask = np.zeros((batch_size, max_sents, self.bert_encoding_dim))
        for i, num in enumerate(num_sents):
            pad_mask[i, :num, :] = 1.0
        pad_mask = Variable(torch.FloatTensor(pad_mask))
        flat_sep_idxs = Variable(flat_sep_idxs)
        if torch.cuda.is_available():
            pad_mask = pad_mask.cuda()
            flat_sep_idxs = flat_sep_idxs.cuda()
        tokid_tt, seg_tt, attnmask_tt = bert_batch['tokid_tt'], bert_batch['seg_tt'], bert_batch['attnmask_tt']
        if torch.cuda.is_available():
            tokid_tt, seg_tt, attnmask_tt = tokid_tt.cuda(), seg_tt.cuda(), attnmask_tt.cuda()
        # Pass input through BERT and return all layer hidden outputs.
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        # Weighted combine the hidden_states which is a list of [bs x max_seq_len x bert_encoding_dim]
        # with as many tensors as layers + 1 input layer.
        # hs_stacked = torch.stack(model_outputs.hidden_states, dim=3)
        # weighted_sum_hs = self.bert_layer_weights(hs_stacked)  # [bs x max_seq_len x bert_encoding_dim x 1]
        # weighted_sum_hs = torch.squeeze(weighted_sum_hs, dim=3)
        # Read of CLS token as document representation.
        doc_cls_reps = final_hidden_state[:, 0, :]
        doc_cls_reps = doc_cls_reps.squeeze()
        # Read off SEP token reps to get sentence reps.
        flattoks = final_hidden_state.view(batch_size*max_seq_len, self.bert_encoding_dim)
        sep_tok_reps = torch.index_select(flattoks, 0, flat_sep_idxs)
        sent_reps = sep_tok_reps.view(batch_size, max_sents, self.bert_encoding_dim)
        sent_reps = sent_reps * pad_mask
        return doc_cls_reps, sent_reps


class WordSentAlignBiEnc(SentAlignBiEnc):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Compute pairwise sentence similarities for query and candidate.
    - Maximize maximum similarity of anchor and positive.
    """
    
    def __init__(self, model_hparams, bert_config=None):
        """
        :param model_hparams: dict(string:int); model hyperparams.
            num_code_vecs: int; number of code vectors to disentangle into.
                The number of facets.
            num_tf_heads: int; number of heads in the context transformer.
        :param bert_config: transformers.configuration_bert.BertConfig; bert
            hyperparam instance.
        """
        torch.nn.Module.__init__(self)
        self.bert_config = bert_config
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        if 'consent-base-pt-layer' in model_hparams:
            self.bert_encoder = AutoModel.from_pretrained(model_hparams['consent-base-pt-layer'])
        else:
            self.bert_encoder = AutoModel.from_pretrained(model_hparams['base-pt-layer'])
        self.bert_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        try:
            if not model_hparams['fine_tune']:
                for param in self.bert_encoder.base_model.parameters():
                    param.requires_grad = False
        except KeyError:
            logging.info('\'fine_tune\' attribute missing in config; Not fine-tuning the base encoder.')
            fine_tune = False
            if not fine_tune:
                for param in self.bert_encoder.base_model.parameters():
                    param.requires_grad = False
        self.score_agg_type = model_hparams['score_aggregation']
        if self.score_agg_type == 'l2lse':
            self.dist_function = pair_dist.allpair_masked_dist_l2lse
        elif self.score_agg_type == 'l2max':
            self.dist_function = pair_dist.allpair_masked_dist_l2max
        elif self.score_agg_type == 'l2top2':
            self.dist_function = pair_dist.allpair_masked_dist_l2topk
        elif self.score_agg_type == 'l2sum':
            self.dist_function = pair_dist.allpair_masked_dist_l2sum
        elif self.score_agg_type == 'l2maxsum':
            self.dist_function = pair_dist.allpair_masked_dist_l2maxsum
        elif self.score_agg_type == 'l2wasserstein':
            ot_distance = pair_dist.AllPairMaskedWasserstein(model_hparams)
            self.dist_function = ot_distance.compute_distance
        elif self.score_agg_type == 'l2attention':
            ot_distance = pair_dist.AllPairMaskedAttention(model_hparams)
            self.dist_function = ot_distance.compute_distance
        else:
            raise ValueError(f'Unknown aggregation: {self.score_agg_type}')
        # Not using the random weights because they'll spoil initial alignments.
        # self.bert_layer_weights = gl.SoftmaxMixLayers(in_features=self.bert_layer_count, out_features=1, bias=False)
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function,
                                                          margin=1.0, reduction='sum')
        self.cd_svalue_l1_prop = float(model_hparams.get('cd_svalue_l1_prop', 0.0))
        self.sent_loss_prop = 1.0
        self.abs_loss_prop = 0.0

    def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Treat a flattened set of user query docs sentence reps as
            a single doc with many query reps.
        - Pad candidate reps to max length.
        - Compute scores and return.
        query_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array})
        cand_encode_ret_dict: list({'sent_reps': numpy.array, 'doc_cls_reps': numpy.array})
        """
        # Flatten the query abstracts sentences into a single doc with many sentences
        queries_sent_reps = [d['sent_reps'] for d in query_encode_ret_dict]
        query_abs_lens = [r.shape[0] for r in queries_sent_reps]
        numq_sents = sum(query_abs_lens)
        flat_query_sent_reps = np.zeros((numq_sents, self.bert_encoding_dim))
        start_idx = 0
        for ex_num_sents, ex_reps in zip(query_abs_lens, queries_sent_reps):
            flat_query_sent_reps[start_idx:start_idx+ex_num_sents, :] = ex_reps
            start_idx += ex_num_sents
        # Pack candidate representations as padded tensors.
        cand_sent_reps = [d['sent_reps'] for d in cand_encode_ret_dicts]
        batch_size = len(cand_sent_reps)
        cand_lens = [r.shape[0] for r in cand_sent_reps]
        flat_query_lens = [numq_sents]*batch_size
        cmax_sents = max(cand_lens)
        padded_cand_sent_reps = np.zeros((batch_size, cmax_sents, self.bert_encoding_dim))
        repeated_query_sent_reps = np.zeros((batch_size, numq_sents, self.bert_encoding_dim))
        for bi, ex_reps in enumerate(cand_sent_reps):
            padded_cand_sent_reps[bi, :cand_lens[bi], :] = ex_reps
            # Repeat the query sents.
            repeated_query_sent_reps[bi, :numq_sents, :] = flat_query_sent_reps
        repeated_query_sent_reps = Variable(torch.FloatTensor(repeated_query_sent_reps))
        padded_cand_sent_reps = Variable(torch.FloatTensor(padded_cand_sent_reps))
        if torch.cuda.is_available():
            repeated_query_sent_reps = repeated_query_sent_reps.cuda()
            padded_cand_sent_reps = padded_cand_sent_reps.cuda()
        # Compute scores as at train time.
        qt = rep_len_tup(embed=repeated_query_sent_reps.permute(0, 2, 1), abs_lens=flat_query_lens)
        ct = rep_len_tup(embed=padded_cand_sent_reps.permute(0, 2, 1), abs_lens=cand_lens)
        if self.score_agg_type in {'l2lse'}:
            batch_sent_sims, pair_sims = pair_dist.allpair_masked_dist_l2max(query=qt, cand=ct, return_pair_sims=True)
        else:
            batch_sent_sims, pair_sims = self.dist_function(query=qt, cand=ct, return_pair_sims=True)
        # In the case of WordSentAbsSupAlignBiEnc which also uses this function if sent_loss_prop is zero
        # use the supervised sent prop instead.
        try:
            sent_loss_prop = max(self.sent_loss_prop, self.sentsup_loss_prop)
        except AttributeError:
            sent_loss_prop = self.sent_loss_prop
        batch_scores = sent_loss_prop*batch_sent_sims
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            batch_scores = batch_scores.cpu().data.numpy()
            if isinstance(pair_sims, list):
                pair_sims = [t.cpu().data.numpy() for t in pair_sims]
            else:
                pair_sims = pair_sims.cpu().data.numpy()
        else:
            batch_scores = batch_scores.data.numpy()
            if isinstance(pair_sims, list):
                pair_sims = [t.data.numpy() for t in pair_sims]
            else:
                pair_sims = pair_sims.data.numpy()
        unpadded_pair_sm = []
        for i, (clen, qlen) in enumerate(zip(cand_lens, flat_query_lens)):
            # Happens in the case of wasserstein distance.
            if len(pair_sims) == 5:
                upsm = [pair_sims[0][i, :qlen], pair_sims[1][i, :clen],
                        pair_sims[2][i, :qlen, :clen], pair_sims[3][i, :qlen, :clen],
                        pair_sims[4][i, :qlen, :clen]]
            # Happens in the case of attention distance.
            elif len(pair_sims) == 3:
                upsm = [pair_sims[0][i, :qlen, :clen], pair_sims[1][i, :qlen, :clen],
                        pair_sims[2][i, :qlen, :clen]]
            else:
                # encoding_dim x num_sents
                upsm = pair_sims[i, :qlen, :clen]
            # return: # num_sents x encoding_dim
            unpadded_pair_sm.append(upsm)
    
        ret_dict = {
            'batch_scores': batch_scores,
            'pair_scores': unpadded_pair_sm
        }
        return ret_dict

    def caching_encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
        doc_query_senttoki = batch_dict['senttok_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        doc_cls_reps, sent_reps = self.partial_forward(bert_batch=doc_bert_batch, abs_lens=doc_abs_lens,
                                                       sent_tok_idxs=doc_query_senttoki)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            sent_reps = sent_reps.cpu().data.numpy()
            doc_cls_reps = doc_cls_reps.cpu().data.numpy()
        else:
            sent_reps = sent_reps.data.numpy()
            doc_cls_reps = doc_cls_reps.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i, num_sents in enumerate(doc_abs_lens):
            # encoding_dim x num_sents
            upsr = sent_reps[i, :, :num_sents]
            # return: # num_sents x encoding_dim
            batch_reps.append({'doc_cls_reps': doc_cls_reps[i, :],
                               'sent_reps': upsr.transpose(1, 0)})
        return batch_reps

    def encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        doc_bert_batch, doc_abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
        doc_query_senttoki = batch_dict['senttok_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        _, sent_reps = self.partial_forward(bert_batch=doc_bert_batch, abs_lens=doc_abs_lens,
                                            sent_tok_idxs=doc_query_senttoki)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            sent_reps = sent_reps.cpu().data.numpy()
        else:
            sent_reps = sent_reps.data.numpy()
        unpadded_sent_reps = []
        for i, num_sents in enumerate(doc_abs_lens):
            # encoding_dim x num_sents
            upsr = sent_reps[i, :, :num_sents]
            # return: # num_sents x encoding_dim
            unpadded_sent_reps.append(upsr.transpose(1, 0))
        ret_dict = {
            'sent_reps': unpadded_sent_reps,
        }
        return ret_dict
    
    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
        {
            'query_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'query_abs_lens': list(int); Number of sentences in query abs.
            'query_senttoki': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
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
        qbert_batch, qabs_lens = batch_rank['query_bert_batch'], batch_rank['query_abs_lens']
        pbert_batch, pabs_lens = batch_rank['pos_bert_batch'], batch_rank['pos_abs_lens']
        query_senttoki, pos_senttoki = batch_rank['query_senttok_idxs'], batch_rank['pos_senttok_idxs']
        # Get the representations from the model.
        _, q_sent_reps = self.partial_forward(bert_batch=qbert_batch, abs_lens=qabs_lens, sent_tok_idxs=query_senttoki)
        _, p_sent_reps = self.partial_forward(bert_batch=pbert_batch, abs_lens=pabs_lens, sent_tok_idxs=pos_senttoki)
        # Happens when running on the dev set.
        if 'neg_bert_batch' in batch_rank:
            nbert_batch, nabs_lens = batch_rank['neg_bert_batch'], batch_rank['neg_abs_lens']
            neg_senttoki = batch_rank['neg_senttok_idxs']
            _, n_sent_reps = self.partial_forward(bert_batch=nbert_batch, abs_lens=nabs_lens, sent_tok_idxs=neg_senttoki)
            # Bundle the lengths with the embeds so the similarity
            # function can use the lens for masking.
            query_sents = rep_len_tup(embed=q_sent_reps, abs_lens=qabs_lens)
            pos_sents = rep_len_tup(embed=p_sent_reps, abs_lens=pabs_lens)
            neg_sents = rep_len_tup(embed=n_sent_reps, abs_lens=nabs_lens)
    
            loss_val = self.criterion(query_sents, pos_sents, neg_sents)
            return loss_val
        else:
            # Use a shuffled set of positives as the negatives. -- in-batch negatives.
            random_idxs = torch.randperm(p_sent_reps.size()[0])
            n_sent_reps = p_sent_reps[random_idxs]
            nabs_lens = [pabs_lens[i] for i in random_idxs.tolist()]
            # Bundle the lengths with the embeds so the similarity
            # function can use the lens for masking.
            query_sents = rep_len_tup(embed=q_sent_reps, abs_lens=qabs_lens)
            pos_sents = rep_len_tup(embed=p_sent_reps, abs_lens=pabs_lens)
            neg_sents = rep_len_tup(embed=n_sent_reps, abs_lens=nabs_lens)
            
            loss_val = self.criterion(query_sents, pos_sents, neg_sents)
            # If asked to regularize the cross doc singular values, do so to make them more sparse.
            if self.cd_svalue_l1_prop > 0:
                # Pad values will be zeros.
                pair_sims = -1*torch.cdist(q_sent_reps.permute(0, 2, 1), p_sent_reps.permute(0, 2, 1))
                _, svalues, _ = torch.linalg.svd(pair_sims)
                if len(svalues.size()) < 2:
                    svalues = svalues.unsqueeze(dim=0)
                svalue_norm = torch.linalg.norm(svalues, ord=1, dim=1)
                svalue_reg = torch.sum(svalue_norm)
                loss_val += self.cd_svalue_l1_prop * svalue_reg
            return loss_val

    def partial_forward(self, bert_batch, abs_lens, sent_tok_idxs):
        """
        Pass a batch of sentences through BERT and read off sentence
        representations based on SEP idxs.
        :return:
            sent_reps: batch_size x encoding_dim x num_sents
        """
        # batch_size x num_sents x encoding_dim
        doc_cls_reps, sent_reps = self.sent_reps_bert(bert_batch=bert_batch, num_sents=abs_lens,
                                                      batch_senttok_idxs=sent_tok_idxs)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        if len(doc_cls_reps.size()) == 1:
            doc_cls_reps = doc_cls_reps.unsqueeze(0)
        # Similarity function expects: batch_size x encoding_dim x q_max_sents;
        return doc_cls_reps, sent_reps.permute(0, 2, 1)
    
    def sent_reps_bert(self, bert_batch, batch_senttok_idxs, num_sents):
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
        model_outputs = self.bert_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
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
            cur_sent_reps = torch.sum(sent_tokens, dim=1)/\
                            torch.count_nonzero(sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
            sent_reps.append(cur_sent_reps.unsqueeze(dim=1))
        # batch_size x max_sents x encoding_dim
        sent_reps = torch.cat(sent_reps, dim=1)
        return doc_cls_reps, sent_reps
