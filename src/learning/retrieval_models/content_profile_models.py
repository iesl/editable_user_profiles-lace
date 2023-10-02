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


def load_aspire_model(expanded_model_name, get_whole_state_dict=False):
    """
    Given the name of a model to load, load it and return its
    base bert encoder.
    :param expanded_model_name: string of the form <dataset>/<model_name>/<run_name> which is part
        of the path to a trained model.
    """
    if 'gypsum' in os.environ['CUR_PROJ_DIR']:  # This is running on unity (in interactive mode)
        trained_model_path = os.path.join('/gypsum/work1/mccallum/smysore/2021-ai2-scisim', 'model_runs', expanded_model_name)
    else:  # This is running on gypsum.
        trained_model_path = os.path.join(os.environ['CUR_PROJ_DIR'], 'model_runs', expanded_model_name)
    model_name = expanded_model_name.split('/')[1]
    with codecs.open(os.path.join(trained_model_path, 'run_info.json'), 'r', 'utf-8') as fp:
        run_info = json.load(fp)
        all_hparams = run_info['all_hparams']
        # Init model:
    if model_name in {'myspecter', 'cospecter'}:
        model = MySPECTER(model_hparams=all_hparams)
    elif model_name in {'miswordbienc'}:
        model = WordSentAlignBiEnc(model_hparams=all_hparams)
    model_fname = os.path.join(trained_model_path, 'model_{:s}.pt'.format('cur_best'))
    if get_whole_state_dict:
        print(f'Returned model: {trained_model_path}')
        return torch.load(model_fname)
    else:
        model.load_state_dict(torch.load(model_fname))
        print(f'Loaded model: {trained_model_path}')
        return model.bert_encoder


def load_upsentconsent_model(expanded_model_name):
    """
    Load a upsentconsent models base bert encoder. This is used for experimenting with a kind
    of oracle model which uses the sentence encoder trained separately with the keyphrase clustering
    mechanism.
    :param expanded_model_name: string of the form <dataset>/<model_name>/<run_name> which is part
        of the path to a trained model.
    """
    if 'gypsum' in os.environ['CUR_PROJ_DIR']:  # This is running on unity (in interactive mode)
        trained_model_path = os.path.join('/gypsum/work1/mccallum/smysore/2021-ai2-scisim', 'model_runs', expanded_model_name)
    else:  # This is running on gypsum.
        trained_model_path = os.path.join(os.environ['CUR_PROJ_DIR'], 'model_runs', expanded_model_name)
    model_name = expanded_model_name.split('/')[1]
    assert(model_name == 'upsentconsent')
    with codecs.open(os.path.join(trained_model_path, 'run_info.json'), 'r', 'utf-8') as fp:
        run_info = json.load(fp)
        all_hparams = run_info['all_hparams']
    model = UPSentAspire(model_hparams=all_hparams)
    model_fname = os.path.join(trained_model_path, 'model_{:s}.pt'.format('cur_best'))
    model.load_state_dict(torch.load(model_fname))
    print(f'Loaded model: {trained_model_path}')
    return model.sent_encoder


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
        if 'consent-base-pt-layer' in model_hparams:  # if this is passed get all model params from there.
            self.load_state_dict(load_aspire_model(expanded_model_name=model_hparams['consent-base-pt-layer'],
                                                   get_whole_state_dict=True))

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
            # This is done in the case of oracle experiments.
            if 'upsentconsent' in model_hparams['consent-base-pt-layer']:
                self.bert_encoder = load_upsentconsent_model(expanded_model_name=model_hparams['consent-base-pt-layer'])
            else:
                self.bert_encoder = load_aspire_model(expanded_model_name=model_hparams['consent-base-pt-layer'])
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


# todo: This is a hack. Fix this with some better import structure to avoid circular imports.

def load_kpenc_model(expanded_model_name):
    """
    Given the name of a model to load, load it and return its
    base bert encoder.
    :param expanded_model_name: string of the form <dataset>/<model_name>/<run_name> which is part
        of the path to a trained model.
    """
    trained_model_path = os.path.join(os.environ['CUR_PROJ_DIR'], 'model_runs', expanded_model_name)
    with codecs.open(os.path.join(trained_model_path, 'run_info.json'), 'r', 'utf-8') as fp:
        run_info = json.load(fp)
        all_hparams = run_info['all_hparams']
    
    kp_encoder = AutoModel.from_pretrained(all_hparams['kp-base-pt-layer'])
    model_fname = os.path.join(trained_model_path, 'kp_encoder_{:s}.pt'.format('cur_best'))
    kp_encoder.load_state_dict(torch.load(model_fname))
    return kp_encoder


class UPNamedFAspireKP(nn.Module):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Cluster user sentences into keyphrases.
    - Recompute keyphrases in terms of sentences.
    - Compute relevance in terms of keyphrase clustered sentences.
    """
    
    def __init__(self, model_hparams):
        """
        :param model_hparams: dict(string:int); model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        if 'upsentconsent' in model_hparams['consent-base-pt-layer']:
            self.sent_encoder = load_upsentconsent_model(expanded_model_name=model_hparams['consent-base-pt-layer'])
        else:
            self.sent_encoder = load_aspire_model(expanded_model_name=model_hparams['consent-base-pt-layer'])
        self.sent_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['sc_fine_tune']:
            for param in self.sent_encoder.base_model.parameters():
                param.requires_grad = False
        self.kp_encoder = load_kpenc_model(expanded_model_name=model_hparams['kp-base-pt-layer'])
        if not model_hparams['kp_fine_tune']:
            for param in self.kp_encoder.base_model.parameters():
                param.requires_grad = False
        self.score_agg_type = model_hparams['score_aggregation']
        self.ot_clustering = pair_dist.AllPairMaskedWassersteinCl(model_hparams)
        if self.score_agg_type == 'l2wasserstein':
            ot_distance = pair_dist.AllPairMaskedWasserstein(model_hparams)
            self.dist_function = ot_distance.compute_distance
        else:
            raise ValueError(f'Unknown aggregation: {self.score_agg_type}')
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function,
                                                          margin=1.0, reduction='sum')
        self.sent_entropy_lossprop = model_hparams.get('sent_entropy_lossprop', 0)
        self.kp_entropy_lossprop = model_hparams.get('kp_entropy_lossprop', 0)
        self.kp_align_lossprop = model_hparams.get('kp_align_lossprop', 0)
        self.candkp_search_lossprop = model_hparams.get('candkp_search_lossprop', 0)
        # self.criterion_entropy = loss.EntropyLossProb(size_average=False)
    
    def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Treat a flattened set of user query docs sentence reps as
            a single doc with many query reps.
        - Pad candidate reps to max length.
        - Compute scores and return.
        query_encode_ret_dict: list({'sent_reps': numpy.array})
        cand_encode_ret_dict: list({'sent_reps': numpy.array})
        """
        # Flatten the query abstracts sentences into a single doc with many sentences
        uq_sent_reps = [d['sent_reps'] for d in query_encode_ret_dict]
        uq_abs_lens = [r.shape[0] for r in uq_sent_reps]
        numq_sents = sum(uq_abs_lens)
        flat_query_sent_reps = np.zeros((numq_sents, self.bert_encoding_dim))
        start_idx = 0
        for ex_num_sents, ex_reps in zip(uq_abs_lens, uq_sent_reps):
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
        qt = pair_dist.rep_len_tup(embed=repeated_query_sent_reps.permute(0, 2, 1), abs_lens=flat_query_lens)
        ct = pair_dist.rep_len_tup(embed=padded_cand_sent_reps.permute(0, 2, 1), abs_lens=cand_lens)
        if self.score_agg_type in {'l2lse'}:
            batch_scores, pair_sims = pair_dist.allpair_masked_dist_l2max(query=qt, cand=ct, return_pair_sims=True)
        else:
            batch_scores, pair_sims = self.dist_function(query=qt, cand=ct, return_pair_sims=True)
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
    
    def user_caching_cluster(self, sent_kp_reps):
        """
        Given a set of sentences and keyphrases cluster the sentences into
        the keyphrases.
        - In the case of a user treat all the sentences in the batch as a single
            set of sentences, and only use unique keyphrases.
        param sent_kp_reps: list(dict); of the kind returned by caching_encode.
            all for one user.
        """
        # Flatten the query abstracts sentences into a single doc with many sentences
        uq_sent_reps = [d['sent_reps'] for d in sent_kp_reps]
        uq_abs_lens = [r.shape[0] for r in uq_sent_reps]
        numq_sents = sum(uq_abs_lens)
        flat_query_sent_reps = np.vstack(uq_sent_reps)
        assert(flat_query_sent_reps.shape[0] == numq_sents)
        # Similarly also flatten the keyphrases for the user into a unique set.
        unique_kp_reps = []
        unique_kp_li = []
        uniq_kps = set()
        for d in sent_kp_reps:
            for kp, kp_rep in d['kp_reps'].items():
                if kp in uniq_kps:
                    continue
                uniq_kps.add(kp)
                unique_kp_li.append(kp)
                unique_kp_reps.append(kp_rep)
        unique_kp_reps = np.vstack(unique_kp_reps)
        assert(unique_kp_reps.shape[0] == len(uniq_kps))
        sent_reps = Variable(torch.FloatTensor(flat_query_sent_reps))
        kp_reps = Variable(torch.FloatTensor(unique_kp_reps))
        if torch.cuda.is_available():
            sent_reps = sent_reps.cuda()
            kp_reps = kp_reps.cuda()
        sent_reps = pair_dist.rep_len_tup(embed=sent_reps.unsqueeze(0).permute(0, 2, 1), abs_lens=[numq_sents])
        kp_reps = pair_dist.rep_len_tup(embed=kp_reps.unsqueeze(0).permute(0, 2, 1), abs_lens=[len(uniq_kps)])
        # 1 x dim x num_uniq_kps.
        weighted_sents, ret_items = self.get_kpreps_scluster(sent_reps=sent_reps, kp_reps=kp_reps)
        weighted_sents = weighted_sents.embed.permute(0, 2, 1)
        qd = ret_items[0]
        cd = ret_items[1]
        dists = ret_items[2]
        tplan = ret_items[3]
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            weighted_sents = weighted_sents.cpu().data.numpy()
            qd = qd.cpu().data.numpy()
            cd = cd.cpu().data.numpy()
            dists = dists.cpu().data.numpy()
            tplan = tplan.cpu().data.numpy()
        else:
            weighted_sents = weighted_sents.data.numpy()
            qd = qd.data.numpy()
            cd = cd.data.numpy()
            dists = dists.data.numpy()
            tplan = tplan.data.numpy()
        upsr = weighted_sents[0, :, :]
        tplan = tplan[0, :, :]
        dists = dists[0, :, :]
        batch_reps = [{'sent_reps': upsr, 's2k_tplan': (qd, cd, dists, tplan), 'uniq_kps': unique_kp_li}]
        return batch_reps
    
    def cand_caching_cluster(self, sent_kp_reps):
        """
        Given a set of sentences and keyphrases cluster the sentences into
        the keyphrases.
        - In the case of a user treat all the sentences in the batch as a single
            set of sentences, and only use unique keyphrases.
        - In the case of a candidate treat all the documents passed as a sep and
            use all the passed keyphrases for the document.
        param sent_kp_reps: list(dict); of the kind returned by caching_encode.
        """
        # Get the cand abstract sents
        cand_sent_reps = [d['sent_reps'] for d in sent_kp_reps]
        batch_size = len(cand_sent_reps)
        cand_sent_lens = [r.shape[0] for r in cand_sent_reps]
        cmax_sents = max(cand_sent_lens)
        # Get the cand keyphrase reps.
        cand_kp_reps = []
        cand_kps = []
        for d in sent_kp_reps:
            doc_kp_reps = np.vstack(list(d['kp_reps'].values()))
            cand_kp_reps.append(doc_kp_reps)
            cand_kps.append(list(d['kp_reps'].keys()))
        cand_kp_lens = [r.shape[0] for r in cand_kp_reps]
        cmax_kps = max(cand_kp_lens)
        padded_cand_sent_reps = np.zeros((batch_size, cmax_sents, self.bert_encoding_dim))
        padded_cand_kp_reps = np.zeros((batch_size, cmax_kps, self.bert_encoding_dim))
        for bi, (dsent_reps, dkp_reps) in enumerate(zip(cand_sent_reps, cand_kp_reps)):
            padded_cand_sent_reps[bi, :cand_sent_lens[bi], :] = dsent_reps
            padded_cand_kp_reps[bi, :cand_kp_lens[bi], :] = dkp_reps
        # Compute clustering.
        sent_reps = Variable(torch.FloatTensor(padded_cand_sent_reps))
        kp_reps = Variable(torch.FloatTensor(padded_cand_kp_reps))
        if torch.cuda.is_available():
            sent_reps = sent_reps.cuda()
            kp_reps = kp_reps.cuda()
        sent_reps = pair_dist.rep_len_tup(embed=sent_reps.permute(0, 2, 1), abs_lens=cand_sent_lens)
        kp_reps = pair_dist.rep_len_tup(embed=kp_reps.permute(0, 2, 1), abs_lens=cand_kp_lens)
        # batch_size x dim x num_uniq_kps.
        weighted_sents, retlist = self.get_kpreps_scluster(sent_reps=sent_reps, kp_reps=kp_reps)
        tplan = retlist[3]
        weighted_sents = weighted_sents.embed.permute(0, 2, 1)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            weighted_sents = weighted_sents.cpu().data.numpy()
            tplan = tplan.cpu().data.numpy()
        else:
            weighted_sents = weighted_sents.data.numpy()
            tplan = tplan.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i, (num_sents, num_kps) in enumerate(zip(cand_sent_lens, cand_kp_lens)):
            # num_sents x encoding_dim
            upsr = weighted_sents[i, :num_kps, :]
            ctplan = tplan[i, :num_sents, :num_kps]
            ckps = cand_kps[i]
            # return: # num_sents x encoding_dim
            batch_reps.append({'sent_reps': upsr, 's2k_tplan': ctplan, 'ckps': ckps})
        return batch_reps
    
    def caching_encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        abs_bert_batch, abs_lens = batch_dict['abs_bert_batch'], batch_dict['abs_lens']
        abs_senttoki = batch_dict['abs_senttok_idxs']
        batch_abs_kps = batch_dict['batch_abs_kps']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        _, sent_reps = self.partial_forward(bert_batch=abs_bert_batch, abs_lens=abs_lens,
                                            sent_tok_idxs=abs_senttoki)
        # num_abs*num_abs_kps x encoding dim
        batch_kp_reps = self.sent_reps_bert(bert_batch=batch_dict['kp_bert_batch'])
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            sent_reps = sent_reps.cpu().data.numpy()
            batch_kp_reps = batch_kp_reps.cpu().data.numpy()
        else:
            sent_reps = sent_reps.data.numpy()
            batch_kp_reps = batch_kp_reps.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        start_kpi = 0
        for i, (num_sents, abs_kps) in enumerate(zip(abs_lens, batch_abs_kps)):
            # num_sents x encoding_dim
            upsr = sent_reps[i, :num_sents, :]
            # num_kps x encoding_dim
            kp_reps = batch_kp_reps[start_kpi:start_kpi+len(abs_kps), :]
            kp_rep_dict = collections.OrderedDict([(kp, kp_reps[k, :]) for k, kp in enumerate(abs_kps)])
            start_kpi += len(abs_kps)
            # return: # num_sents x encoding_dim
            batch_reps.append({'sent_reps': upsr,
                               'kp_reps': kp_rep_dict})
        return batch_reps
    
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
            'abs_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'abs_lens': list(int); Number of sentences in query abs.
            'abs_senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
            'kp_bert_batch': The batch which BERT inputs with keyphrases;
                Tokenized and int mapped sentences and other inputs to BERT.
            Stuff below indexes into the abstract sentence reps and keyphrase reps
            to create an effective batch of users-candidates-negatives.
            'user_papers': idx_len_tup(flat_seq_idxs, seq_lens),
            'user_kps': idx_len_tup(flat_seq_idxs, seq_lens),
            'cand_paper': idx_len_tup(flat_seq_idxs, seq_lens),
            'cand_kps': idx_len_tup(flat_seq_idxs, seq_lens),
            'neg_paper': idx_len_tup(flat_seq_idxs, seq_lens),
            'neg_kps': idx_len_tup(flat_seq_idxs, seq_lens)
        }
        :return: loss_val; torch Variable.
        """
        # Get the abstract sentence representations from the model.
        _, abs_sent_reps = self.partial_forward(bert_batch=batch_rank['abs_bert_batch'],
                                                abs_lens=batch_rank['abs_lens'],
                                                sent_tok_idxs=batch_rank['abs_senttok_idxs'])
        # Get the keyphrase reps from the model.
        keyphrase_reps = self.sent_reps_bert(bert_batch=batch_rank['kp_bert_batch'])
        
        # Build the effective batch from the sentence and kp reps.
        userp_sents, userp_kps, cand_sents, cand_kps, negcand_sents, negcand_kps = \
            self.create_effective_batch(batch_rank, abs_sent_reps, keyphrase_reps)
        userp_kpsents, user_ret_list = self.get_kpreps_scluster(userp_sents, userp_kps)
        cand_kpsents, cand_ret_list = self.get_kpreps_scluster(cand_sents, cand_kps)
        negcand_kpsents, _ = self.get_kpreps_scluster(negcand_sents, negcand_kps)
        
        loss_val = self.criterion(userp_kpsents, cand_kpsents, negcand_kpsents)
        
        # Compute a loss by aligning the keyphrases - in a bid to prevent mode collapses.
        if self.kp_align_lossprop > 0 and self.training:
            loss_val += self.kp_align_lossprop*self.criterion(userp_kps, cand_kps, negcand_kps)
        
        # Compute a search loss by aligning the sentences of the candidate to its keyphrases
        # - in a bid to prevent mode collapses.
        if self.candkp_search_lossprop > 0 and self.training:
            loss_val += self.candkp_search_lossprop*self.criterion(cand_kps, cand_sents, negcand_sents)
        
        # If the sentence and keyphrase entropy regularizations arent zero then add them to the loss.
        # The goal here is to make the sentences more uniformly spread over the keyphrases; since the distrs
        # are computed from the pairwise distances the goal is to make the distrs more flat so all the below
        # entropies are negated.
        if self.sent_entropy_lossprop > 0 and self.training:
            userp_sent_distr = user_ret_list[5]
            cand_sent_distr = cand_ret_list[5]
            loss_val -= self.sent_entropy_lossprop*self.criterion_entropy(userp_sent_distr)
            loss_val -= self.sent_entropy_lossprop*self.criterion_entropy(cand_sent_distr)
        if self.kp_entropy_lossprop > 0 and self.training:
            userp_kp_distr = user_ret_list[6]
            cand_kp_distr = cand_ret_list[6]
            loss_val -= self.kp_entropy_lossprop*self.criterion_entropy(userp_kp_distr)
            loss_val -= self.kp_entropy_lossprop*self.criterion_entropy(cand_kp_distr)
        return loss_val
    
    def get_kpreps_scluster(self, sent_reps, kp_reps):
        """
        Cluster the keyphrase reps in terms of clustered sentences.
        param sent_reps: namedtuple(
            embed: torch.tensor(ef_batch_size x dim x num_sents)
            abs_lens: list(int); number of sentences in every batch element.)
        param kp_reps: namedtuple(
            embed: torch.tensor(ef_batch_size x sim x num_kps)
            abs_lens: list(int); number of kps in every batch element.)
        """
        # Get the transport plan from sentence reps to keyphrase reps.
        _, ret_list = self.ot_clustering.compute_distance(query=sent_reps,
                                                          cand=kp_reps, return_pair_sims=True)
        # ef_batch_size x num_sents x num_kps
        tplan = ret_list[3]
        # Get the keyphrases in terms of the sentences: ef_batch_size x dim x num_kps
        weighted_sents = torch.bmm(sent_reps.embed, tplan)
        weighted_sents = pair_dist.rep_len_tup(embed=weighted_sents,
                                               abs_lens=kp_reps.abs_lens)
        return weighted_sents, ret_list
    
    def create_effective_batch(self, batch_rank, abs_sent_reps, keyphrase_reps):
        """
        Given the sentence reps for the whole batch:
        - Index into the sentence reps to get the positive examples (user-papers paired with
            a single other user paper) and the corresponding negative reps.
        - In getting the positive user-paper pairs, repeatedly sample different user papers
            to represent their profile and treat the other paper as a candidate paired
            with a negative.
        """
        num_abs, max_sents, _ = abs_sent_reps.size()
        num_kps, _ = keyphrase_reps.size()
        kp_pad = Variable(torch.zeros(1, self.bert_encoding_dim))
        sent_pad = Variable(torch.zeros(1, self.bert_encoding_dim))
        up_si, cand_si, neg_si = Variable(batch_rank['user_papers'].flat_seqi), \
                                 Variable(batch_rank['cand_paper'].flat_seqi), \
                                 Variable(batch_rank['neg_paper'].flat_seqi)
        up_kpi, cand_kpi, neg_kpi = Variable(batch_rank['user_kps'].flat_seqi), \
                                    Variable(batch_rank['cand_kps'].flat_seqi), \
                                    Variable(batch_rank['neg_kps'].flat_seqi)
        if torch.cuda.is_available():
            kp_pad, sent_pad = kp_pad.cuda(), sent_pad.cuda()
            up_si, cand_si, neg_si = up_si.cuda(), cand_si.cuda(), neg_si.cuda()
            up_kpi, cand_kpi, neg_kpi = up_kpi.cuda(), cand_kpi.cuda(), neg_kpi.cuda()
        # Flatten the reps and append the zero pad value.
        flat_abs_sent_reps = torch.cat((
            abs_sent_reps.view(num_abs*max_sents, self.bert_encoding_dim), sent_pad), 0)
        flat_kp_reps = torch.cat((keyphrase_reps, kp_pad), 0)
        # Index into sentences and kps.
        efbatch_size = len(batch_rank['user_papers'].seq_lens)
        max_user_sent = max(batch_rank['user_papers'].seq_lens)
        userp_sents = torch.index_select(
            flat_abs_sent_reps, 0, up_si).view(efbatch_size, max_user_sent, self.bert_encoding_dim)
        userp_sents = pair_dist.rep_len_tup(embed=userp_sents.permute(0, 2, 1),
                                            abs_lens=batch_rank['user_papers'].seq_lens)
        max_user_kps = max(batch_rank['user_kps'].seq_lens)
        userp_kps = torch.index_select(
            flat_kp_reps, 0, up_kpi).view(efbatch_size, max_user_kps, self.bert_encoding_dim)
        userp_kps = pair_dist.rep_len_tup(embed=userp_kps.permute(0, 2, 1),
                                          abs_lens=batch_rank['user_kps'].seq_lens)
        max_cand_sent = max(batch_rank['cand_paper'].seq_lens)
        cand_sents = torch.index_select(
            flat_abs_sent_reps, 0, cand_si).view(efbatch_size, max_cand_sent, self.bert_encoding_dim)
        cand_sents = pair_dist.rep_len_tup(embed=cand_sents.permute(0, 2, 1),
                                           abs_lens=batch_rank['cand_paper'].seq_lens)
        max_cand_kps = max(batch_rank['cand_kps'].seq_lens)
        cand_kps = torch.index_select(
            flat_kp_reps, 0, cand_kpi).view(efbatch_size, max_cand_kps, self.bert_encoding_dim)
        cand_kps = pair_dist.rep_len_tup(embed=cand_kps.permute(0, 2, 1),
                                         abs_lens=batch_rank['cand_kps'].seq_lens)
        max_negcand_sent = max(batch_rank['neg_paper'].seq_lens)
        negcand_sents = torch.index_select(
            flat_abs_sent_reps, 0, neg_si).view(efbatch_size, max_negcand_sent, self.bert_encoding_dim)
        negcand_sents = pair_dist.rep_len_tup(embed=negcand_sents.permute(0, 2, 1),
                                              abs_lens=batch_rank['neg_paper'].seq_lens)
        max_negcand_kps = max(batch_rank['neg_kps'].seq_lens)
        negcand_kps = torch.index_select(
            flat_kp_reps, 0, neg_kpi).view(efbatch_size, max_negcand_kps, self.bert_encoding_dim)
        negcand_kps = pair_dist.rep_len_tup(embed=negcand_kps.permute(0, 2, 1),
                                            abs_lens=batch_rank['neg_kps'].seq_lens)
        return userp_sents, userp_kps, cand_sents, cand_kps, negcand_sents, negcand_kps
    
    def partial_forward(self, bert_batch, abs_lens, sent_tok_idxs):
        """
        Pass a batch of sentences through BERT and get contextual sentence reps.
        :return:
            sent_reps: batch_size x num_sents x encoding_dim
        """
        # batch_size x num_sents x encoding_dim
        doc_cls_reps, sent_reps = self.con_sent_reps_bert(bert_batch=bert_batch, num_sents=abs_lens,
                                                          batch_senttok_idxs=sent_tok_idxs)
        if len(sent_reps.size()) == 2:
            sent_reps = sent_reps.unsqueeze(0)
        if len(doc_cls_reps.size()) == 1:
            doc_cls_reps = doc_cls_reps.unsqueeze(0)
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
        model_outputs = self.sent_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
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
    
    def sent_reps_bert(self, bert_batch):
        """
        Get representation for the string passed via bert by averaging
        token embeddings; string can be a sentence or a phrase.
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
        model_outputs = self.kp_encoder(tokid_tt, token_type_ids=seg_tt, attention_mask=attnmask_tt)
        final_hidden_state = model_outputs.last_hidden_state
        sent_tokens = final_hidden_state * sent_mask
        # The sent_masks non zero elements in one slice along embedding dim is the sentence length.
        sent_reps = torch.sum(sent_tokens, dim=1)/torch.count_nonzero(
            sent_mask[:, :, 0], dim=1).clamp(min=1).unsqueeze(dim=1)
        return sent_reps


class UPSentAspire(UPNamedFAspireKP):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Compute relevance in terms of user profile sentences.
    """
    
    def __init__(self, model_hparams):
        """
        :param model_hparams: dict(string:int); model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.sent_encoder = load_aspire_model(expanded_model_name=model_hparams['consent-base-pt-layer'])
        self.sent_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['sc_fine_tune']:
            for param in self.sent_encoder.base_model.parameters():
                param.requires_grad = False
        self.score_agg_type = model_hparams['score_aggregation']
        if self.score_agg_type == 'l2wasserstein':
            ot_distance = pair_dist.AllPairMaskedWasserstein(model_hparams)
            self.dist_function = ot_distance.compute_distance
        else:
            raise ValueError(f'Unknown aggregation: {self.score_agg_type}')
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function,
                                                          margin=1.0, reduction='sum')
    
    def caching_encode(self, batch_dict):
        """
        Function used at test time.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        abs_bert_batch, abs_lens = batch_dict['abs_bert_batch'], batch_dict['abs_lens']
        abs_senttoki = batch_dict['abs_senttok_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        _, sent_reps = self.partial_forward(bert_batch=abs_bert_batch, abs_lens=abs_lens,
                                            sent_tok_idxs=abs_senttoki)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            sent_reps = sent_reps.cpu().data.numpy()
        else:
            sent_reps = sent_reps.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i, num_sents in enumerate(abs_lens):
            # num_sents x encoding_dim
            upsr = sent_reps[i, :num_sents, :]
            # return: # num_sents x encoding_dim
            batch_reps.append({'sent_reps': upsr})
        return batch_reps
    
    def forward_rank(self, batch_rank):
        """
        Function used at training time.
        batch_dict: dict of the form:
        {
            'abs_bert_batch': dict(); The batch which BERT inputs with flattened and
                concated sentences from query abstracts; Tokenized and int mapped
                sentences and other inputs to BERT.
            'abs_lens': list(int); Number of sentences in query abs.
            'abs_senttok_idxs': list(list(list(int))); batch_size(num_abs_sents(
                    num_sent_tokens(ints)))
            Stuff below indexes into the abstract sentence reps and keyphrase reps
            to create an effective batch of users-candidates-negatives.
            'user_papers': idx_len_tup(flat_seq_idxs, seq_lens),
            'cand_paper': idx_len_tup(flat_seq_idxs, seq_lens),
            'neg_paper': idx_len_tup(flat_seq_idxs, seq_lens)
        }
        :return: loss_val; torch Variable.
        """
        # Get the abstract sentence representations from the model.
        _, abs_sent_reps = self.partial_forward(bert_batch=batch_rank['abs_bert_batch'],
                                                abs_lens=batch_rank['abs_lens'],
                                                sent_tok_idxs=batch_rank['abs_senttok_idxs'])
        
        # Build the effective batch from the sentence and kp reps.
        userp_sents, cand_sents, negcand_sents = \
            self.create_effective_batch(batch_rank, abs_sent_reps)
        # print(userp_sents.embed.shape)
        # print(cand_sents.embed.shape)
        # print(negcand_sents.embed.shape)
        loss_val = self.criterion(userp_sents, cand_sents, negcand_sents)
        return loss_val
    
    def create_effective_batch(self, batch_rank, abs_sent_reps):
        """
        Given the sentence reps for the whole batch:
        - Index into the sentence reps to get the positive examples (user-papers paired with
            a single other user paper) and the corresponding negative reps.
        - In getting the positive user-paper pairs, repeatedly sample different user papers
            to represent their profile and treat the other paper as a candidate paired
            with a negative.
        """
        num_abs, max_sents, _ = abs_sent_reps.size()
        sent_pad = Variable(torch.zeros(1, self.bert_encoding_dim))
        up_si, cand_si, neg_si = Variable(batch_rank['user_papers'].flat_seqi), \
                                 Variable(batch_rank['cand_paper'].flat_seqi), \
                                 Variable(batch_rank['neg_paper'].flat_seqi)
        if torch.cuda.is_available():
            sent_pad = sent_pad.cuda()
            up_si, cand_si, neg_si = up_si.cuda(), cand_si.cuda(), neg_si.cuda()
        # Flatten the reps and append the zero pad value.
        flat_abs_sent_reps = torch.cat((
            abs_sent_reps.view(num_abs*max_sents, self.bert_encoding_dim), sent_pad), 0)
        # print(flat_abs_sent_reps.size())
        # Index into sentences and kps.
        efbatch_size = len(batch_rank['user_papers'].seq_lens)
        max_user_sent = max(batch_rank['user_papers'].seq_lens)
        # print(up_si)
        userp_sents = torch.index_select(
            flat_abs_sent_reps, 0, up_si).view(efbatch_size, max_user_sent, self.bert_encoding_dim)
        userp_sents = pair_dist.rep_len_tup(embed=userp_sents.permute(0, 2, 1),
                                            abs_lens=batch_rank['user_papers'].seq_lens)
        max_cand_sent = max(batch_rank['cand_paper'].seq_lens)
        cand_sents = torch.index_select(
            flat_abs_sent_reps, 0, cand_si).view(efbatch_size, max_cand_sent, self.bert_encoding_dim)
        cand_sents = pair_dist.rep_len_tup(embed=cand_sents.permute(0, 2, 1),
                                           abs_lens=batch_rank['cand_paper'].seq_lens)
        max_negcand_sent = max(batch_rank['neg_paper'].seq_lens)
        # print(flat_abs_sent_reps.size())
        # print(efbatch_size, max_negcand_sent, self.bert_encoding_dim)
        negcand_sents = torch.index_select(
            flat_abs_sent_reps, 0, neg_si).view(efbatch_size, max_negcand_sent, self.bert_encoding_dim)
        negcand_sents = pair_dist.rep_len_tup(embed=negcand_sents.permute(0, 2, 1),
                                              abs_lens=batch_rank['neg_paper'].seq_lens)
        return userp_sents, cand_sents, negcand_sents