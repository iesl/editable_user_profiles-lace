"""
Editable expertise and interest models.
"""
import copy
import random
import numpy as np
import torch
from torch import nn as nn
from torch.autograd import Variable
import collections
from transformers import AutoModel

from . import pair_distances as pair_dist


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
        self.bert_encoding_dim = 768
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        self.sent_encoder = AutoModel.from_pretrained(model_hparams['consent-base-pt-layer'])
        self.sent_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['sc_fine_tune']:
            for param in self.sent_encoder.base_model.parameters():
                param.requires_grad = False
        self.kp_encoder = AutoModel.from_pretrained(model_hparams['kp-base-pt-layer'])
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
        loss_margin = model_hparams.get('loss_margin', 1.0)
        self.cl_mag_prune_cand = model_hparams.get('mag_prune_cand', False)
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function,
                                                          margin=loss_margin, reduction='sum')
        self.ict_train_frac = model_hparams.get('ict_train_frac', 1.0)
        self.ict_lossprop = model_hparams.get('ict_lossprop', 0)
        if self.ict_lossprop > 0:
            self.criterion_ict = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
    
    def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
        if self.cl_mag_prune_cand:
            return self.caching_score_pruning(query_encode_ret_dict, cand_encode_ret_dicts)
        else:
            return self.caching_score_unpruned(query_encode_ret_dict, cand_encode_ret_dicts)
        
    def caching_score_unpruned(self, query_encode_ret_dict, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Treat a flattened set of user query docs sentence reps as
            a single doc with many query reps.
        - Pad candidate reps to max length.
        - Compute scores and return.
        This is used by all the models which dont prune candidates in the bottleneck:
        the sentence only model, kp only model, the clustering in bottleneck models.
        query_encode_ret_dict: list({'sent_reps': numpy.array})
        cand_encode_ret_dict: list({'sent_reps': numpy.array})
        """
        # Flatten the query abstracts sentences into a single doc with many sentences
        # In the case of user profile models this will be a single element list.
        uq_sent_reps = [d['sent_reps'][0] for d in query_encode_ret_dict]
        uq_abs_lens = [r.shape[0] for r in uq_sent_reps]
        encoding_dim = uq_sent_reps[0].shape[1]
        numq_sents = sum(uq_abs_lens)
        flat_query_sent_reps = np.zeros((numq_sents, encoding_dim))
        start_idx = 0
        for ex_num_sents, ex_reps in zip(uq_abs_lens, uq_sent_reps):
            flat_query_sent_reps[start_idx:start_idx+ex_num_sents, :] = ex_reps
            start_idx += ex_num_sents
        # Pack candidate representations as padded tensors.
        cand_sent_reps = [d['sent_reps'][0] for d in cand_encode_ret_dicts]
        batch_size = len(cand_sent_reps)
        cand_lens = [r.shape[0] for r in cand_sent_reps]
        flat_query_lens = [numq_sents]*batch_size
        cmax_sents = max(cand_lens)
        padded_cand_sent_reps = np.zeros((batch_size, cmax_sents, encoding_dim))
        repeated_query_sent_reps = np.zeros((batch_size, numq_sents, encoding_dim))
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

    def caching_score_pruning(self, query_encode_ret_dict, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Treat a flattened set of user query docs sentence reps as
            a single doc with many query reps.
        - Pad candidate reps to max length.
        - Compute scores and return.
        This is used by the sentence only models, kp only models, and the kp clustered sents models.
        query_encode_ret_dict: list({'sent_reps': numpy.array})
        cand_encode_ret_dict: list({'sent_reps': numpy.array})
        """
        assert(len(query_encode_ret_dict) == 1)
        # Flatten the query abstracts sentences into a single doc with many sentences
        uq_sent_reps = [d['sent_reps'][0] for d in query_encode_ret_dict]
        q_zero_mask = query_encode_ret_dict[0]['sent_reps'][1]
        uq_abs_lens = [r.shape[0] for r in uq_sent_reps]
        encoding_dim = uq_sent_reps[0].shape[1]
        numq_sents = sum(uq_abs_lens)
        flat_query_sent_reps = np.zeros((numq_sents, encoding_dim))
        start_idx = 0
        for ex_num_sents, ex_reps in zip(uq_abs_lens, uq_sent_reps):
            flat_query_sent_reps[start_idx:start_idx+ex_num_sents, :] = ex_reps
            start_idx += ex_num_sents
        # Pack candidate representations as padded tensors.
        cand_sent_reps = [d['sent_reps'][0] for d in cand_encode_ret_dicts]
        c_zero_masks = [d['sent_reps'][1] for d in cand_encode_ret_dicts]
        batch_size = len(cand_sent_reps)
        cand_lens = [r.shape[0] for r in cand_sent_reps]
        flat_query_lens = [numq_sents]*batch_size
        cmax_sents = max(cand_lens)
        padded_cand_sent_reps = np.zeros((batch_size, cmax_sents, encoding_dim))
        # The pad values are true meaning that they should be left out.
        padded_cand_zm = np.ones((batch_size, cmax_sents), dtype=bool)
        repeated_query_sent_reps = np.zeros((batch_size, numq_sents, encoding_dim))
        q_repeated_zm = np.zeros((batch_size, numq_sents))
        for bi, ex_reps in enumerate(cand_sent_reps):
            padded_cand_sent_reps[bi, :cand_lens[bi], :] = ex_reps
            padded_cand_zm[bi, :cand_lens[bi]] = c_zero_masks[bi]
            # Repeat the query sents.
            repeated_query_sent_reps[bi, :numq_sents, :] = flat_query_sent_reps
            q_repeated_zm[bi, :numq_sents] = q_zero_mask
        repeated_query_sent_reps = Variable(torch.FloatTensor(repeated_query_sent_reps))
        q_repeated_zm = Variable(torch.BoolTensor(q_repeated_zm))
        padded_cand_sent_reps = Variable(torch.FloatTensor(padded_cand_sent_reps))
        padded_cand_zm = Variable(torch.BoolTensor(padded_cand_zm))
        if torch.cuda.is_available():
            repeated_query_sent_reps = repeated_query_sent_reps.cuda()
            padded_cand_sent_reps = padded_cand_sent_reps.cuda()
            q_repeated_zm = q_repeated_zm.cuda()
            padded_cand_zm = padded_cand_zm.cuda()
        # Compute scores as at train time.
        qt = pair_dist.rep_len_tup(embed=repeated_query_sent_reps.permute(0, 2, 1), abs_lens=flat_query_lens,
                                   zero_mask=q_repeated_zm)
        ct = pair_dist.rep_len_tup(embed=padded_cand_sent_reps.permute(0, 2, 1), abs_lens=cand_lens,
                                   zero_mask=padded_cand_zm)
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
        zero_mask = weighted_sents.zero_mask
        weighted_sents = weighted_sents.embed.permute(0, 2, 1)
        qd = ret_items[0]
        cd = ret_items[1]
        dists = ret_items[2]
        tplan = ret_items[3]
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            zero_mask = zero_mask.cpu().data.numpy() if zero_mask!=None else None
            weighted_sents = weighted_sents.cpu().data.numpy()
            qd = qd.cpu().data.numpy() if qd else None  # these are none when using softmax for assignment.
            cd = cd.cpu().data.numpy() if cd else None
            dists = dists.cpu().data.numpy()
            tplan = tplan.cpu().data.numpy()
        else:
            zero_mask = zero_mask.data.numpy() if zero_mask!=None else None
            weighted_sents = weighted_sents.data.numpy()
            qd = qd.data.numpy() if qd else None
            cd = cd.data.numpy() if cd else None
            dists = dists.data.numpy()
            tplan = tplan.data.numpy()
        upsr = weighted_sents[0, :, :]
        tplan = tplan[0, :, :]
        dists = dists[0, :, :]
        batch_reps = [{'sent_reps': (upsr, zero_mask), 's2k_tplan': (qd, cd, dists, tplan), 'uniq_kps': unique_kp_li}]
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
        weighted_sents_tt, retlist = self.get_kpreps_scluster(sent_reps=sent_reps, kp_reps=kp_reps)
        tplan = retlist[3]
        zero_mask = weighted_sents_tt.zero_mask
        weighted_sents = weighted_sents_tt.embed.permute(0, 2, 1)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            zero_mask = zero_mask.cpu().data.numpy() if zero_mask!=None else None
            weighted_sents = weighted_sents.cpu().data.numpy()
            tplan = tplan.cpu().data.numpy()
        else:
            zero_mask = zero_mask.data.numpy() if zero_mask!=None else None
            weighted_sents = weighted_sents.data.numpy()
            tplan = tplan.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i, (num_sents, num_kps) in enumerate(zip(cand_sent_lens, cand_kp_lens)):
            # num_sents x encoding_dim
            zm = zero_mask[i, :num_kps] if weighted_sents_tt.zero_mask!=None else None
            upsr = weighted_sents[i, :num_kps, :]
            ctplan = tplan[i, :num_sents, :num_kps]
            ckps = cand_kps[i]
            # return: # num_sents x encoding_dim
            batch_reps.append({'sent_reps': (upsr, zm), 's2k_tplan': ctplan, 'ckps': ckps})
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

    def encode(self, batch_dict):
        """
        Function used at test time when encoding only sentences.
        This mimics the encode function of disent_models.
        batch_dict: dict of the form accepted by forward_rank but without any of the
            negative examples.
        :return: ret_dict
        """
        abs_bert_batch, abs_lens = batch_dict['bert_batch'], batch_dict['abs_lens']
        abs_senttoki = batch_dict['senttok_idxs']
        # Get the representations from the model; batch_size x encoding_dim x max_sents
        _, sent_reps = self.partial_forward(bert_batch=abs_bert_batch, abs_lens=abs_lens,
                                            sent_tok_idxs=abs_senttoki)
        # Make numpy arrays and return.
        if torch.cuda.is_available():
            sent_reps = sent_reps.cpu().data.numpy()
        else:
            sent_reps = sent_reps.data.numpy()
        # Return a list of reps instead of reps collated as one np array.
        unpadded_sent_reps = []
        for i, num_sents in enumerate(abs_lens):
            # num_sents x encoding_dim
            upsr = sent_reps[i, :num_sents, :]
            # return: # num_sents x encoding_dim
            unpadded_sent_reps.append(upsr)
        ret_dict = {
            'sent_reps': unpadded_sent_reps,
        }
        return ret_dict
    
    def forward(self, batch_dict):
        batch_losses = self.forward_rank(batch_dict['batch_rank'])
        if isinstance(batch_losses, tuple):
            loss_dict = {  # There will be two losses.
                'rankl': batch_losses[0],
                'aux_rankl': batch_losses[1]
            }
        else:
            loss_dict = {
                'rankl': batch_losses
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
        
        recom_loss_val = self.criterion(userp_kpsents, cand_kpsents, negcand_kpsents)
        
        # Call this in case any of the older experiments need to be re-run.
        # self._augment_loss(recom_loss_val, userp_kps, cand_kps, negcand_kps,
        #                    cand_sents, negcand_sents, user_ret_list, cand_ret_list)
        if self.ict_lossprop > 0 and self.training:
            ict_loss_val = self.ict_lossprop * self._compute_ict_loss(cand_kps, cand_sents)
            return recom_loss_val, ict_loss_val
        else:
            return recom_loss_val
    
    def _compute_ict_loss(self, cand_kps, cand_sents):
        """
        Given the candidate docs and kps computes a term match loss.
        """
        assert(cand_kps.abs_lens == cand_sents.abs_lens)
        # bs x num_sents/num_kps x dim
        cand_kp_embeds = cand_kps.embed.permute(0, 2, 1)
        cand_sent_embeds = cand_sents.embed.permute(0, 2, 1)
        bs, max_num_kps, encoding_dim = cand_kp_embeds.shape
        # Compute indices for excluding the pad vectors.
        subsampled_flat_kp_and_si = []
        for absi, num_sents in enumerate(cand_sents.abs_lens):
            cur_cand_sis = []
            for si in range(num_sents):
                cur_cand_sis.append(absi*max_num_kps+si)
            to_sample = min(int(self.ict_train_frac*num_sents), num_sents)
            sampled_train_sentidxs = random.sample(cur_cand_sis, k=to_sample)
            subsampled_flat_kp_and_si.extend(sampled_train_sentidxs)
        # Compute indexes for inbatch negs; do this by masking out
        # all the current abstracts sentences and sampling from the others.
        in_batch_neg_idxs = []
        target_start_i = 0
        all_cand_idxs = list(range(sum(cand_sents.abs_lens)))
        for num_sents in cand_sents.abs_lens:
            to_sample = min(int(self.ict_train_frac*num_sents), num_sents)
            # Its okay if some negatives repeat (choices vs sample)
            cand_neg_idxs = random.choices(all_cand_idxs[:target_start_i] + all_cand_idxs[target_start_i+num_sents:],
                                           k=to_sample)
            in_batch_neg_idxs.extend(copy.copy(cand_neg_idxs))
            target_start_i += num_sents
        subsampled_flat_kp_and_si = Variable(torch.LongTensor(subsampled_flat_kp_and_si))
        in_batch_neg_idxs = Variable(torch.LongTensor(in_batch_neg_idxs))
        if torch.cuda.is_available():
            subsampled_flat_kp_and_si = subsampled_flat_kp_and_si.cuda()
            in_batch_neg_idxs = in_batch_neg_idxs.cuda()
        # Get the sentences and kp embeddings.
        flat_cand_kp_embeds = torch.index_select(cand_kp_embeds.view(bs*max_num_kps, encoding_dim), dim=0,
                                                 index=subsampled_flat_kp_and_si)
        flat_cand_sent_embeds = torch.index_select(cand_sent_embeds.view(bs*max_num_kps, encoding_dim), dim=0,
                                                   index=subsampled_flat_kp_and_si)
        flat_negcand_sent_embeds = torch.index_select(cand_sent_embeds.view(bs*max_num_kps, encoding_dim), dim=0,
                                                      index=in_batch_neg_idxs)
        ict_loss_val = self.criterion_ict(flat_cand_kp_embeds, flat_cand_sent_embeds, flat_negcand_sent_embeds)
        return ict_loss_val
    
    def _augment_loss(self, loss_val, userp_kps, cand_kps, negcand_kps,
                      cand_sents, negcand_sents, user_ret_list, cand_ret_list):
        """
        Various experiments trying to augment the main recomendation loss.
        """
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
    
    def get_kpreps_scluster(self, sent_reps, kp_reps):
        """
        Cluster the keyphrase reps in terms of clustered sentences.
        param sent_reps: namedtuple(
            embed: torch.tensor(ef_batch_size x dim x num_sents)
            abs_lens: list(int); number of sentences in every batch element.)
        param kp_reps: namedtuple(
            embed: torch.tensor(ef_batch_size x dim x num_kps)
            abs_lens: list(int); number of kps in every batch element.)
        """
        # Get the transport plan from sentence reps to keyphrase reps.
        _, ret_list = self.ot_clustering.compute_distance(query=sent_reps,
                                                          cand=kp_reps, return_pair_sims=True)
        # ef_batch_size x num_sents x num_kps
        tplan = ret_list[3]
        # This bool tensor is only present when mag_prune_cand is true else it is a None.
        # It says which kps were pruned.
        zero_mask = ret_list[7]
        # Get the keyphrases in terms of the sentences: ef_batch_size x dim x num_kps
        weighted_sents = torch.bmm(sent_reps.embed, tplan)
        weighted_sents = pair_dist.rep_len_tup(embed=weighted_sents,
                                               abs_lens=kp_reps.abs_lens, zero_mask=zero_mask)
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


class UPNamedFBaryCProj(UPNamedFAspireKP):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Cluster user sentences into keyphrases.
    - Recompute keyphrases in terms of sentences.
    - Compute relevance in terms of keyphrase clustered sentences.
    - Importantly - barycenter projected concepts are normalized with the transport
        plan weight. And the candidate documents are represented as sentences
        -- not with concepts.
    """
    
    def __init__(self, model_hparams):
        """
        :param model_hparams: dict(string:int); model hyperparams.
        """
        torch.nn.Module.__init__(self)
        self.bert_encoding_dim = 768  # bert_config.hidden_size or DistilBertConfig.dim
        self.bert_layer_count = 12 + 1  # plus 1 for the bottom most layer.
        # todo: fine a more general way to determine if its pre-trained.
        self.sent_encoder = AutoModel.from_pretrained(model_hparams['consent-base-pt-layer'])
        self.sent_encoder.config.output_hidden_states = True
        # If fine tune is False then freeze the bert params.
        if not model_hparams['sc_fine_tune']:
            for param in self.sent_encoder.base_model.parameters():
                param.requires_grad = False
        self.kp_encoder = AutoModel.from_pretrained(model_hparams['kp-base-pt-layer'])
        if not model_hparams['kp_fine_tune']:
            for param in self.kp_encoder.base_model.parameters():
                param.requires_grad = False
        self.cluster_function = model_hparams.get('cluster_function', 'l2wasserstein')
        if self.cluster_function == 'l2wasserstein':
            self.ot_clustering = pair_dist.AllPairMaskedWassersteinCl(model_hparams)
        elif self.cluster_function == 'l2attention':
            self.ot_clustering = pair_dist.AllPairMaskedAttentionCl(model_hparams)
        self.score_agg_type = model_hparams.get('score_aggregation', 'l2wasserstein')
        if self.score_agg_type == 'l2wasserstein':
            ot_distance = pair_dist.AllPairMaskedWasserstein(model_hparams)
            self.dist_function = ot_distance.compute_distance
        elif self.score_agg_type == 'l2attention':
            att_distance = pair_dist.AllPairMaskedAttention(model_hparams)
            self.dist_function = att_distance.compute_distance
        else:
            raise ValueError(f'Unknown aggregation: {self.score_agg_type}')
        loss_margin = model_hparams.get('loss_margin', 1.0)
        self.cl_mag_prune_cand = model_hparams.get('mag_prune_cand', False)
        self.criterion = nn.TripletMarginWithDistanceLoss(distance_function=self.dist_function,
                                                          margin=loss_margin, reduction='sum')
        self.ict_train_frac = model_hparams.get('ict_train_frac', 1.0)
        self.ict_lossprop = model_hparams.get('ict_lossprop', 0)
        if self.ict_lossprop > 0:
            self.criterion_ict = nn.TripletMarginLoss(margin=1, p=2, reduction='sum')
    
    def cand_caching_cluster(self, sent_kp_reps):
        """
        In the parent class this is supposed to cluster the sentences into
        the keyphrases. But here it does not do that - it just packages
        the sentence embeddings into a form used by caching_score. This
        also allows pp_gen_nearest to work without any changes for the
        proper barycenter projection variant of the model.
        param sent_kp_reps: list(dict); of the kind returned by caching_encode.
        """
        # Get the cand abstract sents
        cand_sent_reps = [d['sent_reps'] for d in sent_kp_reps]
        cand_sent_lens = [r.shape[0] for r in cand_sent_reps]
        # Return a list of reps instead of reps collated as one np array.
        batch_reps = []
        for i, num_sents in enumerate(cand_sent_lens):
            upsr = cand_sent_reps[i]
            # True indicates which values were pruned, in this case no values are ever pruned.
            # So set all of them to zero. If no pruning is used then set the mask to zero.
            zm = np.zeros(num_sents) if self.ot_clustering.mag_prune_cand == True else None
            batch_reps.append({'sent_reps': (upsr, zm)})
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
        # The concepts are projected into the space of sentences; candidates are already in sentences.
        # cand_kpsents, cand_ret_list = self.get_kpreps_scluster(cand_sents, cand_kps)
        # negcand_kpsents, _ = self.get_kpreps_scluster(negcand_sents, negcand_kps)
        
        recom_loss_val = self.criterion(userp_kpsents, cand_sents, negcand_sents)
        
        # Call this in case any of the older experiments need to be re-run.
        # self._augment_loss(recom_loss_val, userp_kps, cand_kps, negcand_kps,
        #                    cand_sents, negcand_sents, user_ret_list, cand_ret_list)
        if self.ict_lossprop > 0 and self.training:
            ict_loss_val = self.ict_lossprop * self._compute_ict_loss(cand_kps, cand_sents)
            return recom_loss_val, ict_loss_val
        else:
            return recom_loss_val
        
    def get_kpreps_scluster(self, sent_reps, kp_reps):
        """
        Cluster the keyphrase reps in terms of clustered sentences.
        param sent_reps: namedtuple(
            embed: torch.tensor(ef_batch_size x dim x num_sents)
            abs_lens: list(int); number of sentences in every batch element.)
        param kp_reps: namedtuple(
            embed: torch.tensor(ef_batch_size x dim x num_kps)
            abs_lens: list(int); number of kps in every batch element.)
        """
        # Get the transport plan from sentence reps to keyphrase reps.
        _, ret_list = self.ot_clustering.compute_distance(query=sent_reps,
                                                          cand=kp_reps, return_pair_sims=True)
        # ef_batch_size x num_sents x num_kps
        tplan = ret_list[3]
        # Clamp the weights to prevent zero division.
        norm_weights = torch.clamp(torch.sum(tplan, dim=1), min=10e-8)
        # This bool tensor is only present when mag_prune_cand is true else it is a None.
        # It says which kps were pruned.
        zero_mask = ret_list[-1]
        # Get the keyphrases in terms of the sentences: ef_batch_size x dim x num_kps
        # Normalize each; but below line does not mask for the length of the kps
        # but the scoring function will aptly mask based on kp_reps.abs_lens.
        weighted_sents = torch.bmm(sent_reps.embed, tplan)/norm_weights.unsqueeze(dim=1)
        # The barycenter projection;
        weighted_sents = pair_dist.rep_len_tup(embed=weighted_sents,
                                               abs_lens=kp_reps.abs_lens, zero_mask=zero_mask)
        return weighted_sents, ret_list


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
        self.sent_encoder = AutoModel.from_pretrained(model_hparams['consent-base-pt-layer'])
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
        self.cl_mag_prune_cand = model_hparams.get('mag_prune_cand', False)
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
            batch_reps.append({'sent_reps': (upsr, None)})
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


class UPNamedFKPCandSent(UPNamedFBaryCProj):
    """
    - Pass abstracts through Transformer LM, get contextualized sentence reps.
        (sentence reps are obtained by averaging contextual word embeddings)
    - Compute keyphrase embeddings.
    - Compute relevance in terms of profile keyphrases and candidate sentences.
    - Importantly - no clustering of user content to keyphrases is done.
    """
    def caching_score(self, query_encode_ret_dict, cand_encode_ret_dicts):
        return self.caching_score_unpruned(query_encode_ret_dict, cand_encode_ret_dicts)
        
    def caching_score_unpruned(self, query_encode_ret_dict, cand_encode_ret_dicts):
        """
        Called externally from a class using the trained model.
        - Create as many repetitions of query_reps as cand_reps.
        - Treat a flattened set of user query docs sentence reps as
            a single doc with many query reps.
        - Pad candidate reps to max length.
        - Compute scores and return.
        This is used by all the models which dont prune candidates in the bottleneck:
        the sentence only model, kp only model, the clustering in bottleneck models.
        query_encode_ret_dict: list({'sent_reps': numpy.array})
        cand_encode_ret_dict: list({'sent_reps': numpy.array})
        """
        # Flatten the query abstracts sentences into a single doc with many sentences
        # In the case of user profile models this will be a single element list.
        uq_reps = [d['kp_reps'] for d in query_encode_ret_dict]
        uq_abs_lens = [r.shape[0] for r in uq_reps]
        encoding_dim = uq_reps[0].shape[1]
        numq_kps = sum(uq_abs_lens)
        flat_query_kp_reps = np.zeros((numq_kps, encoding_dim))
        start_idx = 0
        for ex_num_sents, ex_reps in zip(uq_abs_lens, uq_reps):
            flat_query_kp_reps[start_idx:start_idx + ex_num_sents, :] = ex_reps
            start_idx += ex_num_sents
        # Pack candidate representations as padded tensors.
        cand_sent_reps = [d['sent_reps'] for d in cand_encode_ret_dicts]
        batch_size = len(cand_sent_reps)
        cand_lens = [r.shape[0] for r in cand_sent_reps]
        flat_query_lens = [numq_kps] * batch_size
        cmax_sents = max(cand_lens)
        padded_cand_sent_reps = np.zeros((batch_size, cmax_sents, encoding_dim))
        repeated_query_reps = np.zeros((batch_size, numq_kps, encoding_dim))
        for bi, ex_reps in enumerate(cand_sent_reps):
            padded_cand_sent_reps[bi, :cand_lens[bi], :] = ex_reps
            # Repeat the query sents.
            repeated_query_reps[bi, :numq_kps, :] = flat_query_kp_reps
        repeated_query_reps = Variable(torch.FloatTensor(repeated_query_reps))
        padded_cand_sent_reps = Variable(torch.FloatTensor(padded_cand_sent_reps))
        if torch.cuda.is_available():
            repeated_query_reps = repeated_query_reps.cuda()
            padded_cand_sent_reps = padded_cand_sent_reps.cuda()
        # Compute scores as at train time.
        qt = pair_dist.rep_len_tup(embed=repeated_query_reps.permute(0, 2, 1), abs_lens=flat_query_lens)
        ct = pair_dist.rep_len_tup(embed=padded_cand_sent_reps.permute(0, 2, 1), abs_lens=cand_lens)
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
            start_kpi += len(abs_kps)
            # return: # num_sents x encoding_dim; # num_kps x encoding_dim.
            batch_reps.append({'sent_reps': upsr,
                               'kp_reps': kp_reps})
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
        
        recom_loss_val = self.criterion(userp_kps, cand_sents, negcand_sents)
        
        return recom_loss_val
