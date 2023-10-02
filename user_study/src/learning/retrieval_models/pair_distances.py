from collections import namedtuple
import numpy as np
import torch
from torch.autograd import Variable
from torch.nn import functional
import geomloss

from ..models_common import activations

rep_len_tup = namedtuple('RepLen', ['embed', 'abs_lens', 'wts', 'zero_mask'], defaults=[None, None, None, None])


class AllPairMaskedWassersteinCl:
    def __init__(self, model_hparams):
        self.geoml_blur = model_hparams.get('cl_geoml_blur', 0.05)
        self.geoml_scaling = model_hparams.get('cl_geoml_scaling', 0.9)
        self.geoml_reach = model_hparams.get('cl_geoml_reach', None)
        # Normalize the query and cand embeddings (sentences and concepts) to be bw 0 and 1.
        self.normalize_cq_points = model_hparams.get('normalize_cq_points', False)
        if 'cl_qsent_sm_temp' in model_hparams and 'cl_csent_sm_temp' in model_hparams:
            self.qsent_sm_temp = model_hparams['cl_qsent_sm_temp']
            self.csent_sm_temp = model_hparams['cl_csent_sm_temp']
        else:  # Backward compatibility.
            self.qsent_sm_temp = model_hparams['cl_sent_sm_temp']
            self.csent_sm_temp = model_hparams['cl_sent_sm_temp']
        self.mag_prune_cand = model_hparams.get('mag_prune_cand', False)
        self.mag_prune_fraction_cand = model_hparams.get('mag_prune_fraction_cand', 0.8)
    
    def compute_distance(self, query, cand, return_pair_sims=False):
        """
        Given a set of query and candidate reps compute the wasserstein distance between
        the query and candidates.
        :param query: namedtuple(
            embed: batch_size x encoding_dim x q_max_sents;
            abs_lens: list(int); number of sentences in every batch element.)
        :param cand: namedtuple(
            embed: batch_size x encoding_dim x q_max_sents;
            abs_lens: list(int); number of sentences in every batch element.)
        :return:
            batch_sims: ef_batch_size; pooled pairwise _distances_ between
                input reps. (distances are just negated similarities here)
        """
        query_reps, query_abs_lens = query.embed, query.abs_lens
        cand_reps, cand_abs_lens, cand_wts = cand.embed, cand.abs_lens, cand.wts
        qef_batch_size, _, qmax_sents = query_reps.size()
        cef_batch_size, encoding_dim, cmax_sents = cand_reps.size()
        pad_mask = np.ones((qef_batch_size, qmax_sents, cmax_sents)) * -10e8
        for i in range(qef_batch_size):
            ql, cl = query_abs_lens[i], cand_abs_lens[i]
            pad_mask[i, :ql, :cl] = 0.0
        pad_mask = Variable(torch.FloatTensor(pad_mask))
        if torch.cuda.is_available():
            pad_mask = pad_mask.cuda()
        assert (qef_batch_size == cef_batch_size)
        if self.normalize_cq_points:
            query_reps = functional.normalize(query_reps, dim=1)
            cand_reps = functional.normalize(cand_reps, dim=1)
        # (effective) batch_size x qmax_sents x cmax_sents
        # inputs are: batch_size x encoding_dim x c/qmax_sents so permute them.
        neg_pair_dists = -1 * torch.cdist(query_reps.permute(0, 2, 1).contiguous(),
                                          cand_reps.permute(0, 2, 1).contiguous())
        if len(neg_pair_dists.size()) == 2:
            neg_pair_dists = neg_pair_dists.unsqueeze(0)
        assert (neg_pair_dists.size(1) == qmax_sents)
        assert (neg_pair_dists.size(2) == cmax_sents)
        # Add very large negative values in the pad positions which will be zero.
        neg_pair_dists = neg_pair_dists + pad_mask
        q_max_sent_sims, _ = torch.max(neg_pair_dists, dim=2)
        c_max_sent_sims, _ = torch.max(neg_pair_dists, dim=1)
        query_distr_raw = functional.log_softmax(q_max_sent_sims, dim=1).exp()
        cand_distr_raw = functional.log_softmax(c_max_sent_sims, dim=1).exp()
        query_distr = functional.log_softmax(q_max_sent_sims / self.qsent_sm_temp, dim=1).exp()
        if self.mag_prune_cand:
            cand_distr, zero_mask = activations.sparse_masking_softmax(
                c_max_sent_sims, cand_abs_lens, self.mag_prune_fraction_cand)
        else:
            cand_distr = functional.log_softmax(c_max_sent_sims / self.csent_sm_temp, dim=1).exp()
            zero_mask = None
        if return_pair_sims:
            ppad_mask = np.zeros((qef_batch_size, qmax_sents, cmax_sents))
            for i in range(qef_batch_size):
                ql, cl = query_abs_lens[i], cand_abs_lens[i]
                ppad_mask[i, :ql, :cl] = 1.0
            ppad_mask = Variable(torch.FloatTensor(ppad_mask))
            if torch.cuda.is_available():
                ppad_mask = ppad_mask.cuda()
            neg_pair_dists = neg_pair_dists * ppad_mask
            # p=1 is the L2 distance oddly enough.
            ot_solver = geomloss.SamplesLoss("sinkhorn", p=1, blur=self.geoml_blur, reach=self.geoml_reach,
                                             scaling=self.geoml_scaling, debias=False, potentials=True)
            # Input reps to solver need to be: batch_size x c/qmax_sents x encoding_dim
            q_pot, c_pot = ot_solver(query_distr, query_reps.permute(0, 2, 1).contiguous(),
                                     cand_distr, cand_reps.permute(0, 2, 1).contiguous())
            # Implement the expression to compute the plan from the potentials:
            # https://www.kernel-operations.io/geomloss/_auto_examples/optimal_transport/
            # plot_optimal_transport_labels.html?highlight=plan#regularized-optimal-transport
            outersum = q_pot.unsqueeze(dim=2).expand(-1, -1, cmax_sents) + \
                       c_pot.unsqueeze(dim=2).expand(-1, -1, qmax_sents).permute(0, 2, 1)
            # Zero out the pad values because they seem to cause nans to occur.
            outersum = outersum * ppad_mask
            exps = torch.exp(torch.div(outersum + neg_pair_dists, self.geoml_blur))
            outerprod = torch.einsum('bi,bj->bij', query_distr, cand_distr)
            transport_plan = exps * outerprod
            pair_sims = neg_pair_dists
            masked_sims = transport_plan * pair_sims
            wasserstein_dists = torch.sum(torch.sum(masked_sims, dim=1), dim=1)
            return wasserstein_dists, [query_distr, cand_distr, pair_sims, transport_plan, masked_sims,
                                       # return these for regularization
                                       query_distr_raw, cand_distr_raw, zero_mask]
        else:
            ot_solver_distance = geomloss.SamplesLoss("sinkhorn", p=1, blur=self.geoml_blur, reach=self.geoml_reach,
                                                      scaling=self.geoml_scaling, debias=False, potentials=False)
            wasserstein_dists = ot_solver_distance(query_distr, query_reps.permute(0, 2, 1).contiguous(),
                                                   cand_distr, cand_reps.permute(0, 2, 1).contiguous())
            return wasserstein_dists


class AllPairMaskedAttentionCl:
    def __init__(self, model_hparams):
        self.cdatt_sm_temp = model_hparams.get('cl_cdatt_sm_temp', 1.0)
        self.mag_prune_cand = model_hparams.get('mag_prune_cand', False)
        self.mag_prune_fraction_cand = model_hparams.get('mag_prune_fraction_cand', 0.8)
    
    def compute_distance(self, query, cand, return_pair_sims=False):
        """
        Given a set of query and candidate reps compute the wasserstein distance between
        the query and candidates.
        :param query: namedtuple(
            embed: batch_size x encoding_dim x q_max_sents;
            abs_lens: list(int); number of sentences in every batch element.)
        :param cand: namedtuple(
            embed: batch_size x encoding_dim x q_max_sents;
            abs_lens: list(int); number of sentences in every batch element.)
        :return:
            batch_sims: ef_batch_size; pooled pairwise _distances_ between
                input reps. (distances are just negated similarities here)
        """
        query_reps, query_abs_lens = query.embed, query.abs_lens
        cand_reps, cand_abs_lens, cand_wts = cand.embed, cand.abs_lens, cand.wts
        qef_batch_size, _, qmax_sents = query_reps.size()
        cef_batch_size, encoding_dim, cmax_sents = cand_reps.size()
        pad_mask = np.ones((qef_batch_size, qmax_sents, cmax_sents)) * -10e8
        for i in range(qef_batch_size):
            ql, cl = query_abs_lens[i], cand_abs_lens[i]
            pad_mask[i, :ql, :cl] = 0.0
        pad_mask = Variable(torch.FloatTensor(pad_mask))
        if torch.cuda.is_available():
            pad_mask = pad_mask.cuda()
        assert (qef_batch_size == cef_batch_size)
        # (effective) batch_size x qmax_sents x cmax_sents
        # inputs are: batch_size x encoding_dim x c/qmax_sents so permute them.
        neg_pair_dists = -1 * torch.cdist(query_reps.permute(0, 2, 1).contiguous(),
                                          cand_reps.permute(0, 2, 1).contiguous())
        if len(neg_pair_dists.size()) == 2:
            neg_pair_dists = neg_pair_dists.unsqueeze(0)
        assert (neg_pair_dists.size(1) == qmax_sents)
        assert (neg_pair_dists.size(2) == cmax_sents)
        # Add very large negative values in the pad positions which will be zero.
        neg_pair_dists = neg_pair_dists + pad_mask
        c_max_sent_sims, _ = torch.max(neg_pair_dists, dim=1)
        if self.mag_prune_cand:
            _, zero_mask = activations.sparse_masking_softmax(
                c_max_sent_sims, cand_abs_lens, self.mag_prune_fraction_cand)
            masked_dists = Variable(torch.zeros((qef_batch_size, qmax_sents, cmax_sents)))
            if torch.cuda.is_available():
                masked_dists = masked_dists.cuda()
            # Fill all the pruned cands with large negative values.
            masked_dists = masked_dists.masked_fill(zero_mask.unsqueeze(1), -10e8)
            neg_pair_dists += masked_dists
        else:
            zero_mask = None
        pair_softmax = activations.masked_2d_softmax(neg_pair_dists / self.cdatt_sm_temp,
                                                     target_lens1=query_abs_lens,
                                                     target_lens2=cand_abs_lens)
        if return_pair_sims:
            pair_sims = neg_pair_dists
            masked_sims = pair_softmax * pair_sims
            doc_sims = torch.sum(torch.sum(masked_sims, dim=1), dim=1)
            return doc_sims, [None, None, pair_sims, pair_softmax, masked_sims, zero_mask]
        else:
            # This never really gets used since i never use the clustering distance now.
            pair_dists = -1 * neg_pair_dists
            masked_dists = pair_softmax * pair_dists
            doc_dists = torch.sum(torch.sum(masked_dists, dim=1), dim=1)
            return doc_dists


class AllPairMaskedWasserstein:
    def __init__(self, model_hparams):
        self.geoml_blur = model_hparams.get('geoml_blur', 0.05)
        self.geoml_scaling = model_hparams.get('geoml_scaling', 0.9)
        self.geoml_reach = model_hparams.get('geoml_reach', None)
        self.normalize_points = model_hparams.get('normalize_points', False)
        if 'qsent_sm_temp' in model_hparams and 'csent_sm_temp' in model_hparams:
            self.qsent_sm_temp = model_hparams['qsent_sm_temp']
            self.csent_sm_temp = model_hparams['csent_sm_temp']
        else:  # Backward compatibility.
            self.qsent_sm_temp = model_hparams['sent_sm_temp']
            self.csent_sm_temp = model_hparams['sent_sm_temp']
        self.cl_mag_prune_cand = model_hparams.get('mag_prune_cand', False)
        self.mag_prune_query = model_hparams.get('mag_prune_query', False)
        self.mag_prune_fraction = model_hparams.get('mag_prune_fraction', 0.8)
    
    def compute_distance(self, query, cand, return_pair_sims=False):
        """
        Given a set of query and candidate reps compute the wasserstein distance between
        the query and candidates.
        :param query: namedtuple(
            embed: batch_size x encoding_dim x q_max_sents;
            abs_lens: list(int); number of sentences in every batch element.)
        :param cand: namedtuple(
            embed: batch_size x encoding_dim x q_max_sents;
            abs_lens: list(int); number of sentences in every batch element.)
        :return:
            batch_sims: ef_batch_size; pooled pairwise _distances_ between
                input reps. (distances are just negated similarities here)
        """
        query_reps, query_abs_lens, query_wts = query.embed, query.abs_lens, query.wts
        cand_reps, cand_abs_lens = cand.embed, cand.abs_lens
        qef_batch_size, _, qmax_sents = query_reps.size()
        cef_batch_size, encoding_dim, cmax_sents = cand_reps.size()
        # Passed when the mag_prune_cand is set to true.
        query_zero_mask, cand_zero_mask = query.zero_mask, cand.zero_mask
        if (query_zero_mask != None) and (cand_zero_mask != None):
            assert (self.cl_mag_prune_cand == True)
            masked_dists = Variable(torch.zeros((qef_batch_size, qmax_sents, cmax_sents)))
            if torch.cuda.is_available():
                masked_dists = masked_dists.cuda()
        pad_mask = np.ones((qef_batch_size, qmax_sents, cmax_sents)) * -10e8
        for i in range(qef_batch_size):
            ql, cl = query_abs_lens[i], cand_abs_lens[i]
            pad_mask[i, :ql, :cl] = 0.0
        pad_mask = Variable(torch.FloatTensor(pad_mask))
        if torch.cuda.is_available():
            pad_mask = pad_mask.cuda()
        assert (qef_batch_size == cef_batch_size)
        # Normalize incoming vectors; [This is only done for non baryc projection models.]
        if self.normalize_points:
            # The experiments with sent cluster and kp can also be handeled by this function
            # but the normalization should have been performed in an earlier step rather than here.
            assert (encoding_dim == 768)
            query_reps = functional.normalize(query_reps, dim=1)
            cand_reps = functional.normalize(cand_reps, dim=1)
        # (effective) batch_size x qmax_sents x cmax_sents
        # inputs are: batch_size x encoding_dim x c/qmax_sents so permute them.
        neg_pair_dists = -1 * torch.cdist(query_reps.permute(0, 2, 1).contiguous(),
                                          cand_reps.permute(0, 2, 1).contiguous())
        # If the kps have been pruned at clustering then set them to zeros here.
        if (query_zero_mask != None) and (cand_zero_mask != None):
            # The zero masks have a True wherever we should have a high distance.
            # We want a logical OR outerproduct but have an AND operator via multiplication
            # so use demorgans laws and express the OR as an AND.
            dist_mask = ~torch.einsum('bi,bj->bij', ~query_zero_mask, ~cand_zero_mask)
            masked_dists = masked_dists.masked_fill(dist_mask, -10e8)
            neg_pair_dists += masked_dists
        if len(neg_pair_dists.size()) == 2:
            neg_pair_dists = neg_pair_dists.unsqueeze(0)
        assert (neg_pair_dists.size(1) == qmax_sents)
        assert (neg_pair_dists.size(2) == cmax_sents)
        # Add very large negative values in the pad positions which will be zero.
        neg_pair_dists = neg_pair_dists + pad_mask
        q_max_sent_sims, _ = torch.max(neg_pair_dists, dim=2)
        c_max_sent_sims, _ = torch.max(neg_pair_dists, dim=1)
        if self.mag_prune_query:
            # If the query elms were pruned at clustering then this stage should account for that.
            if (query_zero_mask != None):
                # The trues in the mask are what should be pruned. So negate it and count the nonzero.
                pruned_q_lens = (~query_zero_mask).count_nonzero(dim=1)
                if torch.cuda.is_available():
                    pruned_q_lens = pruned_q_lens.cpu().data.tolist()
                else:
                    pruned_q_lens = pruned_q_lens.data.tolist()
                assert (len(pruned_q_lens) == len(query_abs_lens))
                for pql, ql in zip(pruned_q_lens, query_abs_lens):
                    assert (pql <= ql)
            else:
                pruned_q_lens = query_abs_lens
            query_distr, m = activations.sparse_masking_softmax(q_max_sent_sims, pruned_q_lens,
                                                                self.mag_prune_fraction)
        else:
            query_distr = functional.log_softmax(q_max_sent_sims / self.qsent_sm_temp, dim=1).exp()
        cand_distr = functional.log_softmax(c_max_sent_sims / self.csent_sm_temp, dim=1).exp()
        
        if return_pair_sims:
            ppad_mask = np.zeros((qef_batch_size, qmax_sents, cmax_sents))
            for i in range(qef_batch_size):
                ql, cl = query_abs_lens[i], cand_abs_lens[i]
                ppad_mask[i, :ql, :cl] = 1.0
            ppad_mask = Variable(torch.FloatTensor(ppad_mask))
            if torch.cuda.is_available():
                ppad_mask = ppad_mask.cuda()
            neg_pair_dists = neg_pair_dists * ppad_mask
            # p=1 is the L2 distance oddly enough.
            ot_solver = geomloss.SamplesLoss("sinkhorn", p=1, blur=self.geoml_blur, reach=self.geoml_reach,
                                             scaling=self.geoml_scaling, debias=False, potentials=True)
            # Input reps to solver need to be: batch_size x c/qmax_sents x encoding_dim
            q_pot, c_pot = ot_solver(query_distr, query_reps.permute(0, 2, 1).contiguous(),
                                     cand_distr, cand_reps.permute(0, 2, 1).contiguous())
            # Implement the expression to compute the plan from the potentials:
            # https://www.kernel-operations.io/geomloss/_auto_examples/optimal_transport/
            # plot_optimal_transport_labels.html?highlight=plan#regularized-optimal-transport
            outersum = q_pot.unsqueeze(dim=2).expand(-1, -1, cmax_sents) + \
                       c_pot.unsqueeze(dim=2).expand(-1, -1, qmax_sents).permute(0, 2, 1)
            # Zero out the pad values because they seem to cause nans to occur.
            outersum = outersum * ppad_mask
            exps = torch.exp(torch.div(outersum + neg_pair_dists, self.geoml_blur))
            outerprod = torch.einsum('bi,bj->bij', query_distr, cand_distr)
            transport_plan = exps * outerprod
            pair_sims = neg_pair_dists
            masked_sims = transport_plan * pair_sims
            wasserstein_dists = torch.sum(torch.sum(masked_sims, dim=1), dim=1)
            return wasserstein_dists, [query_distr, cand_distr, pair_sims, transport_plan, masked_sims]
        else:
            ot_solver_distance = geomloss.SamplesLoss("sinkhorn", p=1, blur=self.geoml_blur, reach=self.geoml_reach,
                                                      scaling=self.geoml_scaling, debias=False, potentials=False)
            wasserstein_dists = ot_solver_distance(query_distr, query_reps.permute(0, 2, 1).contiguous(),
                                                   cand_distr, cand_reps.permute(0, 2, 1).contiguous())
            return wasserstein_dists


class AllPairMaskedAttention:
    def __init__(self, model_hparams):
        self.cdatt_sm_temp = model_hparams.get('cdatt_sm_temp', 1.0)
        self.cl_mag_prune_cand = model_hparams.get('mag_prune_cand', False)
        self.mag_prune_query = model_hparams.get('mag_prune_query', False)
        self.mag_prune_fraction = model_hparams.get('mag_prune_fraction', 0.8)
    
    def compute_distance(self, query, cand, return_pair_sims=False):
        """
        Given a set of query and candidate reps compute the wasserstein distance between
        the query and candidates.
        :param query: namedtuple(
            embed: batch_size x encoding_dim x q_max_sents;
            abs_lens: list(int); number of sentences in every batch element.)
        :param cand: namedtuple(
            embed: batch_size x encoding_dim x q_max_sents;
            abs_lens: list(int); number of sentences in every batch element.)
        :return:
            batch_sims: ef_batch_size; pooled pairwise _distances_ between
                input reps. (distances are just negated similarities here)
        """
        query_reps, query_abs_lens = query.embed, query.abs_lens
        cand_reps, cand_abs_lens = cand.embed, cand.abs_lens
        qef_batch_size, _, qmax_sents = query_reps.size()
        cef_batch_size, encoding_dim, cmax_sents = cand_reps.size()
        # Passed when the mag_prune_cand is set to true.
        query_zero_mask, cand_zero_mask = query.zero_mask, cand.zero_mask
        if (query_zero_mask != None):
            assert (self.cl_mag_prune_cand == True)
            masked_dists = Variable(torch.zeros((qef_batch_size, qmax_sents, cmax_sents)))
            if torch.cuda.is_available():
                masked_dists = masked_dists.cuda()
        pad_mask = np.ones((qef_batch_size, qmax_sents, cmax_sents)) * -10e8
        for i in range(qef_batch_size):
            ql, cl = query_abs_lens[i], cand_abs_lens[i]
            pad_mask[i, :ql, :cl] = 0.0
        pad_mask = Variable(torch.FloatTensor(pad_mask))
        if torch.cuda.is_available():
            pad_mask = pad_mask.cuda()
        assert (qef_batch_size == cef_batch_size)
        # (effective) batch_size x qmax_sents x cmax_sents
        # inputs are: batch_size x encoding_dim x c/qmax_sents so permute them.
        neg_pair_dists = -1 * torch.cdist(query_reps.permute(0, 2, 1).contiguous(),
                                          cand_reps.permute(0, 2, 1).contiguous())
        # Add very large negative values in the pad positions which will be zero.
        neg_pair_dists += pad_mask
        # Select only a subset of the query.
        if self.mag_prune_query:
            # If the query elms were pruned at clustering then this stage should account for that.
            if (query_zero_mask != None):
                # The trues in the mask are what should be pruned. So negate it and count the nonzero.
                pruned_q_lens = (~query_zero_mask).count_nonzero(dim=1)
            else:
                pruned_q_lens = query_abs_lens
            q_max_sent_sims, _ = torch.max(neg_pair_dists, dim=2)
            # Get which values need to be masked from the function but discard the softmax value.
            _, m = activations.sparse_masking_softmax(q_max_sent_sims, pruned_q_lens,
                                                      self.mag_prune_fraction)
            masked_dists = masked_dists.masked_fill(m.unsqueeze(2), -10e8)
            neg_pair_dists += masked_dists
        # The below function will mask the pad values and return a softmax - pruned values will
        # get low probs anyway.
        pair_softmax = activations.masked_2d_softmax(neg_pair_dists / self.cdatt_sm_temp,
                                                     target_lens1=query_abs_lens,
                                                     target_lens2=cand_abs_lens)
        if return_pair_sims:
            pair_sims = neg_pair_dists
            masked_sims = pair_softmax * pair_sims
            doc_sims = torch.sum(torch.sum(masked_sims, dim=1), dim=1)
            return doc_sims, [pair_sims, pair_softmax, masked_sims]
        else:
            # Happens at train time.
            pair_dists = -1 * neg_pair_dists
            masked_dists = pair_softmax * pair_dists
            doc_dists = torch.sum(torch.sum(masked_dists, dim=1), dim=1)
            return doc_dists
