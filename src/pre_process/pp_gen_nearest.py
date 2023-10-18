"""
For a given corpus with facet labels on the abstract sentences
write out run bert on the sentences and write it out.
"""
import os
import sys
import logging
import time
import codecs, json
import argparse
import collections

import joblib
import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer

from . import data_utils as du
from ..learning.retrieval_models import content_profile_models, editable_profile_models, contentcf_models
from ..learning import batchers, rec_batchers

# https://stackoverflow.com/a/46635273/3262406
np.set_printoptions(suppress=True)


class CachingTrainedScoringModelConCF:
    """
    Class to initialize trained document encoding model and learned user latent
    factors cache them, and score query candidate pairs. The query here is a user id.
    """
    def __init__(self, user_ids, all_hparams, model_name, trained_model_path, model_version='cur_best'):
        # Init model:
        self.model_name = model_name
        self.model_hparams = all_hparams
        with codecs.open(os.path.join(trained_model_path, 'uid2idx.json'), 'r') as fp:
            uid2idx = json.load(fp)
            map_uids = set(uid2idx.keys())
            user_ids = set(user_ids)
            assert(user_ids.issubset(map_uids))
            logging.info(f'Read: {fp.name}')
        self.uid2int = uid2idx
        if model_name in {'contentcf'}:
            contentcf_models.ContentCF.num_users = len(uid2idx)
            model = contentcf_models.ContentCF(model_hparams=all_hparams)
            batcher = rec_batchers.UserIDCandBatcher
        else:
            raise ValueError(f'Unknown model: {model_name}')
        model_fname = os.path.join(trained_model_path, 'model_{:s}.pt'.format(model_version))
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_fname))
        else:
            model.load_state_dict(torch.load(model_fname, map_location=torch.device('cpu')))
        logging.info(f'Scoring model: {model_fname}')
        self.abs_tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
        # Move model to the GPU.
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        self.model_name = model_name
        self.model = model
        self.batcher = batcher
        self.pid2model_reps = {}
    
    def save_cache(self, out_fname):
        """
        Saves the cache to disk in case we want to use it ever.
        """
        joblib.dump(self.pid2model_reps, out_fname, compress=('gzip', 3))
    
    def predict(self, user_id, cand_pids, pid2abstract, score_batch_size=64):
        """
        Use trained model to return scores between query and candidate.
        :param user_id: string; The string id for the user - the model has a learned
            rep per user.
        :param cand_pids: list(string)
        :param pid2abstract: dict(string: dict)
        :return:
        """
        # Gets reps of uncached documents.
        self.build_basemodel_reps(cand_pids, pid2abstract)
        user_rep_idx = self.uid2int[user_id]
        ret_dict = self.score_candidates(user_rep_idx, cand_pids, score_batch_size)
        return ret_dict
    
    def score_candidates(self, user_rep_idx, cand_pids, score_batch_size):
        # Score documents based on reps.
        cand_batch = []
        cand_scores = []
        pair_sm = []
        for ci, cpid in enumerate(cand_pids):
            cand_batch.append(self.pid2model_reps[cpid])
            if ci % 1000 == 0:
                logging.info(f'Scoring: {ci}/{len(cand_pids)}')
            if len(cand_batch) == score_batch_size:
                with torch.no_grad():
                    score_dict = self.model.caching_score(query_user_idx=user_rep_idx,
                                                          cand_encode_ret_dicts=cand_batch)
                cand_scores.extend(score_dict['batch_scores'].tolist())
                pair_sm.extend(score_dict['pair_scores'])
                cand_batch = []
        if cand_batch:  # Handle final few candidates.
            with torch.no_grad():
                score_dict = self.model.caching_score(query_user_idx=user_rep_idx,
                                                      cand_encode_ret_dicts=cand_batch)
            cand_scores.extend(score_dict['batch_scores'].tolist())
            pair_sm.extend(score_dict['pair_scores'])
        ret_dict = {'cand_scores': cand_scores, 'pair_scores': pair_sm}
        return ret_dict
    
    def build_basemodel_reps(self, cand_pids, pid2abstract, encode_batch_size=32):
        """
        Given pids, cache their reps from the base models.
        """
        uncached_pids = [cpid for cpid in cand_pids if cpid not in self.pid2model_reps]
        if uncached_pids:
            batch_pids = []
            for i, pid in enumerate(uncached_pids):
                batch_pids.append(pid)
                if i % 1000 == 0:
                    logging.info(f'Encoding: {i}/{len(uncached_pids)}')
                if len(batch_pids) == encode_batch_size:
                    batch_dict = self.batcher.make_test_batch(batch_pids=batch_pids, pid2abstract=pid2abstract,
                                                              abstract_pt_lm_tokenizer=self.abs_tokenizer)
                    with torch.no_grad():
                        batch_rep_dicts = self.model.caching_encode(batch_dict)
                    assert (len(batch_pids) == len(batch_rep_dicts))
                    for upid, batch_reps in zip(batch_pids, batch_rep_dicts):
                        self.pid2model_reps[upid] = batch_reps
                    batch_pids = []
            if batch_pids:  # Last batch.
                batch_dict = self.batcher.make_test_batch(batch_pids=batch_pids, pid2abstract=pid2abstract,
                                                          abstract_pt_lm_tokenizer=self.abs_tokenizer)
                with torch.no_grad():
                    batch_rep_dicts = self.model.caching_encode(batch_dict)
                assert (len(batch_pids) == len(batch_rep_dicts))
                for upid, batch_reps in zip(batch_pids, batch_rep_dicts):
                    self.pid2model_reps[upid] = batch_reps


class CachingTrainedScoringModelUDoc2D:
    """
    Class to initialize trained document encoding model, build document reps,
    cache them, and score query candidate pairs. The "query" here is a SET of
    documents representative of a user.
    """
    def __init__(self, all_hparams, model_name, trained_model_path, model_version='cur_best'):
        # Init model:
        if model_name in {'miswordbienc'}:
            model = content_profile_models.WordSentAlignBiEnc(model_hparams=all_hparams)
            batcher = batchers.AbsSentTokBatcher
        elif model_name in {'sentsbmpnet1b', 'sentsbnlibert'}:
            model = content_profile_models.WordSentAlignBiEnc(model_hparams=all_hparams)
            batcher = batchers.AbsSentTokBatcher
        elif model_name in {'specter', 'scincl'}:
            model = content_profile_models.SPECTER(model_hparams=all_hparams)
            batcher = batchers.AbsTripleBatcher
        elif model_name in {'docsbmpnet1b', 'docsbnlibert'}:
            model = content_profile_models.SPECTER(model_hparams=all_hparams)
            batcher = batchers.AbsTripleBatcher
        else:
            raise ValueError(f'Unknown model: {model_name}')
        if trained_model_path:
            model_fname = os.path.join(trained_model_path, 'model_{:s}.pt'.format(model_version))
            model.load_state_dict(torch.load(model_fname))
            logging.info(f'Scoring model: {model_fname}')
        self.tokenizer = AutoTokenizer.from_pretrained(all_hparams['base-pt-layer'])
        # Move model to the GPU.
        if torch.cuda.is_available():
            model.cuda()
            logging.info('Running on GPU.')
        model.eval()
        self.model_name = model_name
        self.model = model
        self.batcher = batcher
        self.pid2model_reps = {}
    
    def save_cache(self, out_fname):
        """
        Saves the cache to disk in case we want to use it ever.
        """
        joblib.dump(self.pid2model_reps, out_fname, compress=('gzip', 3))
    
    def predict(self, user_query_pids, cand_pids, pid2abstract, facet='all'):
        """
        Use trained model to return scores between query and candidate.
        :param user_query_pids: list(string); Since this is a recommendation application
            the "query" is a set of papers the user has liked.
        :param cand_pids: list(string)
        :param pid2abstract: dict(string: dict)
        :param facet: string; {'all', 'background', 'method', 'result'}
        :return:
        """
        # Gets reps of uncached documents.
        encode_batch_size = 32
        uncached_pids = [cpid for cpid in cand_pids if cpid not in self.pid2model_reps]
        for uq_pid in user_query_pids:
            if uq_pid not in self.pid2model_reps:
                uncached_pids.append(uq_pid)
        if uncached_pids:
            batch_docs = []
            batch_pids = []
            for i, pid in enumerate(uncached_pids):
                batch_docs.append({'TITLE': pid2abstract[pid]['title'],
                                   'ABSTRACT': pid2abstract[pid]['abstract']})
                batch_pids.append(pid)
                if i % 1000 == 0:
                    logging.info(f'Encoding: {i}/{len(uncached_pids)}')
                if len(batch_docs) == encode_batch_size:
                    batch_dict = self.batcher.make_batch(raw_feed={'query_texts': batch_docs},
                                                         pt_lm_tokenizer=self.tokenizer)
                    with torch.no_grad():
                        batch_rep_dicts = self.model.caching_encode(batch_dict)
                    assert(len(batch_pids) == len(batch_rep_dicts))
                    for upid, batch_reps in zip(batch_pids, batch_rep_dicts):
                        self.pid2model_reps[upid] = batch_reps
                    batch_docs = []
                    batch_pids = []
            if batch_docs:  # Last batch.
                batch_dict = self.batcher.make_batch(raw_feed={'query_texts': batch_docs},
                                                     pt_lm_tokenizer=self.tokenizer)
                with torch.no_grad():
                    batch_rep_dicts = self.model.caching_encode(batch_dict)
                assert(len(batch_pids) == len(batch_rep_dicts))
                for upid, batch_reps in zip(batch_pids, batch_rep_dicts):
                    self.pid2model_reps[upid] = batch_reps
        # Score documents based on reps.
        # Get query facet sent idxs.
        user_query_reps = [self.pid2model_reps[uqpid] for uqpid in user_query_pids]
        score_batch_size = 64
        cand_batch = []
        cand_scores = []
        pair_sm = []
        for ci, cpid in enumerate(cand_pids):
            cand_batch.append(self.pid2model_reps[cpid])
            if ci % 1000 == 0:
                logging.info(f'Scoring: {ci}/{len(cand_pids)}')
            if len(cand_batch) == score_batch_size:
                with torch.no_grad():
                    score_dict = self.model.caching_score(query_encode_ret_dict=user_query_reps,
                                                          cand_encode_ret_dicts=cand_batch)
                cand_scores.extend(score_dict['batch_scores'].tolist())
                pair_sm.extend(score_dict['pair_scores'])
                cand_batch = []
        if cand_batch:  # Handle final few candidates.
            with torch.no_grad():
                score_dict = self.model.caching_score(query_encode_ret_dict=user_query_reps,
                                                      cand_encode_ret_dicts=cand_batch)
            cand_scores.extend(score_dict['batch_scores'].tolist())
            pair_sm.extend(score_dict['pair_scores'])
        ret_dict = {'cand_scores': cand_scores, 'pair_scores': pair_sm}
        return ret_dict


class CachingTrainedScoringModelUPro2D:
    """
    Class to initialize trained document encoding model, build user profile reps,
    cache them, and score query candidate pairs. The "query" here is a SET of
    documents representative of a user.
    """
    def __init__(self, all_hparams, model_name, trained_model_path, model_version='cur_best'):
        # Init model:
        self.model_name = model_name
        self.model_hparams = all_hparams
        self.kps_per_abs = all_hparams.get('kps_per_abs', None)
        self.kp2idx = None
        self.uid2kps = None
        
        if model_name in {'upsentconsent'}:
            model = editable_profile_models.UPSentAspire(model_hparams=all_hparams)
            batcher = rec_batchers.UserCandKPBatcher
        elif model_name in {'upnfconsent'} and all_hparams.get('barycenter_projection', False):
            model = editable_profile_models.UPNamedFBaryCProj(model_hparams=all_hparams)
            batcher = rec_batchers.UserCandKPBatcher
        elif model_name in {'upnfkpenc'}:
            model = editable_profile_models.UPNamedFKPCandSent(model_hparams=all_hparams)
            batcher = rec_batchers.UserCandKPBatcher
        else:
            raise ValueError(f'Unknown model: {model_name}')
        model_fname = os.path.join(trained_model_path, 'model_{:s}.pt'.format(model_version))
        if torch.cuda.is_available():
            model.load_state_dict(torch.load(model_fname))
        else:
            model.load_state_dict(torch.load(model_fname, map_location=torch.device('cpu')))
        logging.info(f'Scoring model: {model_fname}')
        if 'consent-base-pt-layer' in all_hparams and 's2orccompsci' in all_hparams['consent-base-pt-layer']:
            self.abs_tokenizer = AutoTokenizer.from_pretrained("allenai/specter")
            self.kp_tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
        else:
            self.abs_tokenizer = AutoTokenizer.from_pretrained(all_hparams['consent-base-pt-layer'])
            self.kp_tokenizer = AutoTokenizer.from_pretrained(all_hparams['kp-base-pt-layer'])
        # Move model to the GPU.
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        self.model_name = model_name
        self.model = model
        self.batcher = batcher
        self.pid2model_reps = {}
        self.pid2clustered_model_reps = {}
    
    def save_cache(self, out_fname):
        """
        Saves the cache to disk in case we want to use it ever.
        """
        joblib.dump(self.pid2model_reps, out_fname, compress=('gzip', 3))
    
    def predict(self, user_id, user_query_pids, cand_pids, pid2abstract, score_batch_size=64):
        """
        Use trained model to return scores between query and candidate.
        :param user_query_pids: list(string); Since this is a recommendation application
            the "query" is a set of papers the user has liked.
        :param cand_pids: list(string)
        :param pid2abstract: dict(string: dict)
        :return:
        """
        self.build_basemodel_reps(user_query_pids, cand_pids, pid2abstract)
        # Further build caches for clustered documents.
        if self.model_name in {'upnfconsent'}:
            user_query_reps = self.build_clustered_reps(user_query_pids, cand_pids)
        else:
            user_query_reps = [self.pid2model_reps[uqpid] for uqpid in user_query_pids]
        ret_dict = self.score_candidates(user_query_reps, cand_pids, score_batch_size)
        return ret_dict

    def editable_predict(self, user_query_reps, cand_pids, pid2abstract, score_batch_size=64):
        """
        Use trained model to return scores between query and candidate.
        The intent behind this function is that it will only be called interactively once the base model and clustered
        reps have been built.
        :param user_query_reps: Since this is a recommendation application
            the "query" is a set of papers the user has liked. These are the query reps for the
            user.
            [dict('sent_reps': np.array, num_kp_clusters x dim;
                  's2k_tplan': (qd, cd, dists, tplan),
                  'uniq_kps': unique_kp_li length num_kp_clusters; aligned to sent_reps.
            )];
        :param cand_pids: list(string)
        :param pid2abstract: dict(string: dict)
        :return:
        """
        assert(self.model_name in {'upnfconsent'})
        # Gets reps of uncached documents; commented out because its assumed that it will already be done.
        # self.build_basemodel_reps(user_query_pids, cand_pids, pid2abstract)
        # Further build caches for clustered documents; commented out because its assumed that it will already be done.
        # user_query_reps = self.build_clustered_reps(user_query_pids, cand_pids)
        ret_dict = self.score_candidates(user_query_reps, cand_pids, score_batch_size)
        return ret_dict
    
    def score_candidates(self, user_query_reps, cand_pids, score_batch_size):
        """
        Code shared between predict and editable_predict function.
        """
        # Score documents based on reps.
        cand_batch = []
        cand_scores = []
        pair_sm = []
        for ci, cpid in enumerate(cand_pids):
            if self.model_name in {'upnfconsent'}:
                cand_batch.append(self.pid2clustered_model_reps[cpid])
            else:
                cand_batch.append(self.pid2model_reps[cpid])
            if ci % 1000 == 0:
                logging.info(f'Scoring: {ci}/{len(cand_pids)}')
            if len(cand_batch) == score_batch_size:
                with torch.no_grad():
                    score_dict = self.model.caching_score(query_encode_ret_dict=user_query_reps,
                                                          cand_encode_ret_dicts=cand_batch)
                cand_scores.extend(score_dict['batch_scores'].tolist())
                pair_sm.extend(score_dict['pair_scores'])
                cand_batch = []
        if cand_batch:  # Handle final few candidates.
            with torch.no_grad():
                score_dict = self.model.caching_score(query_encode_ret_dict=user_query_reps,
                                                      cand_encode_ret_dicts=cand_batch)
            cand_scores.extend(score_dict['batch_scores'].tolist())
            pair_sm.extend(score_dict['pair_scores'])
        ret_dict = {'cand_scores': cand_scores, 'pair_scores': pair_sm}
        return ret_dict
    
    def build_basemodel_reps(self, user_query_pids, cand_pids, pid2abstract, encode_batch_size=32):
        """
        Given pids, cache their reps from the base models.
        """
        uncached_pids = [cpid for cpid in cand_pids if cpid not in self.pid2model_reps]
        for uq_pid in user_query_pids:
            if uq_pid not in self.pid2model_reps:
                uncached_pids.append(uq_pid)
        if uncached_pids:
            batch_pids = []
            for i, pid in enumerate(uncached_pids):
                batch_pids.append(pid)
                if i % 1000 == 0:
                    logging.info(f'Encoding: {i}/{len(uncached_pids)}')
                if len(batch_pids) == encode_batch_size:
                    batch_dict = self.batcher.make_test_batch(batch_pids=batch_pids, pid2abstract=pid2abstract,
                                                              abstract_pt_lm_tokenizer=self.abs_tokenizer,
                                                              kp_pt_lm_tokenizer=self.kp_tokenizer,
                                                              kps_per_abs=self.kps_per_abs,
                                                              kp2idx=self.kp2idx, uid2kps=self.uid2kps)
                    with torch.no_grad():
                        batch_rep_dicts = self.model.caching_encode(batch_dict)
                    assert(len(batch_pids) == len(batch_rep_dicts))
                    for upid, batch_reps in zip(batch_pids, batch_rep_dicts):
                        self.pid2model_reps[upid] = batch_reps
                    batch_pids = []
            if batch_pids:  # Last batch.
                batch_dict = self.batcher.make_test_batch(batch_pids=batch_pids, pid2abstract=pid2abstract,
                                                          abstract_pt_lm_tokenizer=self.abs_tokenizer,
                                                          kp_pt_lm_tokenizer=self.kp_tokenizer,
                                                          kps_per_abs=self.kps_per_abs,
                                                          kp2idx=self.kp2idx, uid2kps=self.uid2kps)
                with torch.no_grad():
                    batch_rep_dicts = self.model.caching_encode(batch_dict)
                assert(len(batch_pids) == len(batch_rep_dicts))
                for upid, batch_reps in zip(batch_pids, batch_rep_dicts):
                    self.pid2model_reps[upid] = batch_reps
    
    def build_clustered_reps(self, user_query_pids, cand_pids, cl_batch_size=64):
        """
        Once the sentence reps are built, build a set of clustered reps
        for the query cand candidates
        """
        uq_reps = [self.pid2model_reps[uqpid] for uqpid in user_query_pids]
        with torch.no_grad():
            uq_clustered_rep = self.model.user_caching_cluster(sent_kp_reps=uq_reps)
        uncached_pids = [cpid for cpid in cand_pids if cpid not in self.pid2clustered_model_reps]
        cl_batch = []
        cl_batch_pids = []
        if uncached_pids:
            for i, cpid in enumerate(uncached_pids):
                cl_batch.append(self.pid2model_reps[cpid])
                cl_batch_pids.append(cpid)
                if i % 1000 == 0:
                    logging.info(f'Clustering: {i}/{len(uncached_pids)}')
                if len(cl_batch) == cl_batch_size:
                    with torch.no_grad():
                        cand_reps = self.model.cand_caching_cluster(sent_kp_reps=cl_batch)
                    # Cache the results so other methods in the class can use them.
                    for pid, creps in zip(cl_batch_pids, cand_reps):
                        self.pid2clustered_model_reps[pid] = creps
                    cl_batch = []
                    cl_batch_pids = []
            if cl_batch_pids:
                with torch.no_grad():
                    cand_reps = self.model.cand_caching_cluster(sent_kp_reps=cl_batch)
                for pid, creps in zip(cl_batch_pids, cand_reps):
                    self.pid2clustered_model_reps[pid] = creps
        # These are not cached since they are only used the one time.
        return uq_clustered_rep


def caching_scoringmodel_rank_pool_sent(root_path, model_name, dataset, run_name, model_version,
                                        model_config_path=None, trained_model_path=None, train_suffix=None,
                                        ann_suffix=False):
    """
    Given a pool of candidates re-rank the pool based on the model scores.
    Function for use when model classes provide methods to encode data, and then score
    documents. Representations are generated at the same time as scoringg, not apriori saved on disk.
    :param root_path: string; directory with abstracts jsonl and citation network data and subdir of
        reps to use for retrieval.
    :param dataset: string; {'citeulikea', 'citeuliket', 'tedrec'}; eval dataset to use.
    :param model_name: string;
    :param ann_suffix: string;
        big: the test set of users with many interactions in citeulike.
        bids: users with bids in OR data.
        assigns: users with assigns in OR data.
    :return: write to disk.
    """
    if dataset in {'tedrec', 'oriclr2019', 'oriclr2020', 'oruai2019'}:
        kp_source = 'gold'
        to_print = 300
    else:
        kp_source = 'forecite'
        to_print = 50
    # Load model config.
    if trained_model_path:
        with codecs.open(os.path.join(trained_model_path, 'run_info.json'), 'r', 'utf-8') as fp:
            run_info = json.load(fp)
            all_hparams = run_info['all_hparams']
    elif model_config_path:
        with codecs.open(model_config_path, 'r', 'utf-8') as fp:
            all_hparams = json.load(fp)
    elif model_name in {'docsbmpnet1b', 'docsbnlibert', 'sentsbmpnet1b', 'sentsbnlibert', 'specter', 'scincl'}:
        base_pt_layer = {'docsbmpnet1b': 'sentence-transformers/all-mpnet-base-v2',
                         'docsbnlibert': 'sentence-transformers/bert-base-nli-mean-tokens',
                         'sentsbmpnet1b': 'sentence-transformers/all-mpnet-base-v2',
                         'sentsbnlibert': 'sentence-transformers/bert-base-nli-mean-tokens',
                         'specter': 'allenai/specter',
                         'scincl': 'malteos/scincl'}
        all_hparams = {
            'base-pt-layer': base_pt_layer[model_name],
            # Unnecessary but expected in model class.
            'score_aggregation': 'l2max',
            'fine_tune': False
        }
    
    warm_start = all_hparams.get('warm_start', False)  # Default is cold-start.
    
    if run_name:
        reps_path = os.path.join(root_path, model_name, run_name)
    else:
        reps_path = os.path.join(root_path, model_name, 'manual_run')
        
    if warm_start:
        if model_name in {'contentcf'}:
            pool_fname = os.path.join(root_path, f'test-uid2anns-{dataset}-wsccf.json')
            out_path = os.path.join(reps_path, f'test-pid2pool-{dataset}-{model_name}-wsccf-ranked.json')
        else:
            pool_fname = os.path.join(root_path, f'test-uid2anns-{dataset}-warms.json')
            out_path = os.path.join(reps_path, f'test-pid2pool-{dataset}-{model_name}-warms-ranked.json')
    else:
        if model_name in {'contentcf'}:
            simpair = all_hparams.get('simpair', False)
            if simpair:
                pool_fname = os.path.join(root_path, f'test-uid2anns-{dataset}-simpair.json')
                out_path = os.path.join(reps_path, f'test-pid2pool-{dataset}-{model_name}-simpair-ranked.json')
            else:
                pool_fname = os.path.join(root_path, f'test-uid2anns-{dataset}-ccf.json')
                out_path = os.path.join(reps_path, f'test-pid2pool-{dataset}-{model_name}-ccf-ranked.json')
        else:
            if ann_suffix in {'big', 'bids', 'assigns'}: # big is only in citeulike; bids and assign are in OR data.
                pool_fname = os.path.join(root_path, f'test-uid2anns-{dataset}-{ann_suffix}.json')
                out_path = os.path.join(reps_path, f'test-pid2pool-{dataset}-{model_name}-{ann_suffix}-ranked.json')
            
            else:
                pool_fname = os.path.join(root_path, f'test-uid2anns-{dataset}.json')
                out_path = os.path.join(reps_path, f'test-pid2pool-{dataset}-{model_name}-ranked.json')
        
    with codecs.open(pool_fname, 'r', 'utf-8') as fp:
        user_id2pool = json.load(fp)
    query_user_ids = list(user_id2pool.keys())
    logging.info(f'Read anns: {dataset}; total: {len(user_id2pool)}')
    
    # Load trained model.
    if model_name in {'specter', 'scincl', 'miswordbienc',
                      'docsbmpnet1b', 'docsbnlibert',
                      'sentsbmpnet1b', 'sentsbnlibert'}:
        trained_model = CachingTrainedScoringModelUDoc2D(all_hparams=all_hparams, model_name=model_name,
                                                         trained_model_path=trained_model_path,
                                                         model_version=model_version)
    elif model_name in {'upsentconsent', 'upnfconsent', 'upnfkpenc'}:
        trained_model = CachingTrainedScoringModelUPro2D(all_hparams=all_hparams, model_name=model_name,
                                                         trained_model_path=trained_model_path,
                                                         model_version=model_version)
    elif model_name in {'contentcf'}:
        trained_model = CachingTrainedScoringModelConCF(user_ids=query_user_ids,
                                                        all_hparams=all_hparams, model_name=model_name,
                                                        trained_model_path=trained_model_path,
                                                        model_version=model_version)
    # Read in abstracts of dataset.
    if train_suffix is None:  # If the suffix isnt passed in from cli then use the one on disk.
        train_suffix = all_hparams.get('train_suffix', None)
    # Dont use suffixes like userqabssubs or cociteabs.
    if train_suffix and train_suffix in {'tfidfcsrr', 'consent', 'sbconsent', 'goldcs'}:
        abs_fname = os.path.join(root_path, f'abstracts-{dataset}-{kp_source}-{train_suffix}.jsonl')
    else:
        abs_fname = os.path.join(root_path, f'abstracts-{dataset}-{kp_source}.jsonl')
    with codecs.open(abs_fname, 'r', 'utf-8') as fp:
        pid2abstract = {}
        for jl in fp:
            d = json.loads(jl.strip())
            pid2abstract[d['paper_id']] = d
        logging.info(f'Read: {abs_fname}')
        logging.info(f'Read docs: {len(pid2abstract)}')
    
    # Go over every query and get the query rep and the reps for the pool and generate ranking.
    query2rankedcands = collections.defaultdict(list)
    readable_dir_path = os.path.join(reps_path, f'{dataset}-{model_name}-ranked')
    du.create_dir(readable_dir_path)
    start = time.time()
    for uidx, user_id in enumerate(query_user_ids):
        # if len(query2rankedcands) == 20:
        #     break
        logging.info('Ranking query {:d}: {:s}'.format(uidx, user_id))
        cand_pids = user_id2pool[user_id]['cands']
        cand_pid_rels = user_id2pool[user_id]['relevance_adju']
        user_query_pids = user_id2pool[user_id]['uquery_pids']
        if model_name in {'contentcf'}:
            ret_dict = trained_model.predict(user_id=user_id, cand_pids=cand_pids,
                                             pid2abstract=pid2abstract)
        elif model_name in {'upsentconsent', 'upnfconsent', 'upnfkpenc'}:
            ret_dict = trained_model.predict(user_id=user_id, user_query_pids=user_query_pids,
                                             cand_pids=cand_pids, pid2abstract=pid2abstract)
        else:
            ret_dict = trained_model.predict(user_query_pids=user_query_pids,
                                             cand_pids=cand_pids, pid2abstract=pid2abstract)
        cand_scores = ret_dict['cand_scores']
        pair_softmax = ret_dict['pair_scores']
        assert(len(cand_pids) == len(cand_scores))
        # Get nearest neighbours.
        cand2sims = {}
        cand_pair_sims_string = {}
        for cpid, cand_sim, pair_sent_sm in zip(cand_pids, cand_scores, pair_softmax):
            cand2sims[cpid] = cand_sim
            cand_pair_sims_string[cpid] = (cand_sim, pair_sent_sm)
        # Build the re-ranked list of paper_ids.
        ranked_cand_pids = []
        ranked_cand_pid_rels = []
        ranked_pair_sim_strings = []
        for cpid, sim in sorted(cand2sims.items(), key=lambda i: i[1], reverse=True):
            ranked_cand_pids.append(cpid)
            rel = cand_pid_rels[cand_pids.index(cpid)]
            ranked_cand_pid_rels.append(rel)
            if len(ranked_pair_sim_strings) < to_print:
                pair_sent_sm = cand_pair_sims_string[cpid][1]
                if isinstance(pair_sent_sm, list):
                    mat = '\n'.join([np.array2string(np.around(t, 4), precision=3) for t in pair_sent_sm])
                else:
                    mat = np.array2string(pair_sent_sm, precision=3)
                string = '{:.4f}\n{:s}'.format(cand_pair_sims_string[cpid][0], mat)
                ranked_pair_sim_strings.append(string)
            query2rankedcands[user_id].append((cpid, sim))
        # Print out the neighbours but only do it for a handful users.
        if len(query2rankedcands) < 100:
            resfile = codecs.open(os.path.join(readable_dir_path, f'{user_id}-{dataset}-{model_name}-ranked.txt'),
                                  'w', 'utf-8')
            print_one_pool_nearest_neighbours(user_id=user_id, uq_docids=user_query_pids,
                                              all_neighbour_docids=ranked_cand_pids,
                                              pid2paperdata=pid2abstract, resfile=resfile,
                                              pid_relevances=ranked_cand_pid_rels,
                                              ranked_pair_sim_strings=ranked_pair_sim_strings)
            resfile.close()
    logging.info('Ranking candidates took: {:.4f}s'.format(time.time()-start))
    with codecs.open(out_path, 'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        logging.info('Wrote: {:s}'.format(fp.name))
        

def print_one_pool_nearest_neighbours(user_id, uq_docids, all_neighbour_docids, pid2paperdata, resfile, pid_relevances,
                                      ranked_pair_sim_strings=None, print_only_relevant=False):
    """
    Given the nearest neighbours indices write out the title and abstract and
    if the neighbour is cited in the query.
    :return:
    """
    resfile.write('======================================================================\n')
    resfile.write('USER_ID: {:s}\n'.format(user_id))
    for uq_docid in uq_docids:
        qtitle = pid2paperdata[uq_docid]['title']
        qabs = '\n'.join(['{:d}: {:s}'.format(i, sent) for i, sent in
                          enumerate(pid2paperdata[uq_docid]['abstract'])])
        resfile.write('PAPER_ID: {:s}\n'.format(uq_docid))
        resfile.write('TITLE: {:s}\n'.format(qtitle))
        try:
            kps = ' - '.join(pid2paperdata[uq_docid]['forecite_tags'][:5])
            resfile.write('KEYPHRASES:\n{:s}\n'.format(kps))
        except KeyError:
            pass
        resfile.write('ABSTRACT:\n{:s}\n'.format(qabs))
    resfile.write('===================================\n')
    for ranki, (ndocid, relevance) in enumerate(zip(all_neighbour_docids, pid_relevances)):
        if relevance != 1 and print_only_relevant:
            continue
        if ranki+1 > len(ranked_pair_sim_strings):
            break
        ntitle = pid2paperdata[ndocid]['title']
        nabs = '\n'.join(['{:d}: {:s}'.format(i, sent) for i, sent in
                          enumerate(pid2paperdata[ndocid]['abstract'])])
        resfile.write('RANK: {:d}\n'.format(ranki))
        resfile.write('PAPER_ID: {:s}\n'.format(ndocid))
        if isinstance(relevance, int):
            resfile.write('RELS: {:}\n'.format(relevance))
        if ranked_pair_sim_strings:
            resfile.write('Query sent sims:\n{:}\n'.format(ranked_pair_sim_strings[ranki]))
        resfile.write('TITLE: {:s}\n'.format(ntitle))
        try:
            nkps = pid2paperdata[ndocid]['forecite_tags']
            print_kps = ' - '.join(pid2paperdata[ndocid]['forecite_tags'][:5])
            resfile.write(f'NUM KPS: {len(nkps)}\n')
            resfile.write(f'KEYPHRASES: {print_kps}\n')
        except KeyError:
            pass
        resfile.write('ABSTRACT:\n{:s}\n\n'.format(nabs))
    resfile.write('======================================================================\n')
    resfile.write('\n')


def lace_als_rerank(als_run_path, lace_run_path, out_path):
    """
    Read the ALS results and re-rank them with LACE.
    - Read the ALS scores
    - Read the LACE scores
    - Rank the ALS topk by LACE scores.
    """
    re_rank_topk = 100
    # Read ALS.
    pid2topk_firststage = {}
    with codecs.open(os.path.join(als_run_path, 'test-pid2pool-tedrec-cfals-wsccf-ranked.json'),
                     'r', 'utf-8') as fp:
        pid2ranks = json.load(fp)
        print(f'ALS ranked query pids: {len(pid2ranks)}')
        for qpid, ranked in pid2ranks.items():
            cand_topk = [pid_score[0] for pid_score in ranked][:re_rank_topk]
            pid2topk_firststage[qpid] = cand_topk
    
    # Read LACE.
    pid2scores_secondstage = {}
    with codecs.open(os.path.join(lace_run_path, 'test-pid2pool-tedrec-upnfconsent-warms-ranked.json'),
                     'r', 'utf-8') as fp:
        pid2ranks = json.load(fp)
        print(f'LACE ranked query pids: {len(pid2ranks)}')
        for qpid, ranked in pid2ranks.items():
            cand2score = dict([(pid_score[0], pid_score[1]) for pid_score in ranked])
            pid2scores_secondstage[qpid] = cand2score
    
    # Re-rank the ALS retrievals with lace.
    qpid2reranked = {}
    for qpid in pid2scores_secondstage:
        first_stage_cands = pid2topk_firststage[qpid]
        re_scored_fs = []
        for pid in first_stage_cands:
            re_scored_fs.append((pid, pid2scores_secondstage[qpid][pid]))
        reranked_secstage = sorted(re_scored_fs, key=lambda t: t[1], reverse=True)
        qpid2reranked[qpid] = reranked_secstage
    
    with codecs.open(os.path.join(out_path, 'test-pid2pool-tedrec-upnfconsent-warms-ranked.json'), 'w', 'utf-8') as fp:
        json.dump(qpid2reranked, fp)
        logging.info('Wrote: {:s}'.format(fp.name))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Rank the pool for every query.
    dataset_rank_pool = subparsers.add_parser('rank_pool')
    dataset_rank_pool.add_argument('--root_path', required=True,
                                   help='Path with abstracts, sentence reps and citation info.')
    dataset_rank_pool.add_argument('--run_name', default=None,
                                   help='Path with trained sentence reps if using.')
    dataset_rank_pool.add_argument('--rep_type', required=True,
                                   choices=['specter', 'scincl', 'miswordbienc',
                                            'upsentconsent', 'upnfconsent', 'upnfkpenc', 'contentcf',
                                            'docsbmpnet1b', 'docsbnlibert',
                                            'sentsbmpnet1b', 'sentsbnlibert'],
                                   help='The kind of rep to use for nearest neighbours.')
    dataset_rank_pool.add_argument('--model_path', default=None,
                                   help='Path to directory with trained model to use for getting scoring function.')
    dataset_rank_pool.add_argument('--config_path', default=None,
                                   help='Path to a specific json config with hyperparams for a model. '
                                        'This will likely be something in the config/models_config directory')
    dataset_rank_pool.add_argument('--model_version', default='cur_best',
                                   choices=['cur_best', 'init', 'final'],
                                   help='The dataset to predict for.')
    dataset_rank_pool.add_argument('--train_suffix',
                                   choices=['tfidfcsrr', 'consent', 'sbconsent', 'goldcs'],
                                   help='The abstract file version to use.')
    dataset_rank_pool.add_argument('--dataset', required=True,
                                   choices=['citeulikea', 'citeuliket', 'tedrec',
                                            'oriclr2019', 'oriclr2020', 'oruai2019'],
                                   help='The dataset to predict for.')
    dataset_rank_pool.add_argument('--log_fname',
                                   help='File name for the log file to which logs get written.')
    dataset_rank_pool.add_argument('--ann_suffix', default=None,
                                   choices=['big', 'bids', 'assigns'])
    dataset_rank_pool.add_argument('--caching_scorer', action="store_true", default=True)
    cl_args = parser.parse_args()
    
    # If a log file was passed then write to it.
    try:
        logging.basicConfig(level='INFO', format='%(message)s',
                            filename=cl_args.log_fname)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))
    # Else just write to stdout.
    except AttributeError:
        logging.basicConfig(level='INFO', format='%(message)s',
                            stream=sys.stdout)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))
    
    if cl_args.subcommand == 'rank_pool':
        if cl_args.dataset in {'citeulikea', 'citeuliket', 'oriclr2019', 'oriclr2020', 'oruai2019'}:
            if cl_args.rep_type in {'specter', 'miswordbienc',
                                    'upsentconsent', 'upnfconsent', 'upnfkpenc', 'contentcf', 'sentsbnlibert'} \
                    and cl_args.caching_scorer:
                caching_scoringmodel_rank_pool_sent(
                    root_path=cl_args.root_path, model_name=cl_args.rep_type, dataset=cl_args.dataset,
                    run_name=cl_args.run_name, trained_model_path=cl_args.model_path,
                    model_config_path=cl_args.config_path,
                    model_version=cl_args.model_version, train_suffix=cl_args.train_suffix,
                    ann_suffix=cl_args.ann_suffix)
        elif cl_args.dataset in {'tedrec'}:
            if cl_args.rep_type in {'docsbmpnet1b', 'specter', 'miswordbienc', 'docsbnlibert',
                                    'sentsbmpnet1b', 'sentsbnlibert', 'upsentconsent',
                                    'upnfconsent', 'upnfkpenc', 'contentcf'} \
                    and cl_args.caching_scorer:
                caching_scoringmodel_rank_pool_sent(
                    root_path=cl_args.root_path, model_name=cl_args.rep_type, dataset=cl_args.dataset,
                    run_name=cl_args.run_name, trained_model_path=cl_args.model_path,
                    model_config_path=cl_args.config_path,
                    model_version=cl_args.model_version, train_suffix=cl_args.train_suffix,
                    ann_suffix=cl_args.ann_suffix)


if __name__ == '__main__':
    main()
