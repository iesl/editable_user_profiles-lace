"""
Batchers for content based recommendation models.
"""
import collections
import os
import codecs, json
import pickle
import pprint, sys
import copy
import random
import itertools
import re
from collections import defaultdict
import logging

import numpy as np
import torch
from transformers import AutoTokenizer

from . import data_utils as du
from . import batchers

idx_len_tup = collections.namedtuple('IdxLen', ['flat_seqi', 'seq_lens'])


class RecGenericBatcher:
    def __init__(self, ex_fnames, all_hparams):
        """
        Maintain batcher variables, state and such. Any batcher for a specific
        model is a subclass of this and implements specific methods that it
        needs.
        - A batcher needs to know how to read from an raw-file or pre-computed iterable of examples.
        - A batcher should yield a dict which you model class knows how to handle.
        :param docs_fname: string; full filename to the file with abstracts per paper.
        :param uid2anns_fname: string; fill filename to the file with user-id to positive paper-id
            mappings.
        :param batch_size: the number of examples to have in a batch.
        """
        self.docs_fname = ex_fnames['docs_fname']
        self.pid2abstract = {}
        with codecs.open(self.docs_fname, 'rb') as fp:
            self.pid2abstract = pickle.load(fp)
        print(f'Read abstracts: {len(self.pid2abstract)}')
        self.uid2anns_fname = ex_fnames['uid2anns_fname']
        self.subsample_user_articles = all_hparams.get('subsample_user_articles', False)
        self.batch_size = all_hparams['batch_size']  # Number of unique papers in a batch.
        self.batches = []
        self.uid2keyphrases = {}  # Used in the models which use per-user keyphrases.
        if 'train' in self.uid2anns_fname:
            self.build_train_batches()
            print(f'Train batches: {len(self.batches)}')
            self.is_train = True
        else:
            self.build_dev_batches()
            print(f'Dev batches: {len(self.batches)}')
            self.is_train = False
        # If number of debug batches is passed then only train on that data.
        if 'debug_batches' in all_hparams:
            self.batches = self.batches[:all_hparams['debug_batches']]
        self.num_batches = len(self.batches)
    
    def build_train_batches(self):
        """
        Read in the abstracts and the uid2anns files and build examples.
        For every user sample up to batch_size/2 papers per user
        as user interaction papers, while also building batches for in-batch negative sampling:
        - Roughly bucket users with approx the same number of papers
        - For every user, sample up to batch_size/2 papers as positive candidate,
            then from the same bucket sample another user and their papers
            for the same batch. The total number of papers should be batch_size large.
            The trade-off here is how many papers there per user profile and how user diverse
            the batch is and hence the in-batch negatives are.
        return: populates a batches attribute as, list({'uquery_pids': list(list())})
            first list is over batches; within each batch is a list of lists, the inner
            list consists of pids for a user, the outer list consists of a list per user.
        """
        # There should be atleast three papers per user to be useful for training.
        min_papers_per_user = 3
        with codecs.open(self.uid2anns_fname, 'rb') as fp:
            # The same random seed used for subsampling in the pre_proc_citeulike.create_itemcoldstart_pairdoc_examples
            # function.
            rng = random.Random(720)
            ann_dict = pickle.load(fp)
            uid2anns = {}
            for uid, ann_d in ann_dict.items():
                if self.subsample_user_articles:
                    # citeulikeA and T 75% of users are below 40 articles per user.
                    uid2anns[uid] = rng.sample(ann_d['uquery_pids'], min(len(ann_d['uquery_pids']), 40))
                else:
                    uid2anns[uid] = ann_d['uquery_pids']
        
        # Bucket users based on the number of papers they have.
        upaper_count2uid = collections.defaultdict(list)
        for uid, upapers in uid2anns.items():
            upaper_count2uid[len(upapers)].append(uid)
        # Merge the count groups which have a single user into the next list.
        # All sublists of the below list will have users with the same number of papers.
        pcount_sorted_uids = [upaper_count2uid[i] for i in sorted(upaper_count2uid.keys())]
        # All sublists of the below list will have users with APPROX the same number of papers
        compacted_csorted_uids = []
        i = 0
        while i < len(pcount_sorted_uids)-1:
            cur_uids = copy.copy(pcount_sorted_uids[i])
            if len(cur_uids) > 1:  # If there is more than one user then its okay.
                compacted_csorted_uids.append(cur_uids)
                i += 1
            else:
                # If there is a single user then add them to the next count group.
                if i < len(pcount_sorted_uids)-1:
                    compacted = cur_uids + copy.copy(pcount_sorted_uids[i+1])
                    i += 2
                    compacted_csorted_uids.append(compacted)
                # This will happen at the very end - if the final count group has a single user
                # add it to the group before it.
                else:
                    i += 1
                    compacted_csorted_uids[-1].extend(cur_uids)
        
        # Go over the users and sample papers to build their user profile.
        # Sample to_sample papers per user until their papers are exhausted.
        all_group_chunks = []
        for gi in range(len(compacted_csorted_uids)):
            cur_uids = copy.copy(compacted_csorted_uids[gi])
            group_chunks = []
            min_pcount = min([len(uid2anns[uid]) for uid in cur_uids])
            if min_pcount < self.batch_size//2:
                to_sample = min_pcount
            else:
                to_sample = self.batch_size//2
            cur_uids_track = copy.copy(cur_uids)
            while len(cur_uids_track) >= 2:
                # A batch will consist of atleast 2 user paper chunks.
                user_paper_chunks = []
                for uid in cur_uids:
                    if len(uid2anns[uid]) >= min_papers_per_user:
                        sampled_pids = set(uid2anns[uid][:to_sample])
                        uid2anns[uid] = [pid for pid in uid2anns[uid] if pid not in sampled_pids]
                        user_paper_chunks.append([(uid, s) for s in sampled_pids])
                    if len(uid2anns[uid]) < min_papers_per_user:
                        cur_uids_track.remove(uid)
                cur_uids = copy.copy(cur_uids_track)
                group_chunks.extend(user_paper_chunks)
            all_group_chunks.append(group_chunks)
        # Go over the group chunks and build batches.
        batches = []
        for group_chunks in all_group_chunks:
            group_batches = []
            batch = collections.defaultdict(list)
            for user_chunk in group_chunks:
                cur_batch_items = sum([len(batch[k]) for k in batch])
                if cur_batch_items + len(user_chunk) <= self.batch_size:
                    # Handles an edge case when diff sized chunks exist in a group.
                    # I think resulting from the compaction step; this will omit one chunk.
                    if cur_batch_items % len(user_chunk) != 0:
                        continue
                    for u, p in user_chunk:
                        batch[u].append(p)
                else:
                    upids = []  # Gets rid of the uids.
                    for uid, pids in batch.items():
                        upids.append([(uid, pid) for pid in pids])
                    # Check that there are negatives for every positive.
                    negs_okay = self._check_batch(upids)
                    if negs_okay:
                        group_batches.append({'uquery_pids': upids})
                        batch = collections.defaultdict(list)
                        for u, p in user_chunk:
                            batch[u].append(p)
                    else:
                        print(upids)
                        batch = collections.defaultdict(list)
            # Handle the final batch.
            if len(batch) >= 2:  # If there are 2 or more users.
                upids = []
                for uid, pids in batch.items():
                    upids.append([(uid, pid) for pid in pids])
                negs_okay = self._check_batch(upids)
                if negs_okay:
                    group_batches.append({'uquery_pids': upids})
            batches.extend(group_batches)
        
        self.batches = batches

    @staticmethod
    def _check_batch(upids):
        """
        :param upids: list(list(str))
        todo: this is a temporary fix and just excludes a batch which isnt correct. --low-pri.
        For any given users papers there should be equal or more other user papers so negatives
        can be created with in batch sampling.
        """
        negs_okay = True
        for ui, upid in enumerate(upids):
            cur_pids = len(upid)
            other_pids = 0
            for i, up in enumerate(upids):
                if i != ui:
                    other_pids += len(up)
            if cur_pids > other_pids:
                negs_okay = False
                break
        return negs_okay
    
    def build_dev_batches(self):
        """
        Read in the abstracts and the uid2anns files and build examples.
        For every user sample up to batch_size/2 papers per user
        as user interaction papers :
        - Roughly bucket users with approx the same number of papers
        - For every user, sample up to batch_size/2 papers as positive candidate,
            then from the same bucket sample another user and their papers
            for the same batch. The total number of papers should be batch_size large.
            The trade-off here is how many papers there per user profile and how user diverse
            the batch is and hence the in-batch negatives are.
        return: populates a batches attribute as,
            list({'uquery_pids': list(), 'neg_pids': list()})
            first list is over batches; within each batch is a list of positive pids per user,
            and a list of negative pids per each positive pid.
        """
        # There should be atleast three papers per user to be useful for training.
        min_papers_per_user = 3
        with codecs.open(self.uid2anns_fname, 'rb') as fp:
            ann_dict = pickle.load(fp)
            uid2anns_pos = {}
            uid2anns_neg = {}
            for uid, ann_d in ann_dict.items():
                uid2anns_pos[uid] = ann_d['uquery_pids']
                uid2anns_neg[uid] = ann_d['neg_pids']
    
        # Go over the users and build batches: chunk the users into up to
        # batch_size/2 papers and pair with the pre-sampled negatives.
        batches = []
        for uid in uid2anns_pos.keys():
            pos_anns = uid2anns_pos[uid]
            neg_anns = uid2anns_neg[uid]
            if len(pos_anns) < self.batch_size//2:
                to_sample = len(pos_anns)
            else:
                to_sample = self.batch_size//2
            user_batches = []
            user_batch = {}
            while len(pos_anns) >= min_papers_per_user:
                # Get positives.
                sampled_pos_pids = set(pos_anns[:to_sample])
                pos_anns = [pid for pid in pos_anns if pid not in sampled_pos_pids]
                user_batch['uquery_pids'] = [(uid, pid) for pid in sampled_pos_pids]
                # Get negatives.
                sampled_neg_pids = set(neg_anns[:to_sample])
                neg_anns = [pid for pid in neg_anns if pid not in sampled_neg_pids]
                # Add a uid even though its not the right one; this is for consistency with the train batches
                # and subsequent batchers for other models.
                user_batch['neg_pids'] = [(uid, pid) for pid in sampled_neg_pids]
                assert(len(sampled_pos_pids) == len(sampled_neg_pids))
                user_batches.append(user_batch)
                user_batch = {}
            batches.extend(user_batches)
        self.batches = batches
        
    def next_batch(self):
        """
        This should yield the dict which your model knows how to make sense of.
        :return:
        """
        raise NotImplementedError


class UserCandKPBatcher(RecGenericBatcher):
    """
    Feeds a model which inputs query, positive. Negatives are in-batch.
    """
    context_bert_config_str = None
    kp_bert_config_str = None
    return_uniq_kps = True  # This is uniq per user.
    kp2idx = None  # Keyphrase to integer index mapping. Used when the kps are updated in an embedding table.
    # This is used by the UserCandSharedKPBatcher
    uid2kps = None  # Keyphrases per user to index into a per-user embedding table.
    
    def __init__(self, ex_fnames, all_hparams):
        """
        Batcher class for the .
        This batcher is only used at train time.
        :param docs_fname: pickle/json/jsonl file with the pid2asbstract dictionary.
        :param return_uniq_kps: bool; says if the kps from the file should be uniq per user or not.
            It should not be when the ICT objective is also being used with the candidate kp pruning.
        :param uid2anns_fname: json file with the user-id to positive item/paper-id mapping.
        :param all_hparams: hyper-parameter dict with batch_size, number of training exampes
            and so on.
        """
        RecGenericBatcher.__init__(self, ex_fnames=ex_fnames, all_hparams=all_hparams)
        self.abstract_pt_lm_tokenizer = AutoTokenizer.from_pretrained(self.context_bert_config_str)
        self.kp_pt_lm_tokenizer = AutoTokenizer.from_pretrained(self.kp_bert_config_str)
        self.kps_per_abs = all_hparams.get('kps_per_abs', None)
        
    def next_batch(self):
        """
        Yield the next batch. Based on whether its training or dev yield a
        different set of items.
        :return:
            batch_doc_ids: list; with the doc_ids corresponding to the
                    examples in the batch.
            batch_dict: see make_batch.
        """
        # Shuffle at the start of every epoch.
        random.shuffle(self.batches)
        for raw_batch in self.batches:
            batch_dict = self.make_batch(raw_batch)
            batch_dict = {
                'batch_rank': batch_dict
            }
            yield None, batch_dict
    
    def make_batch(self, raw_batch):
        """
        Given a batch with user-papers build:
        - a bert batch for the papers so they can get encoded.
        - a bert batch for the keyphrases so they can get encoded.
        - indices for: 1) indexing into the positive sentence per user to create
            profile papers and candidate papers, for each candidate also
            sample negatives from the other papers and build negative indexes.
        :param raw_batch:
            dict('uquery_pids': list(list(pid))) for train batches
                outer list is over individual users papers.
            dict('uquery_pids': list(pid), 'neg_pids': list(pid)) for dev batches
                single list because processing a single users papers in dev for now.
        """
        # Happens at dev time.
        if 'neg_pids' in raw_batch:
            # Planned to have as many negs as there are positives.
            assert(len(raw_batch['uquery_pids']) == len(raw_batch['neg_pids']))
            # Get the pids for all the papers in the batch.
            batch_pids = raw_batch['uquery_pids'] + raw_batch['neg_pids']
            batch_papers = [self.pid2abstract[pid] for _, pid in batch_pids]
            # Get the abstracts batch.
            abs_bert_batch, abs_lens, abs_senttok_idxs = self.prepare_abstracts(
                batch_papers, self.abstract_pt_lm_tokenizer)
            # Get the keyphrases batch.
            kp2uniq_idx, batch_abs_kps, kp_bert_batch = self.prepare_keyphrases(
                uquery_pids=[copy.deepcopy(raw_batch['uquery_pids']), copy.deepcopy(raw_batch['neg_pids'])],
                pid2abstract=self.pid2abstract, pt_lm_tokenizer=self.kp_pt_lm_tokenizer,
                kp2idx=self.kp2idx, uid2kps=self.uid2kps,
                # If every sentence is going to get a kp then account for truncated abstracts.
                kps_per_abs=abs_lens if self.return_uniq_kps == False else self.kps_per_abs,
                return_unique=self.return_uniq_kps)
            batch_dict = self.make_dev_batch(raw_batch=raw_batch, batch_abs_kps=batch_abs_kps,
                                             kp2uniq_idx=kp2uniq_idx, abs_lens=abs_lens)
            batch_dict['abs_bert_batch'] = abs_bert_batch
            batch_dict['kp_bert_batch'] = kp_bert_batch
            batch_dict['abs_lens'] = abs_lens
            batch_dict['abs_senttok_idxs'] = abs_senttok_idxs
        # Happens at train time.
        else:
            # Get the pids for all the papers in the batch.
            batch_pids = [item for sublist in raw_batch['uquery_pids'] for item in sublist]
            batch_papers = [self.pid2abstract[pid] for _, pid in batch_pids]
            # Get the abstracts batch.
            abs_bert_batch, abs_lens, abs_senttok_idxs = self.prepare_abstracts(
                batch_papers, self.abstract_pt_lm_tokenizer)
            # Get the keyphrases batch.
            kp2uniq_idx, batch_abs_kps, kp_bert_batch = self.prepare_keyphrases(
                uquery_pids=raw_batch['uquery_pids'], pid2abstract=self.pid2abstract,
                pt_lm_tokenizer=self.kp_pt_lm_tokenizer, kp2idx=self.kp2idx, uid2kps=self.uid2kps,
                # If every sentence is going to get a kp then account for truncated abstracts.
                kps_per_abs=abs_lens if self.return_uniq_kps == False else self.kps_per_abs,
                return_unique=self.return_uniq_kps)
            assert(len(abs_lens) == len(batch_abs_kps))
            batch_dict = self.make_train_batch(raw_batch=raw_batch, batch_abs_kps=batch_abs_kps,
                                               kp2uniq_idx=kp2uniq_idx, abs_lens=abs_lens)
            batch_dict['abs_bert_batch'] = abs_bert_batch
            batch_dict['kp_bert_batch'] = kp_bert_batch
            batch_dict['abs_lens'] = abs_lens
            batch_dict['abs_senttok_idxs'] = abs_senttok_idxs
        return batch_dict
    
    @staticmethod
    def make_train_batch(raw_batch, batch_abs_kps, kp2uniq_idx, abs_lens):
        """
        Create the indices for making the training batch.
        - Assumes that a batch of sentences and kps will get encoded by a model.
        - Mainly builds indices for treating a single user paper as candidate in
            turn while using all other user papers for profile building.
        - As in in-batch negative sampling, for each target user treats non target
            users papers as negatives. (batch has been planned to have same number
            of non-target papers per each user in the batch)
        """
        # random.seed(401)
        # Get the indices which create the user sentences.
        # 1. First create a set of padding adjusted indices for all the sentences in the batch.
        max_abs_sents = max(abs_lens)
        adj_batch_abs_si = []
        for abs_i, al in enumerate(abs_lens):
            adj_senti = [abs_i*max_abs_sents+si for si in range(al)]
            adj_batch_abs_si.append(adj_senti)
        # 1.1 Create a batch of kp per abstract aptly indexing keyphrases in the flat kp list
        batch_abs_kpi = []
        u_start_i = 0
        for ui, user_pids in enumerate(raw_batch['uquery_pids']):
            num_user_papers = len(user_pids)
            user_abs_kps = batch_abs_kps[u_start_i: u_start_i+num_user_papers]
            for abs_kps in user_abs_kps:
                # For the abstract get the keyphrases as an index into a flat embedding
                # matrix which is created once per batch (either using an LM or from
                # an embedding table) from kp2uniq_idx.
                abs_kpi = [kp2uniq_idx[(ui, kp)] for kp in abs_kps]
                batch_abs_kpi.append(abs_kpi)
            u_start_i += num_user_papers
        # 2. Then for each user get profile and candidate paper sentences.
        efbatch_user_profile_si = []
        efbatch_user_profile_kpi = []
        efbatch_candabs_si = []
        efbatch_candabs_kpi = []
        efbatch_negcandabs_si = []
        efbatch_negcandabs_kpi = []
        papers_per_user = []
        u_start_i = 0
        max_user_sent, max_cand_sent = -1, -1  # Save these for padding.
        max_user_kps, max_cand_kps = -1, -1  # Save these for padding.
        for user_pids in raw_batch['uquery_pids']:
            num_user_papers = len(user_pids)
            papers_per_user.append(num_user_papers)
            # Get the current users papers sent and kp idxs from the batch sent and kp idxs.
            adj_user_abs_si = adj_batch_abs_si[u_start_i: u_start_i+num_user_papers]
            user_abs_kpi = batch_abs_kpi[u_start_i: u_start_i+num_user_papers]
            # Number of user papers should be equal to kp lists.
            assert(len(adj_user_abs_si) == len(user_abs_kpi))
            # User which are not the current users paper are the negative papers.
            all_adj_negabs_si = adj_batch_abs_si[:u_start_i] + \
                                adj_batch_abs_si[u_start_i+num_user_papers:]
            all_negabs_kpi = batch_abs_kpi[:u_start_i] + \
                             batch_abs_kpi[u_start_i+num_user_papers:]
            u_start_i += num_user_papers
            # Sample negatives for the current user from the other users in the batch.
            neg_idxs = random.sample(range(len(all_adj_negabs_si)), num_user_papers)
            adj_user_negabs_si = [copy.deepcopy(all_adj_negabs_si[i]) for i in neg_idxs]
            user_negabs_kpi = [copy.deepcopy(all_negabs_kpi[i]) for i in neg_idxs]
            assert(len(adj_user_negabs_si) == len(user_negabs_kpi))  # Each user paper should have kps.
            assert(len(adj_user_abs_si) == len(adj_user_negabs_si))  # Should be as many negs as user papers.
            efbatch_negcandabs_si.extend(adj_user_negabs_si)
            efbatch_negcandabs_kpi.extend(user_negabs_kpi)
            # Consider each user paper a candidate paper in turn to create an effective batch.
            for cur_abs_i in range(num_user_papers):
                # Save a list of sentence idxs for the papers to use for profile building.
                user_profile_si = [copy.deepcopy(adj_user_abs_si[i]) for i in range(num_user_papers) if i != cur_abs_i]
                user_profile_si = [item for sublist in user_profile_si for item in sublist]
                # Save a list of keyphrase idxs for the papers to use for profile building.
                user_profile_kpi = [copy.deepcopy(user_abs_kpi[i]) for i in range(num_user_papers) if i != cur_abs_i]
                # For the user profile use a unique set of keyphrases; even if different papers
                # have the same kp then they should get collated. This should be the same as the number
                # of kps for a given user in kp2uniq_idx.
                user_profile_kpi = list(set([item for sublist in user_profile_kpi for item in sublist]))
                user_profile_kpi.sort()
                # Get candidate sentence and kp indices.
                user_candabs_si = copy.deepcopy(adj_user_abs_si[cur_abs_i])
                user_candabs_kpi = copy.deepcopy(user_abs_kpi[cur_abs_i])
                if len(user_profile_si) > max_user_sent: max_user_sent = len(user_profile_si)
                if len(user_candabs_si) > max_cand_sent: max_cand_sent = len(user_candabs_si)
                if len(user_profile_kpi) > max_user_kps: max_user_kps = len(user_profile_kpi)
                if len(user_candabs_kpi) > max_cand_kps: max_cand_kps = len(user_candabs_kpi)
                efbatch_user_profile_si.append(user_profile_si)
                efbatch_candabs_si.append(user_candabs_si)
                efbatch_user_profile_kpi.append(user_profile_kpi)
                efbatch_candabs_kpi.append(user_candabs_kpi)
        # print([len(abs_si) for abs_si in efbatch_negcandabs_si])
        max_negcand_sent = max([len(abs_si) for abs_si in efbatch_negcandabs_si])
        max_negcand_kps = max([len(abs_si) for abs_si in efbatch_negcandabs_kpi])
        
        # The effective number of examples should be the same for all of these:
        # num_users_in_batch x num_papers_per_user.
        assert(len(efbatch_user_profile_si) == len(efbatch_user_profile_kpi) ==
               len(efbatch_candabs_si) == len(efbatch_candabs_kpi) ==
               len(efbatch_negcandabs_si) == len(efbatch_negcandabs_kpi) == sum(papers_per_user))
        # print(efbatch_user_profile_si)
        # print(efbatch_candabs_si)
        # print(efbatch_negcandabs_si)
        # 3. Pad the user, candidate, and negative sentence and kp idxs and flatten them.
        up_len, upkp_len, cand_len, candkp_len, negcand_len, negcandkp_len = [], [], [], [], [], []
        up_si, up_kpi, cand_si, cand_kpi, negc_si, negc_kpi = [], [], [], [], [], []
        # Pad with a 1-beyond last index value in the flat tensor because the model
        # will append a zero vector there and read those for pad values indices.
        si_padi = len(adj_batch_abs_si)*max_abs_sents  # The sents are flattened from a padded array.
        kpi_padi = len(kp2uniq_idx)  # The kps are a flat array of kp reps.
        for ex_up_si, ex_up_kpi, ex_cand_si, ex_cand_kpi, ex_negc_si, ex_negc_kpi in zip(
                efbatch_user_profile_si, efbatch_user_profile_kpi, efbatch_candabs_si,
                efbatch_candabs_kpi, efbatch_negcandabs_si, efbatch_negcandabs_kpi):
            # Save the lengths for zeroing out pads.
            up_len.append(len(ex_up_si)); upkp_len.append(len(ex_up_kpi))
            cand_len.append(len(ex_cand_si)); candkp_len.append(len(ex_cand_kpi))
            negcand_len.append(len(ex_negc_si)); negcandkp_len.append(len(ex_negc_kpi))
            # Pad and flatten indices.
            ex_up_si.extend([si_padi] * (max_user_sent - len(ex_up_si)))
            up_si.extend(ex_up_si)
            ex_up_kpi.extend([kpi_padi] * (max_user_kps - len(ex_up_kpi)))
            up_kpi.extend(ex_up_kpi)
            ex_cand_si.extend([si_padi] * (max_cand_sent - len(ex_cand_si)))
            cand_si.extend(ex_cand_si)
            ex_cand_kpi.extend([kpi_padi] * (max_cand_kps - len(ex_cand_kpi)))
            cand_kpi.extend(ex_cand_kpi)
            ex_negc_si.extend([si_padi] * (max_negcand_sent - len(ex_negc_si)))
            negc_si.extend(ex_negc_si)
            ex_negc_kpi.extend([kpi_padi] * (max_negcand_kps - len(ex_negc_kpi)))
            negc_kpi.extend(ex_negc_kpi)
        
        # print(up_si)
        # print(cand_si)
        # print(negc_si)
        batch_dict = {
            'user_papers': idx_len_tup(flat_seqi=torch.LongTensor(up_si), seq_lens=up_len),
            'user_kps': idx_len_tup(flat_seqi=torch.LongTensor(up_kpi), seq_lens=upkp_len),
            'cand_paper': idx_len_tup(flat_seqi=torch.LongTensor(cand_si), seq_lens=cand_len),
            'cand_kps': idx_len_tup(flat_seqi=torch.LongTensor(cand_kpi), seq_lens=candkp_len),
            'neg_paper': idx_len_tup(flat_seqi=torch.LongTensor(negc_si), seq_lens=negcand_len),
            'neg_kps': idx_len_tup(flat_seqi=torch.LongTensor(negc_kpi), seq_lens=negcandkp_len),
            'papers_per_user': papers_per_user
        }
        
        return batch_dict

    @staticmethod
    def make_dev_batch(raw_batch, batch_abs_kps, kp2uniq_idx, abs_lens):
        """
        Create the indices for making the dev batch.
        - Assumes that a batch of sentences and kps will get encoded by a model.
        - Mainly builds indices for treating a single user paper as candidate in
            turn while using the passed other user papers for profile building.
        - For each candidate paper treats the passed papers as negatives.
        """
        # Get the indices which create the user sentences.
        # 1. First create a set of padding adjusted indices for all the sentences in the batch.
        max_abs_sents = max(abs_lens)
        adj_batch_abs_si = []
        for abs_i, al in enumerate(abs_lens):
            adj_senti = [abs_i*max_abs_sents+si for si in range(al)]
            adj_batch_abs_si.append(adj_senti)
        # 1.1 Create a batch of kp per abstract aptly indexing keyphrases in the flat kp list
        batch_abs_kpi = []
        u_start_i = 0
        for ui, user_pids in enumerate([raw_batch['uquery_pids'], raw_batch['neg_pids']]):
            num_user_papers = len(user_pids)
            user_abs_kps = batch_abs_kps[u_start_i: u_start_i+num_user_papers]
            for abs_kps in user_abs_kps:
                abs_kpi = [kp2uniq_idx[(ui, kp)] for kp in abs_kps]
                batch_abs_kpi.append(abs_kpi)
            u_start_i += num_user_papers
        # 2. Then for each user get profile and candidate paper sentences.
        efbatch_user_profile_si = []
        efbatch_user_profile_kpi = []
        efbatch_candabs_si = []
        efbatch_candabs_kpi = []
        efbatch_negcandabs_si = []
        efbatch_negcandabs_kpi = []
        max_user_sent, max_cand_sent = -1, -1  # Save these for padding.
        max_user_kps, max_cand_kps = -1, -1  # Save these for padding.
        num_user_papers = len(raw_batch['uquery_pids'])
        # Get the current users papers sent and kp idxs from the batch sent and kp idxs.
        adj_user_abs_si = adj_batch_abs_si[:num_user_papers]
        user_abs_kpi = batch_abs_kpi[:num_user_papers]
        assert(len(adj_user_abs_si) == len(user_abs_kpi))  # Each user paper should have kps.
        # Papers after the positive papers are pre-sampled random negatives.
        adj_user_negabs_si = adj_batch_abs_si[num_user_papers:]
        user_negabs_kpi = batch_abs_kpi[num_user_papers:]
        assert(len(adj_user_negabs_si) == len(user_negabs_kpi))  # Each user paper should have kps.
        assert(len(adj_user_abs_si) == len(adj_user_negabs_si))  # Should be as many negs as user papers.
        efbatch_negcandabs_si.extend(adj_user_negabs_si)
        efbatch_negcandabs_kpi.extend(user_negabs_kpi)
        # Consider each user paper a candidate paper in turn to create an effective batch.
        for cur_abs_i in range(num_user_papers):
            # Save a list of sentence idxs for the papers to use for profile building.
            user_profile_si = [adj_user_abs_si[i] for i in range(num_user_papers) if i != cur_abs_i]
            user_profile_si = [item for sublist in user_profile_si for item in sublist]
            # Save a list of keyphrase idxs for the papers to use for profile building.
            user_profile_kpi = [user_abs_kpi[i] for i in range(num_user_papers) if i != cur_abs_i]
            # For the user profile use a unique set of keyphrases.
            user_profile_kpi = list(set([item for sublist in user_profile_kpi for item in sublist]))
            user_profile_kpi.sort()
            # Get candidate sentence and kp indices.
            user_candabs_si = adj_user_abs_si[cur_abs_i]
            user_candabs_kpi = user_abs_kpi[cur_abs_i]
            if len(user_profile_si) > max_user_sent: max_user_sent = len(user_profile_si)
            if len(user_candabs_si) > max_cand_sent: max_cand_sent = len(user_candabs_si)
            if len(user_profile_kpi) > max_user_kps: max_user_kps = len(user_profile_kpi)
            if len(user_candabs_kpi) > max_cand_kps: max_cand_kps = len(user_candabs_kpi)
            efbatch_user_profile_si.append(user_profile_si)
            efbatch_candabs_si.append(user_candabs_si)
            efbatch_user_profile_kpi.append(user_profile_kpi)
            efbatch_candabs_kpi.append(user_candabs_kpi)
        max_negcand_sent = max([len(abs_si) for abs_si in efbatch_negcandabs_si])
        max_negcand_kps = max([len(abs_si) for abs_si in efbatch_negcandabs_kpi])
    
        # The effective number of examples should be the same for all of these:
        # num_users_in_batch x num_papers_per_user.
        assert(len(efbatch_user_profile_si) == len(efbatch_user_profile_kpi) ==
               len(efbatch_candabs_si) == len(efbatch_candabs_kpi) ==
               len(efbatch_negcandabs_si) == len(efbatch_negcandabs_kpi))
    
        # 3. Pad the user, candidate, and negative sentence and kp idxs and flatten them.
        up_len, upkp_len, cand_len, candkp_len, negcand_len, negcandkp_len = [], [], [], [], [], []
        up_si, up_kpi, cand_si, cand_kpi, negc_si, negc_kpi = [], [], [], [], [], []
        # Pad with a 1-beyond last index value in the flat tensor because the model
        # will append a zero vector there and read those for pad values indices.
        si_padi = len(adj_batch_abs_si)*max_abs_sents  # The sents are flattened from a padded array.
        kpi_padi = len(kp2uniq_idx)  # The kps are a flat array of kp reps.
        for ex_up_si, ex_up_kpi, ex_cand_si, ex_cand_kpi, ex_negc_si, ex_negc_kpi in zip(
                efbatch_user_profile_si, efbatch_user_profile_kpi, efbatch_candabs_si,
                efbatch_candabs_kpi, efbatch_negcandabs_si, efbatch_negcandabs_kpi):
            # Save the lengths for zeroing out pads.
            up_len.append(len(ex_up_si)); upkp_len.append(len(ex_up_kpi))
            cand_len.append(len(ex_cand_si)); candkp_len.append(len(ex_cand_kpi))
            negcand_len.append(len(ex_negc_si)); negcandkp_len.append(len(ex_negc_kpi))
            # Pad and flatten indices.
            ex_up_si.extend([si_padi] * (max_user_sent - len(ex_up_si)))
            up_si.extend(ex_up_si)
            ex_up_kpi.extend([kpi_padi] * (max_user_kps - len(ex_up_kpi)))
            up_kpi.extend(ex_up_kpi)
            ex_cand_si.extend([si_padi] * (max_cand_sent - len(ex_cand_si)))
            cand_si.extend(ex_cand_si)
            ex_cand_kpi.extend([kpi_padi] * (max_cand_kps - len(ex_cand_kpi)))
            cand_kpi.extend(ex_cand_kpi)
            ex_negc_si.extend([si_padi] * (max_negcand_sent - len(ex_negc_si)))
            negc_si.extend(ex_negc_si)
            ex_negc_kpi.extend([kpi_padi] * (max_negcand_kps - len(ex_negc_kpi)))
            negc_kpi.extend(ex_negc_kpi)
        
        batch_dict = {
            'user_papers': idx_len_tup(flat_seqi=torch.LongTensor(up_si), seq_lens=up_len),
            'user_kps': idx_len_tup(flat_seqi=torch.LongTensor(up_kpi), seq_lens=upkp_len),
            'cand_paper': idx_len_tup(flat_seqi=torch.LongTensor(cand_si), seq_lens=cand_len),
            'cand_kps': idx_len_tup(flat_seqi=torch.LongTensor(cand_kpi), seq_lens=candkp_len),
            'neg_paper': idx_len_tup(flat_seqi=torch.LongTensor(negc_si), seq_lens=negcand_len),
            'neg_kps': idx_len_tup(flat_seqi=torch.LongTensor(negc_kpi), seq_lens=negcandkp_len)
        }
    
        return batch_dict

    @staticmethod
    def make_test_batch(batch_pids, pid2abstract, abstract_pt_lm_tokenizer, kp_pt_lm_tokenizer, kps_per_abs,
                        kp2idx=None, uid2kps=None, user_id=None):
        """
        Create simple batches to encode the keyphrases and abstracts.
        param user_id: string; this will be a valid string when the value is useful
            like in the upusrnfconsent models.
        """
        # Get the pids for all the papers in the batch.
        batch_papers = [pid2abstract[pid] for pid in batch_pids]
        # Get the abstracts batch.
        abs_bert_batch, abs_lens, abs_senttok_idxs = UserCandKPBatcher.prepare_abstracts(
            batch_papers, abstract_pt_lm_tokenizer)
        # Get the keyphrases batch.
        temp_batch_pids = [(user_id, pid) for pid in batch_pids]
        if user_id: # If the user id is passed then its the upusrnfconsent model. This is a bit of a poor proxy for now.
            _, batch_abs_kps, kp_bert_batch = UserCandUserKPBatcher.prepare_keyphrases(
                uquery_pids=[temp_batch_pids], pid2abstract=pid2abstract, pt_lm_tokenizer=kp_pt_lm_tokenizer,
                kps_per_abs=kps_per_abs, return_unique=False, kp2idx=kp2idx, uid2kps=uid2kps)
        else:
            _, batch_abs_kps, kp_bert_batch = UserCandKPBatcher.prepare_keyphrases(
                uquery_pids=[temp_batch_pids], pid2abstract=pid2abstract, pt_lm_tokenizer=kp_pt_lm_tokenizer,
                kps_per_abs=kps_per_abs, return_unique=False, kp2idx=kp2idx, uid2kps=uid2kps)
        assert(len(abs_lens) == len(batch_abs_kps))

        batch_dict = {
            'abs_bert_batch': abs_bert_batch,
            'kp_bert_batch': kp_bert_batch,
            'abs_lens': abs_lens,
            'batch_abs_kps': batch_abs_kps,
            'abs_senttok_idxs': abs_senttok_idxs
        }
        
        return batch_dict
    
    @staticmethod
    def prepare_abstracts(batch_abs, pt_lm_tokenizer):
        """
        Given the abstracts sentences as a list of strings prep them to pass through model.
        :param batch_abs: list(dict); list of example dicts with sentences, facets, titles.
        :return:
            bert_batch: dict(); returned from prepare_bert_sentences.
            abs_lens: list(int); number of sentences per abstract.
            sent_token_idxs: list(list(list(int))); batch_size(num_abs_sents(num_sent_tokens(ints)))
        """
        # Prepare bert batch.
        batch_abs_seqs = []
        # Add the title and abstract concated with seps because thats how SPECTER did it.
        for ex_abs in batch_abs:
            seqs = [ex_abs['title'] + ' [SEP] ']
            seqs.extend([s for s in ex_abs['abstract']])
            batch_abs_seqs.append(seqs)
        bert_batch, tokenized_abs, sent_token_idxs = batchers.AbsSentTokBatcher.prepare_bert_sentences(
            sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer)
        # Get number of sentences in abstract; some of the sentences may have been cut off
        # at some max length.
        abs_lens = []
        for bi, abs_sent_tok_idxs in enumerate(sent_token_idxs):
            num_sents = len(abs_sent_tok_idxs)
            abs_lens.append(num_sents)
            assert (num_sents > 0)
    
        return bert_batch, abs_lens, sent_token_idxs

    @staticmethod
    def prepare_keyphrases(uquery_pids, pid2abstract, pt_lm_tokenizer, kps_per_abs, return_unique=True,
                           kp2idx=None, uid2kps=None):
        """
        Given the abstracts sentences as a list of strings prep them to pass through model.
        param uquery_pids: list(list(uid, pid)); user query papers.
        param kps_per_abs: list(int) OR int; If it is a list of ints then it
        param kp2idx: dict(string: int) mapping of a keyphrase to an integer index
            for a globally shared nn.Embedding table. Supplied as a class variable from the
            main_recomm.py script. This is used in the models which update globally shared
            keyphrase/concept vectors at training time.
        param uid2kps: dict(string: list(string)) mapping of a user to a set of keyphrases.
            The position of the keyphrase in the list is the index of the keyphrase in the
            per user embedding matrix.
        :return:
            batch_kps: list(list(string));
            bert_batch: dict(); returned from prepare_bert_sentences;
        """
        if isinstance(kps_per_abs, list):
            num_papers = sum([len(up) for up in uquery_pids])
            assert(num_papers == len(kps_per_abs))
            kpi = 0
        # Go over the user papers and get the kps for the user
        # All the abstract kps of the batch across users is stored flatly.
        batch_kps = []
        kp2uniq_idx = collections.OrderedDict()
        for ui, upids in enumerate(uquery_pids):
            for uid, pid in upids:
                kps = pid2abstract[pid]['forecite_tags']
                # For now this will read the first k kps - these are sorted by relevance to the sentences score.
                if isinstance(kps_per_abs, int):
                    kps = kps[:kps_per_abs]
                elif isinstance(kps_per_abs, list):
                    to_read_kps = kps_per_abs[kpi]
                    kps = kps[:to_read_kps]
                    kpi += 1
                for kp in kps:
                    # Keep unique keyphrases per user.
                    if (ui, kp) not in kp2uniq_idx:
                        kp2uniq_idx[(ui, kp)] = len(kp2uniq_idx)
                batch_kps.append(copy.deepcopy(kps))
        
        # Prepare bert batch.
        if return_unique:  # This is unique at the user level; used for train and dev.
            batch_kps_flat = []
            for (ui, kp) in kp2uniq_idx.keys():
                batch_kps_flat.append(kp)
        # This happens at test time; the eval functions handle making things unique per user.
        # This also happens when the ICT loss is used with the cand kp pruning.
        else:
            batch_kps_flat = [item for sublist in batch_kps for item in sublist]
        if kp2idx:  # if its not None then use it.
            # Call it bert batch even if it is not.
            kp_bert_batch = torch.LongTensor([kp2idx[kp] for kp in batch_kps_flat])
        else:
            kp_bert_batch, _, _ = batchers.SentTripleBatcher.prepare_bert_sentences(
                sents=batch_kps_flat, tokenizer=pt_lm_tokenizer)
        
        return kp2uniq_idx, batch_kps, kp_bert_batch


class CFRecGenericBatcher:
    user_ids = {}
    
    def __init__(self, ex_fnames, all_hparams):
        """
        Maintain batcher variables, state and such. Any batcher for a specific
        model is a subclass of this and implements specific methods that it
        needs.
        - A batcher needs to know how to read from an raw-file or pre-computed iterable of examples.
        - A batcher should yield a dict which you model class knows how to handle.
        :param docs_fname: string; full filename to the file with abstracts per paper.
        :param uid2anns_fname: string; fill filename to the file with user-id to positive paper-id
            mappings.
        :param batch_size: the number of examples to have in a batch.
        """
        self.docs_fname = ex_fnames['docs_fname']
        # Abstracts.
        self.pid2abstract = {}
        with codecs.open(self.docs_fname, 'rb') as fp:
            self.pid2abstract = pickle.load(fp)
        print(f'Read abstracts: {len(self.pid2abstract)}')
        # User ids to integer mappings.
        self.uid2int = self.user_ids
        print(f'Number of users: {len(self.uid2int)}')
        self.uid2anns_fname = ex_fnames['uid2anns_fname']
        self.subsample_user_articles = all_hparams.get('subsample_user_articles', False)
        self.batch_size = all_hparams['batch_size']  # Number of unique papers in a batch.
        self.batches = []
        # If number of debug batches is passed then only train on that data.
        self.debug_batches = all_hparams.get('debug_batches', None)
        
        if 'train' in self.uid2anns_fname:
            self.build_train_batches()
            print(f'Train batches: {len(self.batches)}')
        else:
            self.build_dev_batches()
            print(f'Dev batches: {len(self.batches)}')

        self.num_batches = len(self.batches)
    
    def build_train_batches(self):
        """
        Read in the uid2anns files and build batches of user-document batches.
        return: populates a batches attribute as, list({'uquery_pids': list(tuple())})
            first list is over batches; within each batch is a list of tuples, the list
            is over user-paper pairs. The tuple is the user-paper tuple.
        """
        # There should be atleast three papers per user to be useful for training.
        with codecs.open(self.uid2anns_fname, 'rb') as fp:
            # The same random seed used for subsampling in the pre_proc_citeulike.create_itemcoldstart_pairdoc_examples
            # function.
            rng = random.Random(720)
            ann_dict = pickle.load(fp)
            uid2anns = {}
            for uid, ann_d in ann_dict.items():
                if self.subsample_user_articles:
                    # citeulikeA and T 75% of users are below 40 articles per user.
                    uid2anns[uid] = rng.sample(ann_d['uquery_pids'], min(len(ann_d['uquery_pids']), 40))
                else:
                    uid2anns[uid] = ann_d['uquery_pids']
        
        # Get the flat set of examples.
        examples = []
        for uid, uq_pids in uid2anns.items():
            for pid in uq_pids:
                examples.append((uid, pid))
        random.shuffle(examples)
        
        # Batch them into batch_size
        batches = []
        cur_batch = []
        for sample in examples:
            if len(cur_batch) < self.batch_size:
                cur_batch.append(sample)
            else:
                batches.append({'uquery_pids': copy.deepcopy(cur_batch)})
                cur_batch = []
        # Skip the final batch.

        # Do this here (instead of __init__) because the next_batch calls this to shuffle examples.
        if self.debug_batches:
            self.batches = batches[:self.debug_batches]
        else:
            self.batches = batches
    
    def build_dev_batches(self):
        """
        Read in the uid2anns files and build batches of user-document batches.
        return: populates a batches attribute as,
            list({'uquery_pids': list(tuple()), 'neg_pids': list(tuple())})
            first list is over batches; within each batch is a list of tuples, the list
            is over user-paper pairs. The tuple is the user-paper tuple. The negatives are
            aligned to the positives.
        """
        with codecs.open(self.uid2anns_fname, 'rb') as fp:
            ann_dict = pickle.load(fp)
            uid2anns_pos = {}
            uid2anns_neg = {}
            for uid, ann_d in ann_dict.items():
                uid2anns_pos[uid] = ann_d['uquery_pids']
                uid2anns_neg[uid] = ann_d['neg_pids']

        # Get the flat set of examples.
        pos_examples = []
        neg_examples = []
        for uid in uid2anns_pos.keys():
            pos_pids = uid2anns_pos[uid]
            neg_pids = uid2anns_neg[uid]
            for ppid, npid in zip(pos_pids, neg_pids):
                pos_examples.append((uid, ppid))
                neg_examples.append((uid, npid))

        # Batch them into batch_size
        batches = []
        cur_batch_pos = []
        cur_batch_neg = []
        for pos_sample, neg_sample in zip(pos_examples, neg_examples):
            if len(cur_batch_neg) < self.batch_size:
                cur_batch_pos.append(pos_sample)
                cur_batch_neg.append(neg_sample)
            else:
                batches.append({
                    'uquery_pids': copy.deepcopy(cur_batch_pos),
                    'neg_pids': copy.deepcopy(cur_batch_neg)
                })
                cur_batch_pos = []
                cur_batch_neg = []
        # Skip the final batch.
        self.batches = batches
    
    def next_batch(self):
        """
        This should yield the dict which your model knows how to make sense of.
        :return:
        """
        raise NotImplementedError


class UserIDCandBatcher(CFRecGenericBatcher):
    """
    Feeds a model which inputs query, positive. Negatives are in-batch.
    """
    context_bert_config_str = None
    
    def __init__(self, ex_fnames, all_hparams):
        """
        Batcher class for the .
        This batcher is only used at train time.
        :param docs_fname: pickle/json/jsonl file with the pid2asbstract dictionary.
        :param return_uniq_kps: bool; says if the kps from the file should be uniq per user or not.
            It should not be when the ICT objective is also being used with the candidate kp pruning.
        :param uid2anns_fname: json file with the user-id to positive item/paper-id mapping.
        :param all_hparams: hyper-parameter dict with batch_size, number of training examples
            and so on.
        """
        CFRecGenericBatcher.__init__(self, ex_fnames=ex_fnames, all_hparams=all_hparams)
        self.abstract_pt_lm_tokenizer = AutoTokenizer.from_pretrained(self.context_bert_config_str)
    
    def next_batch(self):
        """
        Yield the next batch. Based on whether its training or dev yield a
        different set of items.
        :return:
            batch_doc_ids: list; with the doc_ids corresponding to the
                    examples in the batch.
            batch_dict: see make_batch.
        """
        # Build a new set of batches because that shuffles the examples.
        self.build_train_batches()
        for raw_batch in self.batches:
            batch_dict = self.make_batch(raw_batch)
            batch_dict = {
                'batch_rank': batch_dict
            }
            yield None, batch_dict
    
    def make_batch(self, raw_batch):
        """
        Given a batch with user-papers build a bert batch for the papers so they can get encoded.
        :param raw_batch:
            dict('uquery_pids': list(tuple)) for train batches in-batch negative samples
                are used.
            dict('uquery_pids': list(tuple), 'neg_pids': list(tuple)) for dev batches
                positives are paired with negatives.
        """
        # Happens at dev time.
        if 'neg_pids' in raw_batch:
            # Get the pids for all the papers in the batch.
            batch_pos_papers = [self.pid2abstract[pid] for uid, pid in raw_batch['uquery_pids']]
            batch_neg_papers = [self.pid2abstract[pid] for uid, pid in raw_batch['neg_pids']]
            user_int_ids = [self.uid2int[uid] for uid, pid in raw_batch['uquery_pids']]
            # Get the abstracts batch.
            pos_abs_bert_batch, _, _ = self.prepare_abstracts(batch_pos_papers, self.abstract_pt_lm_tokenizer)
            neg_abs_bert_batch, _, _ = self.prepare_abstracts(batch_neg_papers, self.abstract_pt_lm_tokenizer)
            batch_dict = {
                'pos_bert_batch': pos_abs_bert_batch,
                'neg_bert_batch': neg_abs_bert_batch,
                'user_idxs': torch.LongTensor(user_int_ids)
            }
        # Happens at train time.
        else:
            # Get the pids for all the papers in the batch.
            batch_pos_papers = [self.pid2abstract[pid] for uid, pid in raw_batch['uquery_pids']]
            user_int_ids = [self.uid2int[uid] for uid, pid in raw_batch['uquery_pids']]
            # Get the abstracts batch.
            pos_abs_bert_batch, _, _ = self.prepare_abstracts(batch_pos_papers, self.abstract_pt_lm_tokenizer)
            batch_dict = {
                'pos_bert_batch': pos_abs_bert_batch,
                'user_idxs': torch.LongTensor(user_int_ids)
            }
        return batch_dict
    
    @staticmethod
    def make_test_batch(batch_pids, pid2abstract, abstract_pt_lm_tokenizer):
        """
        Create simple batches to encode the abstracts.
        """
        # Get the pids for all the papers in the batch.
        batch_papers = [pid2abstract[pid] for pid in batch_pids]
        # Get the abstracts batch.
        abs_bert_batch, _, _ = UserCandKPBatcher.prepare_abstracts(
            batch_papers, abstract_pt_lm_tokenizer)
        
        batch_dict = {
            'abs_bert_batch': abs_bert_batch
        }
        
        return batch_dict
    
    @staticmethod
    def prepare_abstracts(batch_abs, pt_lm_tokenizer):
        """
        Given the abstracts sentences as a list of strings prep them to pass through model.
        :param batch_abs: list(dict); list of example dicts with sentences, facets, titles.
        :return:
            bert_batch: dict(); returned from prepare_bert_sentences.
            abs_lens: list(int); number of sentences per abstract.
            sent_token_idxs: list(list(list(int))); batch_size(num_abs_sents(num_sent_tokens(ints)))
        """
        # Prepare bert batch.
        batch_abs_seqs = []
        # Add the title and abstract concated with seps because thats how SPECTER did it.
        for ex_abs in batch_abs:
            seqs = [ex_abs['title'] + ' [SEP] ']
            seqs.extend([s for s in ex_abs['abstract']])
            batch_abs_seqs.append(seqs)
        bert_batch, tokenized_abs, sent_token_idxs = batchers.AbsSentTokBatcher.prepare_bert_sentences(
            sents=batch_abs_seqs, tokenizer=pt_lm_tokenizer)
        # Get number of sentences in abstract; some of the sentences may have been cut off
        # at some max length.
        abs_lens = []
        for bi, abs_sent_tok_idxs in enumerate(sent_token_idxs):
            num_sents = len(abs_sent_tok_idxs)
            abs_lens.append(num_sents)
            assert (num_sents > 0)
        
        return bert_batch, abs_lens, sent_token_idxs
    