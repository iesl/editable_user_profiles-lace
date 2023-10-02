"""
Pre-process citeulike-{t/a} papers.
- Creates json files from the raw datasets for my pipeline
- Create cold and warm start splits for various models
- Create processed files from the split data for the modeling code to consume
- Prefetch concepts for documents
"""
import ast
import copy
import csv
import itertools
import json, codecs
import collections
import os
import re
import pickle
import random
import sys

import numpy as np
import pandas as pd
from sklearn import feature_extraction as sk_featext
import spacy
import torch
from sentence_transformers import SentenceTransformer, models
from sklearn import neighbors

from . import pre_proc_buildreps, data_utils

scispacy_model = spacy.load("en_core_sci_sm")
scispacy_model.add_pipe('sentencizer')


def citeulikeA_to_json(raw_path, json_path):
    """
    Read the article text, the article tags, and the user likes.
    Write out:
    - One json file with pid2abstracts for all the papers alongside tags.
        [For now skip the tags per paper because there is one fewer line of tags
        then there are articles and a manual examination of the tags dint match up
        at the very last line. Look at this later.]
    - One json file with userid2pids.
    - One text file with the users articles and the articles tags.
    """
    # Read abstracts into dicts.
    pid2abstract = {}
    tokens_per_abs = []
    sents_per_abs = []
    with open(os.path.join(raw_path, 'raw-data.csv'), 'r', errors='ignore') as fp:
        reader = csv.DictReader(fp)
        for line in reader:
            abstract_sents = scispacy_model(line['raw.abstract'],
                                            disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                     'lemmatizer', 'parser', 'ner'])
            tokens_per_abs.append(len(line['raw.abstract'].split()))
            abstract_sents = [sent.text for sent in abstract_sents.sents]
            sents_per_abs.append(len(abstract_sents))
            d = {
                'paper_id': f'{int(line["doc.id"])-1}',
                'title': line['raw.title'],
                'abstract': abstract_sents,
                'tags': []
            }
            # The doc.id is 1 indexed so make it zero indexed for consistency.
            pid2abstract[f'{int(line["doc.id"])-1}'] = d
    print('Tokens per abstract:\n {:}'.format(pd.DataFrame(tokens_per_abs).describe()))
    print('Sentences per abstract:\n {:}'.format(pd.DataFrame(sents_per_abs).describe()))

    # Read tags for the abstracts.
    tags_f = codecs.open(os.path.join(raw_path, 'tags.dat'), 'r', 'utf-8')
    tagid2tag = {}  # The tags are zero indexed ids.
    for ti, line in enumerate(tags_f):
        tagid2tag[f'{ti}'] = line.strip()
    
    items2tags_f = codecs.open(os.path.join(raw_path, 'item-tag.dat'), 'r', 'utf-8')
    # There is one fewer line of item-tag than there are items. Ignore the final item.
    for item_id, tag_items in enumerate(items2tags_f):
        tag_ids = [i for i in tag_items.split()[1:]]
        for tid in tag_ids:
            tag = tagid2tag[tid]
            pid2abstract[f'{item_id}']['tags'].append(tag)
    
    # Read users and their article interactions.
    userid2likedpids = {}
    pids2userids = collections.defaultdict(list)
    liked_per_user = []
    with open(os.path.join(raw_path, 'users.dat'), 'r', errors='ignore') as fp:
        for userid, line in enumerate(fp):
            # The doc ids here are 0 indexed even tho the doc.id above is 1 indexed;
            # skip the first element because its a count of liked papers per user.
            likedpids = [lpid for lpid in line.strip().split()[1:]]
            assert('-1' not in likedpids)  # Make sure it isnt already zero indexed.
            assert(len(likedpids) == len(set(likedpids)))  # Make sure there are no repetitions.
            userid2likedpids[userid] = likedpids
            liked_per_user.append(len(likedpids))
            for lp in likedpids:
                pids2userids[lp].append(userid)
    print('Articles per user:\n {:}'.format(pd.DataFrame(liked_per_user).describe()))
    users_per_article = [len(us) for us in pids2userids.values()]
    print('Users per article:\n {:}'.format(pd.DataFrame(users_per_article).describe()))
    
    # Print a sample of users articles to a text file for examination.
    with codecs.open(os.path.join(json_path, 'user2item-citeulikea-sample.txt'), 'w') as fp:
        for i, (uid, likedpids) in enumerate(userid2likedpids.items()):
            fp.write(f'UID: {uid}\n')
            for lp in random.sample(likedpids, min(5, len(likedpids))):
                fp.write(f'PID: {lp}\n')
                fp.write(f'Title: {pid2abstract[lp]["title"]}\n')
                abs_text = "\n".join(pid2abstract[lp]["abstract"])
                fp.write(f'Abstract: {abs_text}\n')
            fp.write('\n')
            if i > 500:
                break
        print(f'Wrote: {fp.name}')

    with codecs.open(os.path.join(json_path, 'user2item-citeulikea.json'), 'w', 'utf-8') as fp:
        json.dump(userid2likedpids, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, 'abstracts-citeulikea.json'), 'w', 'utf-8') as fp:
        json.dump(pid2abstract, fp)
        print(f'Wrote: {fp.name}')


def citeulikeT_to_json(raw_path, json_path):
    """
    Read the article text, the article tags, and the user likes.
    Write out:
    - One json file with pid2abstracts for all the papers alongside tags.
    - One json file with userid2pids.
    - One text file with the users articles and the articles tags.
    """
    # Read abstracts into dicts.
    tokens_per_abs = []
    sents_per_abs = []
    pid2abstract = {}
    pid = 0
    with codecs.open(os.path.join(raw_path, 'rawtext.dat'), 'r', 'utf-8') as fp:
        for line in fp:
            if line.startswith('##'):
                continue
            else:
                abstract_sents = scispacy_model(line.strip(),
                                                disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                         'lemmatizer', 'parser', 'ner'])
                tokens_per_abs.append(len(line.strip().split()))
                abstract_sents = [sent.text for sent in abstract_sents.sents]
                sents_per_abs.append(len(abstract_sents))
                # For some reason the title and abstract in this dataset are fused.
                # Just split the first sequence in half and assume it to be the title
                # and first sentence.
                sent0_toks = abstract_sents[0].split()
                title, sent0 = " ".join(sent0_toks[:len(sent0_toks)//2]), \
                               " ".join(sent0_toks[len(sent0_toks)//2:])
                abstract_sents[0] = sent0
                pid2abstract[f'{pid}'] = {
                    'paper_id': f'{pid}',
                    'title': title,
                    'abstract': abstract_sents,
                    'tags': []
                }
                pid += 1
    print('Tokens per abstract:\n {:}'.format(pd.DataFrame(tokens_per_abs).describe()))
    print('Sentences per abstract:\n {:}'.format(pd.DataFrame(sents_per_abs).describe()))
    
    # Read tags for the abstracts.
    tags_f = codecs.open(os.path.join(raw_path, 'tags.dat'), 'r', 'utf-8')
    tags2items_f = codecs.open(os.path.join(raw_path, 'tag-item.dat'), 'r', 'utf-8')
    for tag, tag_items in zip(tags_f, tags2items_f):
        item_ids = [i for i in tag_items.split()[1:]]
        for itemid in item_ids:
            pid2abstract[itemid]['tags'].append(tag.strip())
    
    # Read users and their article interactions.
    userid2likedpids = {}
    pids2userids = collections.defaultdict(list)
    liked_per_user = []
    with open(os.path.join(raw_path, 'users.dat'), 'r', errors='ignore') as fp:
        for userid, line in enumerate(fp):
            likedpids = [lpid for lpid in line.strip().split()[1:]]
            assert('-1' not in likedpids)  # Make sure it isnt already zero indexed.
            assert(len(likedpids) == len(set(likedpids)))  # Make sure there are no repetitions.
            userid2likedpids[userid] = likedpids
            liked_per_user.append(len(likedpids))
            for lp in likedpids:
                pids2userids[lp].append(userid)
    print('Articles per user:\n {:}'.format(pd.DataFrame(liked_per_user).describe()))
    users_per_article = [len(us) for us in pids2userids.values()]
    print('Users per article:\n {:}'.format(pd.DataFrame(users_per_article).describe()))
    
    # Print a sample of users articles to a text file for examination.
    with codecs.open(os.path.join(json_path, 'user2item-citeuliket-sample.txt'), 'w') as fp:
        for i, (uid, likedpids) in enumerate(userid2likedpids.items()):
            fp.write(f'UID: {uid}\n')
            for lp in random.sample(likedpids, min(5, len(likedpids))):
                fp.write(f'PID: {lp}\n')
                fp.write(f'Title: {pid2abstract[lp]["title"]}\n')
                abs_text = "\n".join(pid2abstract[lp]["abstract"])
                fp.write(f'Abstract: {abs_text}\n')
            fp.write('\n')
            if i > 500:
                break
        print(f'Wrote: {fp.name}')
    
    with codecs.open(os.path.join(json_path, 'user2item-citeuliket.json'), 'w', 'utf-8') as fp:
        json.dump(userid2likedpids, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, 'abstracts-citeuliket.json'), 'w', 'utf-8') as fp:
        json.dump(pid2abstract, fp)
        print(f'Wrote: {fp.name}')


def filter_user_keyphrases(json_path, dataset):
    """
    Given a file with docs per paper get the keyphrases out and filter them 
    by frequency and by noisiness.
    """
    acronyms2expansion = {'hci': 'human computer interaction',  'nlp': 'natural language processing',
                'p2p': 'peer2peer', 'sna': 'social network analysis', 'hmm': 'hidden markov model',
                'pca': 'principle component analysis', 'ner': 'named entity recognition',
                'sts': 'science and technology studies', 'svd': 'singular value decomposition',
                'cmc': 'computer mediated communication', 'gpu': 'graphics processing unit',
                'ica': 'independent component analysis', 'lda': 'linear discriminant analysis',
                'ann': 'artificial neural network', 'crf': 'conditional random field',
                'hpc': 'high performance computing', 'networks': 'network', 'stats': 'statistics',
                'stat': 'statistics',
                'retrieval': 'information retrieval', 'recommender': 'recommender systems',
                'recommendation': 'recommender systems', 'web 20': 'web20',
                'cscw': 'computer supported cooperative work', 'bioinf': 'bioinformatics',
                'sysbio': 'systems biology', 'system biology': 'systems biology'}
    with codecs.open(os.path.join(json_path, f'abstracts-{dataset}.json'), 'r', 'utf-8') as fp:
        pid2abstract = json.load(fp)
        print(f'Papers: {len(pid2abstract)}')
        
    # Get tag counts.
    tags2mentions = collections.defaultdict(list)
    for pid, paper in pid2abstract.items():
        raw_tags = paper['tags']
        # Nominally normalize the tags - some users use _ and - instead of spaces.
        norm_tags = [re.sub('\-|\_', ' ', t).strip() for t in raw_tags]
        for nt in norm_tags:
            # Expand the popular keyphrases which are acronymized.
            if nt in acronyms2expansion:
                nt = acronyms2expansion[nt]
            tags2mentions[nt].append(pid)
    print(f'Number of normalized kps: {len(tags2mentions)}')

    # Filter keyphrases: based on number of mentions, length in characters, and presence of special chars.
    # This mostly matches forecite filtering.
    start_ignore = {'new', 'file', 'grant', 'grant2', 'for', 'references'}
    tags2mentions_filt = collections.defaultdict(list)
    for kp, mention_pids in tags2mentions.items():
        # Skip kps with fewer than 3 or greater than 5000 mentions.
        if len(mention_pids) < 5:
            continue
        # If its too long or small then skip it.
        if len(kp.strip()) > 100 or len(kp.strip()) < 5:
            continue
        # Exclude kps with special characters.
        ks = re.sub('[a-z0-9\/\.\-\_\*\']', '', kp)  # Remove valid characters.
        if ks.strip() and ks.replace(' ', '').isalnum() is False:  # If string is only special characters exclude it.
            continue
        # Filter if it starts with a number.
        words = kp.split()
        firstword = words[0]
        if firstword.isnumeric() or firstword in start_ignore:
            continue
        # If passed all filters then add it.
        tags2mentions_filt[kp] = mention_pids
    
    # Based on a manual examination of kps remove some of them.
    kps_to_remove = ['review', 'analysis', 'methods', 'method', 'methodology', 'theory', 'thesis', 'protein', 'proteins',
                     'model', 'models', 'modelling', 'human', 'social', 'algorithm', 'algorithms',
                     'information', 'data', 'tool', 'tools',  'book', 'books', 'survey', 'systems', 'graph',
                     'science', 'technology', 'development', 'programming', 'coding', 'computational',
                     'complex', 'binding', 'knowledge', 'function', 'comparison', 'variation', 'text',
                     'communication', 'collaborative', 'computation', 'context', 'adaptation',
                     'computing', 'user', 'next generation', 'education', 'reviews', 'management',
                     'experiment', 'article', 'overview', 'system', 'read', 'similarity', 'classic',
                     'highly', 'thesis', 'bayes', 'discovery', 'math', 'printed', 'reference', 'application',
                     'study', 'digital', 'mobile', 'short read', 'toread', 'tutorial', 'teaching',
                     'meta analysis', 'short reads', 'trust', 'journal', 'computer', 'folding', 'media',
                     'neural', 'library', 'target', 'feedback', 'dissertation', 'cluster', 'engineering',
                     'findpdf', 'paper', 'misc', 'nature', 'interesting', 'experimental', 'navigation',
                     'textbook', 'citation', 'tag', 'tags', 'comps', 'testing', 'machine', 'random',
                     'query', 'multiple', 'assoc', 'processing', 'publishing', 'blogs', 'feature', 'metrics',
                     'reputation', 'open source', 'biological', 'copy number', 'basic', 'open', 'journal club',
                     'pubmed', 'google', 'seminal', 'journalclub', 'libraries', 'innovation', 'qual',
                     'meta analisis de literatura', 'blog', 'global', 'storage', 'writing', 'noncoding',
                     'non coding', 'eisen journal club', 'kristina', 'citations', 'file import 09 04 28',
                     'file import 09 06 23', 'chapter', 'foundational', 'senior project', 'g doc', 'case studies'
                     'group', 'change', 'quals', 'primer', 'sigir', 'prelim', 'open access', 'proposal', 'reading',
                     'miranda', 'research', 'major proposal', 'toprint', 'project', 'background reading', 'lecture',
                     'full text', 'publications', 'course', 'arxiv', 'escience', 'scholar', 'browser']
    for kp in kps_to_remove:
        try:
            tags2mentions_filt.pop(kp)
        except KeyError:  # Some of the kps are excluded by length but keep them anyway for documentation.
            continue
    print(f'Number of filtered kps: {len(tags2mentions_filt)}')
    
    # Build a mapping from paper id to kps.
    pid2filt_kps = collections.defaultdict(set)
    for kp, mention_pids in tags2mentions_filt.items():
        for pid in mention_pids:
            pid2filt_kps[pid].add(kp)
    
    # Add the new filtered kps to the pid2abstract.
    for pid in pid2abstract:
        try:
            filt_kps = list(pid2filt_kps[pid])
        except KeyError:
            print(f'No kps: {pid}')
            filt_kps = []
        pid2abstract[pid]['tags_filt'] = filt_kps

    with codecs.open(os.path.join(json_path, f'abstracts-{dataset}-kpfilt.json'), 'w', 'utf-8') as fp:
        json.dump(pid2abstract, fp)
    

def create_itemwarmstart_splits(json_path, dataset):
    """
    Create item cold start splits without the cross-validation protocol
    following: https://arxiv.org/pdf/1609.02116.pdf; "Ask the GRU" paper.
    - Remove users who have fewer than 5 likes.
    - For each user sample 20% of their papers and make that the test set.
    """
    random.seed(592)
    with codecs.open(os.path.join(json_path, f'user2item-{dataset}.json'), 'r', 'utf-8') as fp:
        userid2likedpids = json.load(fp)
        print(f'Read {fp.name}')
    
    # Remove users who have fewer than 5 likes.
    userid2likedpids_filt = {}
    for userid, likedpids in userid2likedpids.items():
        if len(likedpids) < 5:
            continue
        else:
            userid2likedpids_filt[userid] = list(set(likedpids))
    
    # Get the papers which have more than 5 user likes; only these are
    # included in the per-user test set.
    pids2userids = collections.defaultdict(list)
    for userid, likedpids in userid2likedpids_filt.items():
        for lp in likedpids:
            pids2userids[lp].append(userid)
    valid_test_pids = set()
    for pid in pids2userids:
        if len(pids2userids[pid]) < 5:
            continue
        else:
            valid_test_pids.add(pid)
            
    # Go over the users and build a per-user training set.
    test_uid2test_pids = {}
    train_uid2ann_cands = {}
    pids_in_train = set()
    pids_in_test = set()
    train_uq_per_user = []
    skipped_users = 0
    subsampled_user = 0
    for uid, likedpids in userid2likedpids_filt.items():
        to_sample = max(int(0.2*len(likedpids)), 1)
        # Sample 20% of the liked pids as test documents and only retain the ones which are liked
        # more than 5 times.
        user_test_pids = likedpids[:to_sample]
        valid_user_test_pids = set.intersection(set(user_test_pids), valid_test_pids)
        user_query_pids = likedpids[to_sample:]
        pids_in_test.update(valid_user_test_pids)
        # Save the query pids as a set of training examples; exclude the users who have
        # too many papers for training. (a subsampling given batch size constraints)
        if len(user_query_pids) < 100:
            train_uid2ann_cands[uid] = {'uquery_pids': list(user_query_pids)}
        else:
            user_query_pids = random.sample(user_query_pids, k=100)
            train_uid2ann_cands[uid] = {'uquery_pids': list(user_query_pids)}
            subsampled_user += 1
            # print(f'Offending user: {uid}; interacted items: {len(user_query_pids)}')
        assert (len(user_query_pids) > 3)
        if len(valid_user_test_pids) > 0:
            test_uid2test_pids[uid] = {
                'test_pids': valid_user_test_pids,
                'uquery_pids': user_query_pids
            }
        pids_in_train.update(user_query_pids)
        train_uq_per_user.append(len(user_query_pids))
    assert(len(test_uid2test_pids) <= len(userid2likedpids_filt))
    
    # Go over the users and remove the papers in the test set which are never seen in train.
    test_uid2ann_cands = {}
    test_cand_pids = list(set.intersection(pids_in_train, pids_in_test))
    test_uq_per_user = []
    test_relevant_per_user = []
    for uid, test_pids in test_uid2test_pids.items():
        user_test_pids = test_uid2test_pids[uid]['test_pids']
        warmstart_pids = set.intersection(set(user_test_pids), set(test_cand_pids))
        if len(warmstart_pids) > 0:
            relevance = [1 if tpid in warmstart_pids else 0 for tpid in test_cand_pids]
            assert (sum(relevance) == len(warmstart_pids))
            test_uid2ann_cands[uid] = {
                'cands': test_cand_pids,
                'relevance_adju': relevance,
                'uquery_pids': test_uid2test_pids[uid]['uquery_pids']
            }
            test_uq_per_user.append(len(test_uid2test_pids[uid]['uquery_pids']))
            test_relevant_per_user.append(len(warmstart_pids))
            
    # Create a dev set in the same form as the train set; but with fixed sampled negative per positive.
    dev_uid2ann_cands = {}
    dev_uids = random.sample(list(train_uid2ann_cands.keys()), int(0.05 * len(train_uid2ann_cands.keys())))
    print(f'Development UIDs: {len(dev_uids)}')
    for duid in dev_uids:
        ann_d = train_uid2ann_cands.pop(duid)
        cand_neg_pids = set.difference(pids_in_train, set(ann_d['uquery_pids']))
        neg_pids = random.sample(list(cand_neg_pids), len(ann_d['uquery_pids']))
        dev_uid2ann_cands[duid] = {
            'uquery_pids': ann_d['uquery_pids'],
            'neg_pids': neg_pids
        }
    with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}-warms.json'), 'w', 'utf-8') as fp:
        json.dump(test_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'train-uid2anns-{dataset}-warms.json'), 'w', 'utf-8') as fp:
        json.dump(train_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'dev-uid2anns-{dataset}-warms.json'), 'w', 'utf-8') as fp:
        json.dump(dev_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    print('Test set query articles per user:\n {:}'.format(pd.DataFrame(test_uq_per_user).describe()))
    print(f'Total test users: {len(test_uid2ann_cands)}; Total test candidate articles: {len(test_cand_pids)}')
    print('Test relevant candidates per user:\n {:}'.format(pd.DataFrame(test_relevant_per_user).describe()))
    print('Train set query/relevant articles per user:\n {:}'.format(pd.DataFrame(train_uq_per_user).describe()))
    print(f'Total train users: {len(train_uid2ann_cands)}; Total train articles: {len(pids_in_train)}')
    print(f'Skipped train users: {skipped_users}')
    print(f'Subsampled test users: {subsampled_user}')
    

def create_itemcoldstart_splits(json_path, dataset):
    """
    Create item cold start splits without the cross-validation protocol
    following: https://arxiv.org/pdf/1609.02116.pdf; "Ask the GRU" paper.
    Sample 20% of the papers, remove papers which have fewer than 5 user
    likes and use those as the test set.
    """
    random.seed(592)
    with codecs.open(os.path.join(json_path, f'user2item-{dataset}.json'), 'r', 'utf-8') as fp:
        userid2likedpids = json.load(fp)
        print(f'Read {fp.name}')
    
    # Remove papers which have fewer than 5 user likes; This isnt necessary without
    # 5-fold cross val but keep it anyway.
    pids2userids = collections.defaultdict(list)
    for userid, likedpids in userid2likedpids.items():
        for lp in likedpids:
            pids2userids[lp].append(userid)
    valid_pids = []
    for pid in pids2userids:
        if len(pids2userids[pid]) < 5:
            continue
        else:
            valid_pids.append(pid)
    valid_pids.sort()
            
    # Sample 20% of the papers as a test set.
    test_pids = random.sample(valid_pids, int(0.2*len(valid_pids)))
    train_pids = [pid for pid in valid_pids if pid not in test_pids]
    
    # Now create "gold annotation" file where the "queries" are users,
    # candidates are test_pids, and the relevance is 1 if the user saved
    # an item; also create a train file with only the positive pids per user.
    test_uid2ann_cands = {}
    train_uid2ann_cands = {}
    test_relevant_per_user = []
    train_uq_per_user = []
    test_uq_per_user = []
    skipped_users = 0
    subsampled_user = 0
    for uid, likedpids in userid2likedpids.items():
        likedpids = set(likedpids)
        # Get me the liked pids which are in the test pids
        user_test_pids = set.intersection(likedpids, set(test_pids))
        # Get me the liked pids which are in NOT the test pids; this way something that is
        # an invalid test paper (< 5 likes) is still a valid train paper.
        user_query_pids = set.difference(likedpids, set(test_pids))
        # Save the query pids as a set of training examples.
        if 100 > len(user_query_pids) >= 3:
            train_uid2ann_cands[uid] = {'uquery_pids': list(user_query_pids)}
            train_uq_per_user.append(len(user_query_pids))
        else:
            print(f'Offending user: {uid}; interacted items: {len(user_query_pids)}')
            skipped_users += 1
        # If its not a user cold start and if the user has items in the test set
        # use the user as a test user; also limit to users with fewer than 50 query
        # articles for inference scaling for now.
        if len(user_test_pids) > 0 and len(user_query_pids) >= 50:
            relevance = [1 if tpid in user_test_pids else 0 for tpid in test_pids]
            assert(sum(relevance) == len(user_test_pids))
            # If there are too many user queries (few users with > 1000 papers) then subsample to a smaller number.
            if len(user_query_pids) > 300:
                user_query_pids = random.sample(user_query_pids, k=300)
                subsampled_user += 1
            test_uid2ann_cands[uid] = {
                'cands': test_pids,
                'relevance_adju': relevance,
                'uquery_pids': list(user_query_pids)
            }
            test_relevant_per_user.append(len(user_test_pids))
            test_uq_per_user.append(len(user_query_pids))
    # Create a dev set in the same form as the train set; but with fixed sampled negative per positive.
    dev_uid2ann_cands = {}
    dev_uids = random.sample(list(train_uid2ann_cands.keys()), int(0.05*len(train_uid2ann_cands.keys())))
    print(f'Development UIDs: {len(dev_uids)}')
    for duid in dev_uids:
        ann_d = train_uid2ann_cands.pop(duid)
        cand_neg_pids = set.difference(set(train_pids), set(ann_d['uquery_pids']))
        neg_pids = random.sample(list(cand_neg_pids), len(ann_d['uquery_pids']))
        dev_uid2ann_cands[duid] = {
            'uquery_pids': ann_d['uquery_pids'],
            'neg_pids': neg_pids
        }
    with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}-big.json'), 'w', 'utf-8') as fp:
        json.dump(test_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    # with codecs.open(os.path.join(json_path, f'train-uid2anns-{dataset}.json'), 'w', 'utf-8') as fp:
    #     json.dump(train_uid2ann_cands, fp)
    #     print(f'Wrote: {fp.name}')
    # with codecs.open(os.path.join(json_path, f'dev-uid2anns-{dataset}.json'), 'w', 'utf-8') as fp:
    #     json.dump(dev_uid2ann_cands, fp)
    #     print(f'Wrote: {fp.name}')
    print('Test set query articles per user:\n {:}'.format(pd.DataFrame(test_uq_per_user).describe()))
    print(f'Total test users: {len(test_uid2ann_cands)}; Total test candidate articles: {len(test_pids)}')
    print('Test relevant candidates per user:\n {:}'.format(pd.DataFrame(test_relevant_per_user).describe()))
    print('Train set query/relevant articles per user:\n {:}'.format(pd.DataFrame(train_uq_per_user).describe()))
    print(f'Total train users: {len(train_uid2ann_cands)}; Total train articles: {len(train_pids)}')
    print(f'Skipped train users: {skipped_users}')
    print(f'Subsampled test users: {subsampled_user}')


def create_itemwarmstart_splits_fromlace_contentcf(json_path, dataset):
    """
    Read the warmstart splits for LACE and create train, dev, and test splits
    for warmstart models: CF models and ContentCF models.
    - This is to ensure that the exact same users and items are in the train
        and test sets.
    """
    # Read the test, train, and dev sets for LACE.
    with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}-warms.json'), 'r', 'utf-8') as fp:
        test_uid2ann_cands_lace = json.load(fp)
        print(f'Read: {fp.name}')
    with codecs.open(os.path.join(json_path, f'train-uid2anns-{dataset}-warms.json'), 'r', 'utf-8') as fp:
        train_uid2ann_cands_lace = json.load(fp)
        print(f'Read: {fp.name}')
    with codecs.open(os.path.join(json_path, f'dev-uid2anns-{dataset}-warms.json'), 'r', 'utf-8') as fp:
        dev_uid2ann_cands_lace = json.load(fp)
        print(f'Read: {fp.name}')
    print('Train users: {:d}; Dev users: {:d}; Test users: {:d}'.
          format(len(train_uid2ann_cands_lace), len(dev_uid2ann_cands_lace), len(test_uid2ann_cands_lace)))
    
    # There should be no overlapping users in the lace dev and train sets.
    assert(len(set.intersection(set(dev_uid2ann_cands_lace.keys()), set(train_uid2ann_cands_lace.keys()))) == 0)
    # Merge the dev set into the train so that all users in dev are included in train.
    train_uid2ann_cands = copy.deepcopy(train_uid2ann_cands_lace)
    for dev_uid, ann_d in dev_uid2ann_cands_lace.items():
        dev_qpids = ann_d['uquery_pids']
        train_uid2ann_cands[dev_uid] = {'uquery_pids': dev_qpids}
    
    # Now for the lace dev set sample half the queries and treat them as positive and pair them with random negatives.
    dev_uid2ann_cands = {}
    dev_uids = list(dev_uid2ann_cands_lace.keys())
    for duid in dev_uids:
        user_train_pids = train_uid2ann_cands[duid]['uquery_pids']
        user_neg_pids = dev_uid2ann_cands_lace[duid]['neg_pids']
        assert(len(user_train_pids) == len(user_neg_pids))
        num_train_pids = len(user_train_pids)
        if num_train_pids < 2:  # There need to be atleast 2 for the halving to make sense.
            continue
        # Keep half the qpids for training and the other half for dev.
        split_user_dev_pids = user_train_pids[:num_train_pids // 2]
        neg_pids = user_neg_pids[:num_train_pids // 2]
        split_user_train_pids = user_train_pids[num_train_pids // 2:]
        dev_uid2ann_cands[duid] = {
            'uquery_pids': split_user_dev_pids,
            'neg_pids': neg_pids
        }
        # Overwrite the train qpids for the user.
        train_uid2ann_cands[duid] = {'uquery_pids': split_user_train_pids}
    # The test set is the same in both cases.
    test_uid2ann_cands = copy.deepcopy(test_uid2ann_cands_lace)
    print('Train users: {:d}; Dev users: {:d}; Test users: {:d}'.
          format(len(train_uid2ann_cands), len(dev_uid2ann_cands), len(test_uid2ann_cands)))
    
    # Run some checks to make sure things are correct.
    test_uids = set(test_uid2ann_cands.keys())
    dev_uids = set(dev_uid2ann_cands.keys())
    train_uids = set(train_uid2ann_cands.keys())
    # There should be no user in test and dev which is not in train.
    assert (len(set.difference(test_uids, train_uids)) == 0)
    assert (len(set.difference(dev_uids, train_uids)) == 0)
    # All the items in test should be seen at train time.
    train_cand_pids = []
    for uid, d in dev_uid2ann_cands.items():
        train_cand_pids.extend(d['uquery_pids'])
    for uid, d in train_uid2ann_cands.items():
        train_cand_pids.extend(d['uquery_pids'])
    train_cand_pids = set(train_cand_pids)
    tuid = list(test_uid2ann_cands.keys())[0]
    test_cand_pids = set(test_uid2ann_cands[tuid]['cands'])  # uid is a random uid from train.
    try:
        assert (len(set.difference(test_cand_pids, train_cand_pids)) == 0)
    except AssertionError:
        print(len(train_cand_pids))
        print(len(train_cand_pids))
        print(len(set.difference(test_cand_pids, train_cand_pids)))
        sys.exit()
    
    # Write out the results.
    with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}-wsccf.json'), 'w', 'utf-8') as fp:
        json.dump(test_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'train-uid2anns-{dataset}-wsccf.json'), 'w', 'utf-8') as fp:
        json.dump(train_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'dev-uid2anns-{dataset}-wsccf.json'), 'w', 'utf-8') as fp:
        json.dump(dev_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')


def create_itemcoldstart_splits_contentcf(json_path, dataset):
    """
    Create item cold start splits without the cross-validation protocol
    following: https://arxiv.org/pdf/1609.02116.pdf; "Ask the GRU" paper.
    Sample 20% of the papers, remove papers which have fewer than 5 user
    likes and use those as the test set.
    - The key difference here is that the dev set users are not disjoint from the
        training set users. To ensure that the content+collaborative filtering
        models do have a learned embedding for the user.
    - The train set won't filter users based on number of query or test
        items. It retains all the users in the test set. (the test set should be the
        same as the normal+big test sets used for evaling the older models)
    """
    random.seed(592)
    with codecs.open(os.path.join(json_path, f'user2item-{dataset}.json'), 'r', 'utf-8') as fp:
        userid2likedpids = json.load(fp)
        print(f'Read {fp.name}')
    
    # Remove papers which have fewer than 5 user likes; This isnt necessary without
    # 5-fold cross val but keep it anyway.
    pids2userids = collections.defaultdict(list)
    for userid, likedpids in userid2likedpids.items():
        for lp in likedpids:
            pids2userids[lp].append(userid)
    valid_pids = []
    for pid in pids2userids:
        if len(pids2userids[pid]) < 5:
            continue
        else:
            valid_pids.append(pid)
    valid_pids.sort()
    print(f'Valid pids: {len(valid_pids)}; All pids: {len(pids2userids)}')
    
    # Sample 20% of the papers as a test set.
    test_pids = random.sample(valid_pids, int(0.2 * len(valid_pids)))
    neg_cand_train_pids = [pid for pid in valid_pids if pid not in test_pids]
    
    # Now create "gold annotation" file where the "queries" are users,
    # candidates are test_pids, and the relevance is 1 if the user saved
    # an item; also create a train file with only the positive pids per user.
    test_uid2ann_cands = {}
    train_uid2ann_cands = {}
    test_relevant_per_user = []
    train_uq_per_user = []
    test_uq_per_user = []
    train_pids = set()
    skipped_users = 0
    for uid, likedpids in userid2likedpids.items():
        likedpids = set(likedpids)
        # Get me the liked pids which are in the test pids
        user_test_pids = set.intersection(likedpids, set(test_pids))
        # Get me the liked pids which are in NOT the test pids; this way something that is
        # an invalid test paper (< 5 likes) is still a valid train paper.
        user_query_pids = set.difference(likedpids, set(test_pids))
        # Save the query pids as a set of training examples.
        if len(user_test_pids) > 0 and len(user_query_pids) > 0:
            train_uid2ann_cands[uid] = {'uquery_pids': list(user_query_pids)}
            train_pids.update(user_query_pids)
            train_uq_per_user.append(len(user_query_pids))
            # If its not a user cold start and if the user has items in the test set
            # use the user as a test user
            relevance = [1 if tpid in user_test_pids else 0 for tpid in test_pids]
            assert (sum(relevance) == len(user_test_pids))
            test_uid2ann_cands[uid] = {
                'cands': test_pids,
                'relevance_adju': relevance,
                'uquery_pids': list(user_query_pids)
            }
            test_relevant_per_user.append(len(user_test_pids))
            test_uq_per_user.append(len(user_query_pids))
        else:
            skipped_users += 1
            
    # Create a dev set in the same form as the train set; but with fixed sampled negative per positive.
    dev_uid2ann_cands = {}
    dev_uids = random.sample(list(train_uid2ann_cands.keys()), int(0.05 * len(train_uid2ann_cands.keys())))
    print(f'Development UIDs: {len(dev_uids)}')
    for duid in dev_uids:
        user_train_pids = train_uid2ann_cands[duid]['uquery_pids']
        num_train_pids = len(user_train_pids)
        if num_train_pids < 2:  # There need to be atleast 2 for the halving to make sense.
            continue
        # Keep half the qpids for training and the other half for dev.
        split_user_dev_pids = user_train_pids[:num_train_pids//2]
        split_user_train_pids = user_train_pids[num_train_pids//2:]
        # Sample random negatives and save them as dev data.
        cand_neg_pids = set.difference(set(neg_cand_train_pids), set(user_train_pids))
        neg_pids = random.sample(list(cand_neg_pids), len(split_user_dev_pids))
        dev_uid2ann_cands[duid] = {
            'uquery_pids': split_user_dev_pids,
            'neg_pids': neg_pids
        }
        # Overwrite the train qpids for the user.
        train_uid2ann_cands[duid] = {'uquery_pids': split_user_train_pids}
    
    # Write out the dev splits.
    with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}-ccf.json'), 'w', 'utf-8') as fp:
        json.dump(test_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'train-uid2anns-{dataset}-ccf.json'), 'w', 'utf-8') as fp:
        json.dump(train_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'dev-uid2anns-{dataset}-ccf.json'), 'w', 'utf-8') as fp:
        json.dump(dev_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    print('Test set query articles per user:\n {:}'.format(pd.DataFrame(test_uq_per_user).describe()))
    print(f'Total test users: {len(test_uid2ann_cands)}; Total test candidate articles: {len(test_pids)}')
    print('Test relevant candidates per user:\n {:}'.format(pd.DataFrame(test_relevant_per_user).describe()))
    print('Train set query/relevant articles per user:\n {:}'.format(pd.DataFrame(train_uq_per_user).describe()))
    print(f'Total train users: {len(train_uid2ann_cands)}; Total train articles: {len(train_pids)}')
    print(f'Skipped train users: {skipped_users}')


def split_dev_test_users(json_path, dataset, ann_suffix):
    """
    Read the test set and the big-test set and chunk some 10% of the users into a separate dev set.
    """
    assert (ann_suffix in {'warms', 'wsccf'})
    random.seed(592)
    # with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}-big.json'), 'r', 'utf-8') as fp:
    #     test_uid2ann_cands_big = json.load(fp)
    #     print(f'Read: {fp.name}')
    #     print(f'Len: {len(test_uid2ann_cands_big)}')
    with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}-{ann_suffix}.json'), 'r', 'utf-8') as fp:
        test_uid2ann_cands = json.load(fp)
        print(f'Read: {fp.name}')
        print(f'Len: {len(test_uid2ann_cands)}')
    
    all_user_ids = list(test_uid2ann_cands.keys())  # + list(test_uid2ann_cands_big.keys())
    all_user_ids.sort()
    random.shuffle(all_user_ids)
    random.shuffle(all_user_ids)
    dev_user_ids = all_user_ids[:int(0.1 * len(all_user_ids))]
    test_user_ids = all_user_ids[int(0.1 * len(all_user_ids)):]
    print(f'all users: {len(all_user_ids)}; dev: {len(dev_user_ids)}; test: {len(test_user_ids)}')
    
    eval_splits = {
        'dev': dev_user_ids,
        'test': test_user_ids
    }
    
    with codecs.open(os.path.join(json_path, f'{dataset}-evaluation_splits-{ann_suffix}.json'), 'w', 'utf-8') as fp:
        json.dump(eval_splits, fp)
        print(f'Wrote: {fp.name}')


def create_processed_examples(json_path, proc_path, dataset, model_name, cold_warm_start, ann_suffix, abstract_suffix):
    """
    Given the itemcoldstart splits from create_itemcoldstart_splits write
    out uid2anns and pid2abstract files for the train and dev splits in the
    model training directory.
    :param ann_suffix:
        ccf: content collaborative filtering.
        wsccf: warmstart for content collaborative filtering.
        warms: warmstart for proposed models.
        simpair: cold start for training contentcf models for merged user editability eval.
    :param cold_warm_start: {'cold_start', 'warm_start', 'simpair'}
    """
    # Make the output examples directory.
    out_path = os.path.join(proc_path, dataset, model_name, cold_warm_start)
    data_utils.create_dir(out_path)
    
    # Setup input and out files.
    out_dev_ann = os.path.join(out_path, f'dev-uid2anns.pickle')  # Ann files never have a suffix in out_path.
    out_train_ann = os.path.join(out_path, f'train-uid2anns.pickle')
    if ann_suffix:
        assert(ann_suffix in {'ccf', 'warms', 'wsccf', 'simpair'})
        train_ann_json = os.path.join(json_path, f'train-uid2anns-{dataset}-{ann_suffix}.json')
        dev_ann_json = os.path.join(json_path, f'dev-uid2anns-{dataset}-{ann_suffix}.json')
    else:  # This is cold-start for the proposed models - the default.
        train_ann_json = os.path.join(json_path, f'train-uid2anns-{dataset}.json')
        dev_ann_json = os.path.join(json_path, f'dev-uid2anns-{dataset}.json')
    
    if abstract_suffix:
        assert (abstract_suffix in {'tfidfcsrr', 'tfidf', 'csotfidf', 'csotfidfcsrr'})
        abstract_json = os.path.join(json_path, f'abstracts-{dataset}-forecite-{abstract_suffix}.jsonl')
        out_dev_abs = os.path.join(out_path, f'dev-abstracts-{abstract_suffix}.pickle')
        out_train_abs = os.path.join(out_path, f'train-abstracts-{abstract_suffix}.pickle')
    else:
        abstract_json = os.path.join(json_path, f'abstracts-{dataset}-forecite.jsonl')
        out_dev_abs = os.path.join(out_path, f'dev-abstracts.pickle')
        out_train_abs = os.path.join(out_path, f'train-abstracts.pickle')
    
    # Read in annotation jsons and abstracts.
    with codecs.open(train_ann_json, 'r', 'utf-8') as fp:
        train_uid2anns = json.load(fp)
        print(f'Read: {fp.name}')
    with codecs.open(dev_ann_json, 'r', 'utf-8') as fp:
        dev_uid2anns = json.load(fp)
        print(f'Read: {fp.name}')

    with codecs.open(abstract_json, 'r', 'utf-8') as fp:
        pid2abstract = {}
        for jl in fp:
            d = json.loads(jl.strip())
            pid2abstract[d['paper_id']] = d

    # If its a collab filtering model write a train_uids2idx file to the proc dir.
    if model_name in {'contentcf'}:
        train_uids = list(train_uid2anns.keys())
        train_uids.sort()
        train_uid2idx = {}
        for uid in train_uids:
            train_uid2idx[uid] = len(train_uid2idx)
        with codecs.open(os.path.join(out_path, 'uid2idx.json'), 'w') as fp:
            json.dump(train_uid2idx, fp)
            print(f'Wrote: {fp.name}')

    # Get the pid2abstract for the dev and train files.
    dev_pids = set()
    for uid, anns in dev_uid2anns.items():
        upids = set(anns['uquery_pids'] + anns['neg_pids'])
        dev_pids.update(upids)
    print(f'Dev pids: {len(dev_pids)}')
    devpid2abstract = {}
    for pid in dev_pids:
        devpid2abstract[pid] = pid2abstract[pid]
    with codecs.open(out_dev_abs, 'wb') as fp:
        pickle.dump(devpid2abstract, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(out_dev_ann, 'wb') as fp:
        pickle.dump(dev_uid2anns, fp)
        print(f'Wrote: {fp.name}')

    train_pids = set()
    for uid, anns in train_uid2anns.items():
        upids = set(anns['uquery_pids'])
        train_pids.update(upids)
    print(f'Train pids: {len(train_pids)}')
    trpid2abstract = {}
    for pid in train_pids:
        trpid2abstract[pid] = pid2abstract[pid]
    with codecs.open(out_train_abs, 'wb') as fp:
        pickle.dump(trpid2abstract, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(out_train_ann, 'wb') as fp:
        pickle.dump(train_uid2anns, fp)
        print(f'Wrote: {fp.name}')


def create_pairdoc_examples(json_path, proc_path, dataset, model_name, subsample_user=False):
    """
    Given the itemcoldstart splits from create_itemcoldstart_splits write
    out train and dev files for the document level similarity model.
    """
    random.seed(720)
    # Make the output examples directory.
    out_path = os.path.join(proc_path, dataset, model_name, 'warm_start')
    data_utils.create_dir(out_path)
    
    with codecs.open(os.path.join(json_path, f'train-uid2anns-{dataset}-warms.json'), 'r', 'utf-8') as fp:
        train_uid2anns = json.load(fp)
        print(f'Read: {fp.name}')
    with codecs.open(os.path.join(json_path, f'dev-uid2anns-{dataset}-warms.json'), 'r', 'utf-8') as fp:
        dev_uid2anns = json.load(fp)
        print(f'Read: {fp.name}')
    
    with codecs.open(os.path.join(json_path, f'abstracts-{dataset}-forecite.jsonl'), 'r', 'utf-8') as fp:
        pid2abstract = {}
        for jl in fp:
            d = json.loads(jl.strip())
            pid2abstract[d['paper_id']] = d
        all_abs_pids = list(pid2abstract.keys())

    for split_str, split_uid2anns in [('train', train_uid2anns), ('dev', dev_uid2anns)]:
        if subsample_user:
            out_ex_file = codecs.open(os.path.join(out_path, f'{split_str}-userqabssubs.jsonl'), 'w', 'utf-8')
        else:
            out_ex_file = codecs.open(os.path.join(out_path, f'{split_str}-userqabs.jsonl'), 'w', 'utf-8')
        out_examples = 0
        examples_per_user = []
        for user_id, user_anns in split_uid2anns.items():
            userq_pids = user_anns['uquery_pids']
            # Generate all combinations of length 2 for the users docs.
            cidxs = list(itertools.combinations(range(len(userq_pids)), 2))
            if subsample_user:
                cidxs = random.sample(cidxs, min(len(cidxs), 500))  # citeulikeA and T 75% of users are below 500.
            examples_per_user.append(len(cidxs))
            for idxs in cidxs:
                anchor_pid = userq_pids[idxs[0]]
                pos_pid = userq_pids[idxs[1]]
                anchor_abs = {'TITLE': pid2abstract[anchor_pid]['title'],
                              'ABSTRACT': pid2abstract[anchor_pid]['abstract']}
                pos_abs = {'TITLE': pid2abstract[pos_pid]['title'],
                           'ABSTRACT': pid2abstract[pos_pid]['abstract']}
                out_ex = {
                    'user_id': user_id,
                    'cited_pids': [anchor_pid, pos_pid],
                    'query': anchor_abs,
                    'pos_context': pos_abs
                }
                # Of its dev also add a random negative context.
                if split_str == 'dev':
                    neg_pid = random.choice(all_abs_pids)
                    neg_abs = {'TITLE': pid2abstract[neg_pid]['title'],
                               'ABSTRACT': pid2abstract[neg_pid]['abstract']}
                    out_ex['neg_context'] = neg_abs
                out_ex_file.write(json.dumps(out_ex)+'\n')
                out_examples += 1
                if out_examples % 200000 == 0:
                    print(f'{split_str}; {out_examples}')
        print(f'Wrote: {out_ex_file.name}')
        out_ex_file.close()
        all_summ = pd.DataFrame(examples_per_user).describe()
        print('Examples per user: {:}'.format(all_summ))
        print(f'Number of examples: {out_examples}')
        
    
def get_abstract_forecitekps_tfidf_consent_doc(trained_kpenc_path, concepts_path, citeulike_path, dataset):
    """
    Read in abstracts
    - retrieve keyphrases for the abstracts sentences using simple term matches.
    - then re-rank these with the contextual sentence encoder and use those.
    """
    # Read in keyphrases; the output file name suffix changes based on what the source is.
    # Somewhat dirtily calling the cso concepts also forecite. shrug.
    if 'forecite' in concepts_path:
        kp_file = codecs.open(os.path.join(concepts_path, 'gorccompscicit-keyphrases-forecite-filt-cul.csv'), 'r', 'utf-8')
        kp_csv = csv.DictReader(kp_file)
        kp2kpd = {}
        for kpd in kp_csv:
            kpd['mention_pids'] = set(ast.literal_eval(kpd['mention_pids']))
            kp2kpd[kpd['keyphrase']] = kpd
        print(f'Filtered KPs: {len(kp2kpd)}')
        keyphrases = list(kp2kpd.keys())
        outfile = codecs.open(os.path.join(citeulike_path, f'abstracts-{dataset}-forecite-tfidfcsrr.jsonl'), 'w', 'utf-8')
        outfile_readable = codecs.open(os.path.join(citeulike_path, f'abstracts-{dataset}-forecite-tfidfcsrr.txt'), 'w',
                                       'utf-8')
    elif 'computer_science_ontology' in concepts_path:
        kp_file = codecs.open(os.path.join(concepts_path, 'cso-concepts.txt'), 'r', 'utf-8')
        keyphrases = [line.strip() for line in kp_file]
        outfile = codecs.open(os.path.join(citeulike_path, f'abstracts-{dataset}-forecite-csotfidfcsrr.jsonl'), 'w',
                              'utf-8')
        outfile_readable = codecs.open(os.path.join(citeulike_path, f'abstracts-{dataset}-forecite-csotfidfcsrr.txt'), 'w',
                                       'utf-8')

    # Read in abstracts.
    with codecs.open(os.path.join(citeulike_path, f'abstracts-{dataset}.json'), 'r', 'utf-8') as fp:
        pid2abstract = json.load(fp)
    abstract_stream = list(pid2abstract.items())
    print(f'Abstracts: {len(abstract_stream)}')

    # Get the dense abstract sentence embeddings.
    trained_sentenc_path = os.path.join(os.environ['CUR_PROJ_DIR'], 'model_runs',
                                        "s2orccompsci/miswordbienc/miswordbienc-2021_10_07-09_06_18-ot-best-rr1")
    pid2sentembeds_dense = pre_proc_buildreps.get_wholeabs_sent_reps(doc_stream=abstract_stream,
                                                                     model_name='miswordbienc',
                                                                     trained_model_path=trained_sentenc_path)
    print(f'Encoded abstracts: {len(pid2sentembeds_dense)}')
    flat_sent_reps_dense = []
    for pid, paper_dict in abstract_stream:
        flat_sent_reps_dense.append(pid2sentembeds_dense[pid])
    flat_sent_reps_dense = np.vstack(flat_sent_reps_dense)
    print(f'Encoded abstract sentences: {flat_sent_reps_dense.shape}')
    
    # Get the sparse abstract sentence embeddings.
    sent_stream = []
    for pid, paper_dict in abstract_stream:
        num_dense_sentence_reps = pid2sentembeds_dense[pid].shape[0]
        # Truncate the abstract for the sparse reps to the point where you have dense reps.
        for si, sent in enumerate(paper_dict['abstract'][:num_dense_sentence_reps]):
            if si == 0:
                sent_stream.append(paper_dict['title'] + ' ' + paper_dict['abstract'][si])
            else:
                sent_stream.append(paper_dict['abstract'][si])
    tfidf_vectorizer = sk_featext.text.TfidfVectorizer(
        encoding='utf-8', decode_error='strict', lowercase=True, analyzer='word',
        stop_words=None, norm='l2', use_idf=True, smooth_idf=True)
    flat_sent_reps_sparse = tfidf_vectorizer.fit_transform(sent_stream)
    print(f'Encoded abstract sentences: {flat_sent_reps_sparse.shape}')
    start_idx = 0
    pid2sentembeds_sparse = {}
    for pid, paper_dict in abstract_stream:
        num_dense_sentence_reps = pid2sentembeds_dense[pid].shape[0]  # chunk based on how many were encoded.
        pid2sentembeds_sparse[pid] = flat_sent_reps_sparse[start_idx: start_idx+num_dense_sentence_reps, :]
        start_idx += num_dense_sentence_reps
    print(f'Encoded abstracts: {len(pid2sentembeds_sparse)}')
    
    # Get the kp reps in the same space as the sentences;
    # but exclude the obviously oov keywords so they dont rank on top.
    doc_vocab = set(tfidf_vectorizer.vocabulary_.keys())
    doc_tokenizer = tfidf_vectorizer.build_tokenizer()
    non_oov_keyphrases = []
    for kp in keyphrases:
        kp_toks = set(doc_tokenizer(kp))
        if len(set.intersection(kp_toks, doc_vocab)) > 0:
            non_oov_keyphrases.append(kp)
    kp_embeddings_sparse = tfidf_vectorizer.transform(non_oov_keyphrases)
    print(f'Encoded keyphrases: {kp_embeddings_sparse.shape}')
    
    # Initialize the kpencoder model and compute keyphrase representations.
    word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased', max_seq_length=512)
    trained_model_fname = os.path.join(trained_kpenc_path, 'kp_encoder_cur_best.pt')
    word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    kp_enc_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
    kp_embeddings_dense = kp_enc_model.encode(non_oov_keyphrases)
    print(f'Encoded keyphrases: {kp_embeddings_dense.shape}')
    
    n_neighbors = 30
    index = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute')
    index.fit(kp_embeddings_sparse)
    
    # Get the nearest neighbours for the sentences in one shot.
    all_num_sents = flat_sent_reps_sparse.shape[0]
    bsize = 1000
    dense_enc_dim = kp_embeddings_dense.shape[1]
    print(f'Sentence reps: {flat_sent_reps_sparse.shape}')
    all_nearest_sparse_dists, all_nearest_sparse_idxs = np.zeros((all_num_sents, n_neighbors)), \
                                                        np.zeros((all_num_sents, n_neighbors))
    all_nearest_dists_dense = np.zeros((all_num_sents, n_neighbors))
    num_batches = all_num_sents//bsize
    start_i = 0
    for bi in range(num_batches):
        b_nearest_dists_sp, b_nearest_idxs = index.kneighbors(X=flat_sent_reps_sparse[start_i:start_i+bsize, :])
        b_nearest_kp_dense = kp_embeddings_dense[b_nearest_idxs, :]
        b_sents_dense = flat_sent_reps_dense[start_i:start_i+bsize, :]
        b_nearest_dists_dense = np.linalg.norm(np.repeat(b_sents_dense, repeats=n_neighbors, axis=0)-
                                               np.reshape(b_nearest_kp_dense, (bsize*n_neighbors, dense_enc_dim)),
                                               axis=1)
        assert(b_nearest_dists_dense.shape[0] == bsize*n_neighbors)
        all_nearest_dists_dense[start_i:start_i+bsize, :] = np.reshape(b_nearest_dists_dense, (bsize, n_neighbors))
        all_nearest_sparse_dists[start_i:start_i+bsize, :] = b_nearest_dists_sp
        all_nearest_sparse_idxs[start_i:start_i+bsize, :] = b_nearest_idxs
        if bi % 10 == 0:
            print(f'KNN batch: {bi}/{num_batches}')
        start_i += bsize
    if start_i < all_num_sents:
        b_nearest_dists_sp, b_nearest_idxs = index.kneighbors(X=flat_sent_reps_sparse[start_i:, :])
        b_nearest_kp_dense = kp_embeddings_dense[b_nearest_idxs, :]
        b_sents_dense = flat_sent_reps_dense[start_i:, :]
        end_bsize = b_sents_dense.shape[0]
        b_nearest_dists_dense = np.linalg.norm(np.repeat(b_sents_dense, repeats=n_neighbors, axis=0)-
                                               np.reshape(b_nearest_kp_dense, (end_bsize*n_neighbors, dense_enc_dim)),
                                               axis=1)
        assert(b_nearest_dists_dense.shape[0] == end_bsize*n_neighbors)
        all_nearest_dists_dense[start_i:, :] = np.reshape(b_nearest_dists_dense, (end_bsize, n_neighbors))
        all_nearest_sparse_dists[start_i:, :] = b_nearest_dists_sp
        all_nearest_sparse_idxs[start_i:, :] = b_nearest_idxs
    all_nearest_sparse_idxs = all_nearest_sparse_idxs.astype('int32')
    
    all_tags_per_paper = []
    start_idx = 0
    for pid, sentembeds_sparse in pid2sentembeds_sparse.items():
        paper_dict = pid2abstract[pid]
        num_sents = sentembeds_sparse.shape[0]
        nearest_idxs = all_nearest_sparse_idxs[start_idx:start_idx+num_sents, :]
        nearest_dists_dense = all_nearest_dists_dense[start_idx:start_idx+num_sents, :]
        start_idx += num_sents
        kpis2qdists = collections.defaultdict(list)
        valid_abs_sent_kps = []
        raw_abs_sent_kps = []
        for si in range(sentembeds_sparse.shape[0]):
            sent_kpidxs = nearest_idxs[si, :].tolist()
            sent_kpdists_dense = nearest_dists_dense[si, :].tolist()
            for skp_i, skp_dist in zip(sent_kpidxs, sent_kpdists_dense):
                kpis2qdists[skp_i].append(skp_dist)
            rr_lexical_match_kps = []
            for kpi, dense_dist in sorted(zip(sent_kpidxs, sent_kpdists_dense), key= lambda tu: tu[1]):
                # if (non_oov_keyphrases[kpi] in sent) and (len(non_oov_keyphrases[kpi]) > 4):
                rr_lexical_match_kps.append(non_oov_keyphrases[kpi])
            raw_match_kps = []
            for kpi in sent_kpidxs:
                raw_match_kps.append(non_oov_keyphrases[kpi])
            valid_abs_sent_kps.append(rr_lexical_match_kps)
            raw_abs_sent_kps.append(raw_match_kps)
        # If there are sentences missing matches then randomly use one of the other sentences.
        assert(sentembeds_sparse.shape[0] == len(valid_abs_sent_kps) == len(raw_abs_sent_kps))
        abs_sent_kps = [skp[0] for skp in valid_abs_sent_kps]
        assert(len(abs_sent_kps) == sentembeds_sparse.shape[0])
        all_tags_per_paper.append(len(abs_sent_kps))
        if len(abs_sent_kps) > 20:  # There are some papers with full-text. Truncate those right away.
            paper_dict['abstract'] = paper_dict['abstract'][:20]
            paper_dict['forecite_tags'] = abs_sent_kps[:20]
        else:
            paper_dict['forecite_tags'] = abs_sent_kps
        outfile.write(json.dumps(paper_dict)+'\n')
        if len(all_tags_per_paper) < 100000:
            outfile_readable.write(f'{pid}\n')
            outfile_readable.write('\n'.join([paper_dict['title']]+paper_dict['abstract'])+'\n')
            outfile_readable.write(f'{valid_abs_sent_kps}'+'\n')
            outfile_readable.write(f'{raw_abs_sent_kps}'+'\n')
            outfile_readable.write(f'\n')
        # else:
        #     break
        if len(all_tags_per_paper) % 1000 == 0:
            print(f'Wrote: {pid}; {len(all_tags_per_paper)}')
    print(f'Wrote: {outfile.name}')
    outfile.close()
    outfile_readable.close()
    all_summ = pd.DataFrame(all_tags_per_paper).describe()
    print('Untruncated Forecite tags per paper:\n {:}'.format(all_summ))


if __name__ == '__main__':
    # Process the raw data into jsons used in my code.
    citeulikeA_to_json(raw_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeulike-a'),
                       json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeulikea-json'))
    citeulikeT_to_json(raw_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeulike-t'),
                       json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeuliket-json'))
    # Create cold start splits for LACE.
    create_itemcoldstart_splits(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeuliket'),
                                dataset=os.path.join(os.environ['CUR_PROJ_DIR'], 'citeuliket'))
    create_itemcoldstart_splits(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeulikea'),
                                dataset=os.path.join(os.environ['CUR_PROJ_DIR'], 'citeulikea'))
    # Create warm start splits for LACE.
    create_itemwarmstart_splits(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeuliket'),
                                dataset=os.path.join(os.environ['CUR_PROJ_DIR'], 'citeuliket'))
    create_itemwarmstart_splits(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeulikea'),
                                dataset=os.path.join(os.environ['CUR_PROJ_DIR'], 'citeulikea'))
    # Create cold start splits for Hybrid model.
    create_itemcoldstart_splits_contentcf(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeuliket'),
                                          dataset=os.path.join(os.environ['CUR_PROJ_DIR'], 'citeuliket'))
    create_itemcoldstart_splits_contentcf(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeulikea'),
                                          dataset=os.path.join(os.environ['CUR_PROJ_DIR'], 'citeulikea'))
    # Create warm start splits for Hybrid model.
    create_itemwarmstart_splits_fromlace_contentcf(
        json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeulikea'), dataset='citeulikea')
    create_itemwarmstart_splits_fromlace_contentcf(
        json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeuliket'), dataset='citeuliket')
    # Create test and dev splits for warm start and warm start Hybrid model data.
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeuliket'),
                         dataset='citeuliket', ann_suffix='warms')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeulikea'),
                         dataset='citeulikea', ann_suffix='warms')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeuliket'),
                         dataset='citeuliket', ann_suffix='wsccf')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeulikea'),
                         dataset='citeulikea', ann_suffix='wsccf')
    get_abstract_forecitekps_tfidf_consent_doc(
        trained_kpenc_path=os.path.join(os.environ['CUR_PROJ_DIR'],
                                        '/model_runs/gorccompscicit/kpencconsent/kpencconsent-2022_01_19-21_54_43-scib'),
        concepts_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/gorccompscicit'),
        citeulike_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeulikea'),
        dataset='citeulikea')
    get_abstract_forecitekps_tfidf_consent_doc(
        trained_kpenc_path=os.path.join(os.environ['CUR_PROJ_DIR'],
                                        'model_runs/gorccompscicit/kpencconsent/kpencconsent-2022_01_19-21_54_43-scib'),
        concepts_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/gorccompscicit'),
        citeulike_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/citeuliket'),
        dataset='citeuliket')
