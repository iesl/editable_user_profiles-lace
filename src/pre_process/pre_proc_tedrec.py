"""
Pre-proc the TED recommendation dataset
- Creates json files from the raw datasets for my pipeline
- Create cold and warm start splits for various models
- Create processed files from the split data for the modeling code to consume
- Prefetch concepts for documents
"""
import copy
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
import spacy
from sentence_transformers import SentenceTransformer, models
from sklearn import neighbors

from . import pre_proc_buildreps, data_utils

scispacy_model = spacy.load("en_core_sci_sm")
scispacy_model.add_pipe('sentencizer')


def tedrec_to_json(raw_path, json_path):
    """
    Read the article text, the article tags, and the user likes.
    Write out:
    - One json file with pid2abstracts for all the papers alongside tags.
    - One json file with userid2pids.
    - One text file with the users articles and the articles tags.
    """
    # Exclude non-contentful tags.
    tags_to_exclude = {'TED Fellows', 'TED Prize',
                       'TED2009', 'TEDWomen', 'TEDxFeatured',
                       'Tedbooks', 'Tedglobal2009', 'Tedx'}
    # Read abstracts into dicts.
    pid2abstract = {}
    tokens_per_abs = []
    sents_per_abs = []
    tags_per_abs = []
    uniq_tags = set()
    # The titles are not unique and the users are listed with talk titles so make titles unique.
    title2id = {}
    with codecs.open(os.path.join(raw_path, 'ted_talks-25-Apr-2012.json'), 'r', 'utf-8') as fp:
        talk_list = json.load(fp)
        for talk_id, talkdict in enumerate(talk_list):
            abstract_sents = scispacy_model(talkdict['description'],
                                            disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                     'lemmatizer', 'parser', 'ner'])
            abstract_sents = [sent.text for sent in abstract_sents.sents if sent.text.strip() != '']
            if talkdict['title'] not in title2id:
                filt_tags = [t for t in talkdict['related_tags'] if t not in tags_to_exclude]
                d = {
                    'paper_id': f'{talk_id}',
                    'title': talkdict['title'],
                    'abstract': abstract_sents,
                    'tags': filt_tags
                }
                title2id[talkdict['title']] = f'{talk_id}'
                pid2abstract[f'{talk_id}'] = d
            else:
                continue
            tokens_per_abs.append(len(talkdict['description'].split()))
            sents_per_abs.append(len(abstract_sents))
            tags_per_abs.append(len(filt_tags))
            uniq_tags.update(filt_tags)
    print(f'Uniq tags: {len(uniq_tags)}')
    print('Tags per abstract:\n {:}'.format(pd.DataFrame(tags_per_abs).describe()))
    print('Tokens per abstract:\n {:}'.format(pd.DataFrame(tokens_per_abs).describe()))
    print('Sentences per abstract:\n {:}'.format(pd.DataFrame(sents_per_abs).describe()))
    
    # Read users and their article interactions.
    userid2likedpids = {}
    pids2userids = collections.defaultdict(list)
    liked_per_user = []
    with codecs.open(os.path.join(raw_path, 'ted_users-25-Apr-2012.json'), 'r', 'utf-8') as fp:
        user_list = json.load(fp)
        for userid, talkdict in enumerate(user_list):
            likedtalks = talkdict['favorites']
            assert (len(likedtalks) == len(set(likedtalks)))  # Make sure there are no repetitions.
            liked_ids = [title2id[title] for title in likedtalks]
            userid2likedpids[userid] = liked_ids
            liked_per_user.append(len(likedtalks))
            for lp in likedtalks:
                pids2userids[lp].append(userid)
    print('Articles per user:\n {:}'.format(pd.DataFrame(liked_per_user).describe()))
    users_per_article = [len(us) for us in pids2userids.values()]
    print('Users per article:\n {:}'.format(pd.DataFrame(users_per_article).describe()))
    
    with codecs.open(os.path.join(json_path, 'user2item-tedrec.json'), 'w', 'utf-8') as fp:
        json.dump(userid2likedpids, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, 'abstracts-tedrec.json'), 'w', 'utf-8') as fp:
        json.dump(pid2abstract, fp)
        print(f'Wrote: {fp.name}')
    
    
def create_itemwarmstart_splits(json_path, dataset):
    """
    Create item cold start splits without the cross-validation protocol.
    Sample 20% of the papers, remove users who have fewer than 12 likes
    from the whole corpus.
    Follow protocol of: https://ieeexplore.ieee.org/document/6576551
    """
    random.seed(592)
    with codecs.open(os.path.join(json_path, f'user2item-{dataset}.json'), 'r', 'utf-8') as fp:
        userid2likedpids_ori = json.load(fp)
        print(f'Read {fp.name}')
    
    # Remove users who have fewer than 12 likes.
    userid2likedpids = {}
    valid_test_pids = set()
    for userid, likedpids in userid2likedpids_ori.items():
        if len(likedpids) < 12:
            continue
        else:
            userid2likedpids[userid] = list(set(likedpids))
            valid_test_pids.update(likedpids)
    
    # Go over the users and build a per-user training set.
    test_uid2test_pids = {}
    train_uid2ann_cands = {}
    pids_in_train = set()
    pids_in_test = set()
    train_uq_per_user = []
    skipped_users = 0
    subsampled_user = 0
    for uid, likedpids in userid2likedpids.items():
        to_sample = max(int(0.2 * len(likedpids)), 1)
        # Sample 20% of the liked pids as test documents.
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
    assert (len(test_uid2test_pids) <= len(userid2likedpids))
    
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
    dev_uids = random.sample(list(train_uid2ann_cands.keys()), int(0.10 * len(train_uid2ann_cands.keys())))
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
    Create item cold start splits without the cross-validation protocol.
    Sample 20% of the papers, remove users who have fewer than 12 likes
    from the whole corpus.
    Follow protocol of: https://ieeexplore.ieee.org/document/6576551
    """
    random.seed(592)
    with codecs.open(os.path.join(json_path, f'user2item-{dataset}.json'), 'r', 'utf-8') as fp:
        userid2likedpids_ori = json.load(fp)
        print(f'Read {fp.name}')
    
    # Remove users who have fewer than 12 likes.
    valid_pids = set()
    userid2likedpids = {}
    valid_interactions = 0
    for userid, likedpids in userid2likedpids_ori.items():
        if len(likedpids) < 12:
            continue
        userid2likedpids[userid] = likedpids
        valid_pids.update(likedpids)
        valid_interactions += len(likedpids)
    valid_pids = list(valid_pids)
    valid_pids.sort()
    print(f'Valid users: {len(userid2likedpids)}')
    print(f'Valid items: {len(valid_pids)}')
    print(f'Valid interactions: {valid_interactions}')
    
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
    skipped_users = 0
    subsampled_user = 0
    for uid, likedpids in userid2likedpids.items():
        likedpids = set(likedpids)
        # Get me the liked pids which are in the test pids
        user_test_pids = set.intersection(likedpids, set(test_pids))
        # Get me the liked pids which are in NOT the test pids.
        user_query_pids = set.difference(likedpids, set(test_pids))
        # Save the query pids as a set of training examples.
        if len(user_query_pids) >= 100:  # Subsample active users for training.
            # On average users in tedrec have 10 items.
            train_uid2ann_cands[uid] = {'uquery_pids': random.sample(user_query_pids, k=10)}
        elif 100 > len(user_query_pids) >= 3:  # Min 3 is needed for batching i think.
            train_uid2ann_cands[uid] = {'uquery_pids': list(user_query_pids)}
            train_uq_per_user.append(len(user_query_pids))
        else:
            # Skip users who have fewer than 3 interactions. This is a change as of 14 Nov 2022.
            # print(f'Offending user: {uid}; interacted items: {len(user_query_pids)}')
            # skipped_users += 1
            skipped_users += 1
        # If its not a user cold start and if the user has items in the test set
        # use the user as a test user.
        if len(user_test_pids) > 0 and len(user_query_pids) > 0:
            relevance = [1 if tpid in user_test_pids else 0 for tpid in test_pids]
            assert (sum(relevance) == len(user_test_pids))
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
    dev_uids = random.sample(list(train_uid2ann_cands.keys()), int(0.10 * len(train_uid2ann_cands.keys())))
    print(f'Development UIDs: {len(dev_uids)}')
    for duid in dev_uids:
        ann_d = train_uid2ann_cands.pop(duid)
        cand_neg_pids = set.difference(set(neg_cand_train_pids), set(ann_d['uquery_pids']))
        neg_pids = random.sample(list(cand_neg_pids), len(ann_d['uquery_pids']))
        dev_uid2ann_cands[duid] = {
            'uquery_pids': ann_d['uquery_pids'],
            'neg_pids': neg_pids
        }
    with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}.json'), 'w', 'utf-8') as fp:
        json.dump(test_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'train-uid2anns-{dataset}.json'), 'w', 'utf-8') as fp:
        json.dump(train_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'dev-uid2anns-{dataset}.json'), 'w', 'utf-8') as fp:
        json.dump(dev_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    print('Test set query articles per user:\n {:}'.format(pd.DataFrame(test_uq_per_user).describe()))
    print(f'Total test users: {len(test_uid2ann_cands)}; Total test candidate articles: {len(test_pids)}')
    print('Test relevant candidates per user:\n {:}'.format(pd.DataFrame(test_relevant_per_user).describe()))
    print('Train set query/relevant articles per user:\n {:}'.format(pd.DataFrame(train_uq_per_user).describe()))
    print(f'Total train users: {len(train_uid2ann_cands)}; Total train articles: {len(neg_cand_train_pids)}')
    print(f'Total train users: {len(train_uid2ann_cands)}; Total train articles: {len(neg_cand_train_pids)}')
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
    assert (len(set.intersection(set(dev_uid2ann_cands_lace.keys()), set(train_uid2ann_cands_lace.keys()))) == 0)
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
        assert (len(user_train_pids) == len(user_neg_pids))
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
    
    with codecs.open(os.path.join(json_path, f'abstracts-{dataset}-gold-consent.jsonl'), 'r', 'utf-8') as fp:
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
                cidxs = random.sample(cidxs, min(len(cidxs), 300))  # tedrec 75% of users generate below 275 examples.
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
                out_ex_file.write(json.dumps(out_ex) + '\n')
                out_examples += 1
                if out_examples % 200000 == 0:
                    print(f'{split_str}; {out_examples}')
        print(f'Wrote: {out_ex_file.name}')
        out_ex_file.close()
        all_summ = pd.DataFrame(examples_per_user).describe()
        print('Examples per user: {:}'.format(all_summ))
        print(f'Number of examples: {out_examples}')


def split_dev_test_users(json_path, dataset, ann_suffix=None):
    """
    Read the test set and the big-test set and chunk some 10% of the users into a separate dev set.
    """
    if ann_suffix:
        assert (ann_suffix in {'warms', 'wsccf'})
        testfname = os.path.join(json_path, f'test-uid2anns-{dataset}-{ann_suffix}.json')
        outfname = os.path.join(json_path, f'{dataset}-evaluation_splits-{ann_suffix}.json')
    else:
        testfname = os.path.join(json_path, f'test-uid2anns-{dataset}.json')
        outfname = os.path.join(json_path, f'{dataset}-evaluation_splits.json')
    random.seed(592)
    with codecs.open(testfname, 'r', 'utf-8') as fp:
        test_uid2ann_cands = json.load(fp)
        print(f'Read: {fp.name}')
        print(f'Len: {len(test_uid2ann_cands)}')
    
    all_user_ids = list(test_uid2ann_cands.keys())
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
    
    with codecs.open(outfname, 'w', 'utf-8') as fp:
        json.dump(eval_splits, fp)
        print(f'Wrote: {fp.name}')


def get_abstract_kps_consent(json_path, dataset, depth=0):
    """
    Read in abstracts and retrieve keyphrases for the abstracts sentences.
    """
    # Read in abstracts.
    with codecs.open(os.path.join(json_path, f'abstracts-{dataset}.json'), 'r', 'utf-8') as fp:
        pid2abstract = json.load(fp)
    abstract_stream = list(pid2abstract.items())
    print(f'Abstracts: {len(abstract_stream)}')
    # Read in keyphrases.
    if depth == 0:  # if depth is zero use the original tags in the dataset
        keyphrases = set()
        for k, paperd in pid2abstract.items():
            keyphrases.update(paperd['tags'])
        keyphrases = list(keyphrases)
        print(f'Filtered KPs: {len(keyphrases)}')
        outfile = codecs.open(os.path.join(json_path, f'abstracts-{dataset}-gold-consent.jsonl'), 'w', 'utf-8')
        outfile_readable = codecs.open(os.path.join(json_path, f'abstracts-{dataset}-gold-consent.txt'), 'w', 'utf-8')
    else:
        raise NotImplementedError

    # Get the abstract sentence embeddings.
    pid2sentembeds = pre_proc_buildreps.get_wholeabs_sent_reps(doc_stream=abstract_stream,
                                                               model_name='sentence-transformers/all-mpnet-base-v2',
                                                               trained_model_path=None)
    print(f'Encoded abstracts: {len(pid2sentembeds)}')
    
    # Initialize the kpencoder model and compute keyphrase representations.
    kp_enc_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    kp_embeddings = kp_enc_model.encode(keyphrases)
    print(f'Encoded keyphrases: {kp_embeddings.shape}')
    
    n_neighbors = 10
    index = neighbors.NearestNeighbors(n_neighbors=n_neighbors, algorithm='brute')
    index.fit(kp_embeddings)
    
    # Get the nearest neighbours in one shot.
    all_sent_reps = np.vstack(list(pid2sentembeds.values()))
    all_num_sents = all_sent_reps.shape[0]
    bsize = 1000
    print(f'Sentence reps: {all_sent_reps.shape}')
    all_nearest_dists, all_nearest_idxs = np.zeros((all_num_sents, n_neighbors)), \
                                          np.zeros((all_num_sents, n_neighbors))
    num_batches = all_num_sents // bsize
    start_i = 0
    for bi in range(num_batches):
        b_nearest_dists, b_nearest_idxs = index.kneighbors(X=all_sent_reps[start_i:start_i + bsize, :])
        all_nearest_dists[start_i:start_i + bsize, :] = b_nearest_dists
        all_nearest_idxs[start_i:start_i + bsize, :] = b_nearest_idxs
        print(f'KNN batch: {bi}/{num_batches}')
        start_i += bsize
    if start_i < all_num_sents:
        b_nearest_dists, b_nearest_idxs = index.kneighbors(X=all_sent_reps[start_i:, :])
        all_nearest_dists[start_i:, :] = b_nearest_dists
        all_nearest_idxs[start_i:, :] = b_nearest_idxs
    all_nearest_idxs = all_nearest_idxs.astype('int32')
    all_tags_per_paper = []
    truncated_tags_per_paper = []
    start_idx = 0
    no_gold_tags = 0
    for pid, sentembeds in pid2sentembeds.items():
        paper_dict = pid2abstract[pid]
        num_sents = sentembeds.shape[0]
        nearest_dists, nearest_idxs = all_nearest_dists[start_idx:start_idx + num_sents, :], \
                                      all_nearest_idxs[start_idx:start_idx + num_sents, :]
        start_idx += num_sents
        abs_sent_kps = []
        abs_sent_kps_readable = []
        uniq_abs_sent_kps = set()
        for si in range(sentembeds.shape[0]):
            sent_kpidxs = nearest_idxs[si, :].tolist()
            sent_kpdists = nearest_dists[si, :].tolist()
            sent_kps = [keyphrases[ki] for ki in sent_kpidxs]
            # Redundant may be good for performance so keep it that way.
            abs_sent_kps.append(sent_kps[0])
            abs_sent_kps_readable.append([(keyphrases[ki], '{:.4f}'.format(d)) for
                                          ki, d in zip(sent_kpidxs, sent_kpdists)])
        all_tags_per_paper.append(len(abs_sent_kps))
        # Get at most 10 of the keyphrases and truncate very long documents.
        if sentembeds.shape[0] > 10:
            paper_dict['abstract'] = paper_dict['abstract'][:10]
            paper_dict['forecite_tags'] = abs_sent_kps[:10]
        else:
            paper_dict['forecite_tags'] = abs_sent_kps[:sentembeds.shape[0]]
        truncated_tags_per_paper.append(len(paper_dict['forecite_tags']))
        outfile.write(json.dumps(paper_dict) + '\n')
        outfile_readable.write(f'{pid}\n')
        outfile_readable.write(
            '\n'.join([f'{s}\n{t}\n' for s, t in zip(paper_dict['abstract'], abs_sent_kps_readable)]))
        outfile_readable.write(f'\n')
        # if len(all_tags_per_paper) > 50:
        #     break
    print(f'Wrote: {outfile.name}')
    outfile.close()
    outfile_readable.close()
    all_summ = pd.DataFrame(all_tags_per_paper).describe()
    print('Untruncated Forecite tags per paper:\n {:}'.format(all_summ))
    all_summ = pd.DataFrame(truncated_tags_per_paper).describe()
    print('Truncated Forecite tags per paper:\n {:}'.format(all_summ))
    print(f'Papers without gold tags: {no_gold_tags}')


def create_processed_examples(json_path, proc_path, dataset, model_name, cold_warm_start, ann_suffix, abstract_suffix):
    """
    Given the itemcoldstart splits from create_itemcoldstart_splits write
    out uid2anns and pid2abstract files for the train and dev splits in the
    model training directory.
    :param ann_suffix:
        ccf: content collaborative filtering.
        wsccf: warmstart for content collaborative filtering.
        warms: warmstart for proposed models.
        simpair: cold start for training contentcf models for merged user editability eval -- UNUSED HERE.
    :param cold_warm_start: {'cold_start', 'warm_start'}
    """
    # Make the output examples directory.
    out_path = os.path.join(proc_path, dataset, model_name, cold_warm_start)
    data_utils.create_dir(out_path)
    
    # Setup input and out files.
    out_dev_ann = os.path.join(out_path, f'dev-uid2anns.pickle')  # Ann files never have a suffix in out_path.
    out_train_ann = os.path.join(out_path, f'train-uid2anns.pickle')
    if ann_suffix:
        assert (ann_suffix in {'ccf', 'warms', 'wsccf'})
        train_ann_json = os.path.join(json_path, f'train-uid2anns-{dataset}-{ann_suffix}.json')
        dev_ann_json = os.path.join(json_path, f'dev-uid2anns-{dataset}-{ann_suffix}.json')
    else:  # This is cold-start for the proposed models - the default.
        train_ann_json = os.path.join(json_path, f'train-uid2anns-{dataset}.json')
        dev_ann_json = os.path.join(json_path, f'dev-uid2anns-{dataset}.json')
    
    if abstract_suffix:
        assert (abstract_suffix in {'consent', 'sbconsent', 'goldcs',
                                    'consentd1', 'consentd2', 'consentd3'})
        abstract_json = os.path.join(json_path, f'abstracts-{dataset}-gold-{abstract_suffix}.jsonl')
        out_dev_abs = os.path.join(out_path, f'dev-abstracts-{abstract_suffix}.pickle')
        out_train_abs = os.path.join(out_path, f'train-abstracts-{abstract_suffix}.pickle')
    else:
        abstract_json = os.path.join(json_path, f'abstracts-{dataset}-gold.jsonl')
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
    
    
if __name__ == '__main__':
    tedrec_to_json(raw_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/ted_rec/TED_dataset'),
                   json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/tedrec-json'))
    create_itemcoldstart_splits(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/tedrec'),
                                dataset='tedrec')
    create_itemwarmstart_splits(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/tedrec'),
                                dataset='tedrec')
    create_itemwarmstart_splits_fromlace_contentcf(
        json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/tedrec'), dataset='tedrec')
    get_abstract_kps_consent(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/tedrec'),
                             dataset='tedrec')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/tedrec'),
                         dataset='tedrec')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/tedrec'),
                         dataset='tedrec', ann_suffix='wsccf')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/tedrec'),
                         dataset='tedrec', ann_suffix='warms')
