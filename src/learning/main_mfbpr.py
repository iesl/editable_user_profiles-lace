"""
Train a BPR model on user-item interaction data and write predictions with it.
"""
import argparse
import collections
import os
import codecs
import json

from implicit.bpr import BayesianPersonalizedRanking
from implicit.als import AlternatingLeastSquares
import numpy as np
import scipy.sparse as sp


def train_predict_mf(json_path, dataset, model_name, out_path=None):
    """
    - Load json interaction data.
    - Convert it into a set of ints.
    - Convert dict of ints for interactions to a CSR matrix.
    - Train BPR matrix factorization
    - Predict BPR matrix factorization.
    - Write out results.
    """
    out_path = os.path.join(out_path, f'test-pid2pool-{dataset}-{model_name}-wsccf-ranked.json')
    # Read train set and convert it into ints.
    uid2intidx = {}
    itemid2intidx = {}
    train_uid2items = {}
    with codecs.open(os.path.join(json_path, f'dev-uid2anns-{dataset}-wsccf.json'), 'r', 'utf-8') as fp:
        dev_uid2anns = json.load(fp)
    with codecs.open(os.path.join(json_path, f'train-uid2anns-{dataset}-wsccf.json'), 'r', 'utf-8') as fp:
        train_uid2anns = json.load(fp)
        for uid in train_uid2anns:
            uid2intidx[uid] = len(uid2intidx)
            user_items = train_uid2anns[uid]['uquery_pids']
            if uid in dev_uid2anns:  # Merge the items which were split into the dev set.
                user_items += dev_uid2anns[uid]['uquery_pids']
            for pid in user_items:
                if pid not in itemid2intidx:
                    itemid2intidx[pid] = len(itemid2intidx)
            # Get the train set in int form
            int_uid = uid2intidx[uid]
            int_items = [itemid2intidx[pid] for pid in user_items]
            train_uid2items[int_uid] = int_items
    print(f'Train: Users: {len(uid2intidx)}, Items: {len(itemid2intidx)}')
    intidx2uid = dict([(v, k) for k, v in uid2intidx.items()])
    intidx2itemid = dict([(v, k) for k, v in itemid2intidx.items()])
    print(f'Train: Users: {len(intidx2uid)}, Items: {len(intidx2itemid)}')
    
    # Read the test set in and convert it into ints.
    test_uid2item_cands = {}
    with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}-wsccf.json'), 'r', 'utf-8') as fp:
        test_uid2anns = json.load(fp)
        for uid in test_uid2anns:
            cand_pids = test_uid2anns[uid]['cands']
            rel_cands = [pid for pid, rel in zip(test_uid2anns[uid]['cands'], test_uid2anns[uid]['relevance_adju']) if rel == 1]
            assert(len(set.intersection(set(rel_cands), set(train_uid2anns[uid]['uquery_pids']))) < 1)
            int_cands = [itemid2intidx[pid] for pid in cand_pids]
            int_uid = uid2intidx[uid]
            test_uid2item_cands[int_uid] = int_cands
    print(f'Test: Users: {len(test_uid2item_cands)}')
    
    # Convert to csr matrices
    train_interactions = sp.dok_matrix((len(uid2intidx), len(itemid2intidx)), dtype=np.int8)
    
    for user_id, item_ids in train_uid2items.items():
        train_interactions[user_id, item_ids] = 1
    
    train_interactions = train_interactions.tocsr()
    print(train_interactions.shape)
    
    # Train BPR model; hyperparams from tuning on dev set.
    if model_name == 'cfbpr':
        if dataset == 'tedrec':
            factors, reg, lr = 200, 0.01, 0.1
        elif dataset in {'citeulikea', 'citeuliket'}:
            factors, reg, lr = 200, 0.01, 0.1
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        mf_model = BayesianPersonalizedRanking(factors=factors, learning_rate=lr,
                                               regularization=reg, iterations=100,
                                               verify_negative_samples=True)
    elif model_name == 'cfals':
        if dataset == 'tedrec':
            factors, reg, alpha = 50, 0.001, 1
        elif dataset in {'citeulikea', 'citeuliket'}:
            factors, reg, alpha = 200, 0.01, 40
        else:
            raise ValueError(f'Unknown dataset: {dataset}')
        mf_model = AlternatingLeastSquares(factors=factors, regularization=reg, alpha=alpha,
                                           iterations=15, calculate_training_loss=True)
    mf_model.fit(train_interactions, show_progress=False)
    
    # Make predictions with the model.
    query2rankedcands = {}
    for count, test_uid in enumerate(test_uid2item_cands):
        cand_items = set(test_uid2item_cands[test_uid])
        num_items = len(cand_items)
        ranked_items, rank_scores = mf_model.recommend(userid=test_uid, user_items=train_interactions[test_uid, :],
                                                       N=num_items, items=list(cand_items))
        ranked_pids = [(intidx2itemid[i], float(s)) for i, s in zip(ranked_items, rank_scores)]
        uid = intidx2uid[test_uid]
        query2rankedcands[uid] = ranked_pids
        if count % 1000 == 0:
            print(f'{count}; {uid}')
    
    with codecs.open(out_path, 'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        print('Wrote: {:s}'.format(fp.name))


def train_predict_popular(json_path, dataset, out_path=None):
    """
    - Load json interaction data.
    - Convert it into a set of ints.
    - Convert dict of ints for interactions to a CSR matrix.
    - Train BPR matrix factorization
    - Predict BPR matrix factorization.
    - Write out results.
    """
    out_path = os.path.join(out_path, f'test-pid2pool-{dataset}-popular-wsccf-ranked.json')
    # Read train set and convert it into ints.
    uid2intidx = {}
    itemid2intidx = {}
    train_uid2items = {}
    with codecs.open(os.path.join(json_path, f'dev-uid2anns-{dataset}-wsccf.json'), 'r', 'utf-8') as fp:
        dev_uid2anns = json.load(fp)
    with codecs.open(os.path.join(json_path, f'train-uid2anns-{dataset}-wsccf.json'), 'r', 'utf-8') as fp:
        train_uid2anns = json.load(fp)
        for uid in train_uid2anns:
            uid2intidx[uid] = len(uid2intidx)
            user_items = train_uid2anns[uid]['uquery_pids']
            if uid in dev_uid2anns:  # Merge the items which were split into the dev set.
                user_items += dev_uid2anns[uid]['uquery_pids']
            for pid in user_items:
                if pid not in itemid2intidx:
                    itemid2intidx[pid] = len(itemid2intidx)
            # Get the train set in int form
            int_uid = uid2intidx[uid]
            int_items = [itemid2intidx[pid] for pid in user_items]
            train_uid2items[int_uid] = int_items
    print(f'Train: Users: {len(uid2intidx)}, Items: {len(itemid2intidx)}')
    intidx2uid = dict([(v, k) for k, v in uid2intidx.items()])
    intidx2itemid = dict([(v, k) for k, v in itemid2intidx.items()])
    print(f'Train: Users: {len(intidx2uid)}, Items: {len(intidx2itemid)}')
    
    # Read the test set in and convert it into ints.
    test_uid2item_cands = {}
    with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}-wsccf.json'), 'r', 'utf-8') as fp:
        test_uid2anns = json.load(fp)
        for uid in test_uid2anns:
            cand_pids = test_uid2anns[uid]['cands']
            # rel_cands = [pid for pid, rel in zip(test_uid2anns[uid]['cands'], test_uid2anns[uid]['relevance_adju']) if rel == 1]
            # assert(len(set.intersection(set(rel_cands), set(train_uid2anns[uid]['uquery_pids']))) < 1)
            int_cands = [itemid2intidx[pid] for pid in cand_pids]
            int_uid = uid2intidx[uid]
            test_uid2item_cands[int_uid] = int_cands
    print(f'Test: Users: {len(test_uid2item_cands)}')
    
    item_id2count = collections.defaultdict(int)
    for user_id, item_ids in train_uid2items.items():
        for item_id in item_ids:
            item_id2count[item_id] += 1
    
    # Make predictions with the model.
    query2rankedcands = {}
    for count, test_uid in enumerate(test_uid2item_cands):
        cand_items = set(test_uid2item_cands[test_uid])
        pop_scored = [(intidx2itemid[i], item_id2count[i]) for i in cand_items]
        ranked_pids = sorted(pop_scored, key=lambda t: t[1], reverse=True)
        uid = intidx2uid[test_uid]
        query2rankedcands[uid] = ranked_pids
        if count % 1000 == 0:
            print(f'{count}; {uid}')
    
    with codecs.open(out_path, 'w', 'utf-8') as fp:
        json.dump(query2rankedcands, fp)
        print('Wrote: {:s}'.format(fp.name))


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Train the model.
    train_args = subparsers.add_parser('train_predict_mf_model')
    # Where to get what.
    train_args.add_argument('--model_name', required=True,
                            choices=['cfbpr', 'popular', 'cfals'],
                            help='The name of the model to train.')
    train_args.add_argument('--dataset', required=True,
                            choices=['citeulikea', 'citeuliket', 'tedrec'],
                            help='The dataset to train and predict on.')
    train_args.add_argument('--data_path', required=True,
                            help='Path to the jsonl dataset.')
    train_args.add_argument('--run_path', required=True,
                            help='Path to directory to save all run items to.')
    cl_args = parser.parse_args()
    
    if cl_args.subcommand == 'train_predict_mf_model':
        if cl_args.model_name in {'cfbpr', 'cfals'}:
            train_predict_mf(json_path=cl_args.data_path, out_path=cl_args.run_path,
                             dataset=cl_args.dataset, model_name=cl_args.model_name)
        elif cl_args.model_name == 'popular':
            train_predict_popular(json_path=cl_args.data_path, out_path=cl_args.run_path,
                                  dataset=cl_args.dataset)
    

if __name__ == '__main__':
    main()
        