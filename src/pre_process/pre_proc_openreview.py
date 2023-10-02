"""
Pre-process the peer review assignment and bidding data from Openreview.
Pre-proc the data for ICLR2018-2020, and UAI 2019.
- Creates json files from the raw datasets for my pipeline
- Create cold start splits for various models
- Create processed files from the split data for the modeling code to consume
- Prefetch concepts for documents
"""
import itertools
import json, codecs
import collections
import os
import re
import pickle
import random
import statistics

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


def read_bids_2019onward(raw_path, train_uid2ann_cands, test_pids):
    """
    2019 onward the bids are formatted differently.
    """
    str2rating = {
        'Very High': 3, 'High': 2, 'Neutral': 1, 'No Bid': 0, 'Low': -1, 'Very Low': -2
    }
    bids_missing_in_test = set()
    authors_missing_in_train = set()
    with codecs.open(os.path.join(raw_path, 'source_data', 'bids', 'bids.json'), 'r', 'utf-8') as fp:
        author2bids = json.load(fp)
        author2bids_norm = {}
        for aname in author2bids:
            try:
                assert (aname in train_uid2ann_cands)
            except AssertionError:
                authors_missing_in_train.add(aname)
                continue
            norm_bids = {}
            for bid in author2bids[aname]:
                try:
                    assert (bid['forum'] in test_pids)
                except AssertionError:
                    # print(aname, bid['forum'])
                    bids_missing_in_test.add(bid['forum'])
                    continue
                norm_bids[bid['forum']] = str2rating[bid['tag']]
            author2bids_norm[aname] = norm_bids
    print(f'Bid missing in test: {len(bids_missing_in_test)}; '
          f'Authors missing in train: {len(authors_missing_in_train)}')
    print(f'Authors with bids: {len(author2bids_norm)}')
    return author2bids_norm


def read_bids_2019pre(raw_path, train_uid2ann_cands, test_pids):
    """
    2018 the bids are per submission in a jsonl. Turns out the 2018 ICLR
    also has a bids.json file but i dint know -- just letting this function be now.
    """
    str2rating = {
        'I want to review': 3, 'I can review': 2,
        'I can probably review but am not an expert': 1, 'No bid': 0, 'I cannot review': -1
    }
    bids_missing_in_test = set()
    authors_missing_in_train = set()
    author2bids_norm = collections.defaultdict(list)
    sub_fnames = os.listdir(os.path.join(raw_path, 'source_data', 'bids'))
    sub_fnames.remove('bids.json')  # idek why this is in this directory along side the other files. tch >_<
    for test_pid in sub_fnames:
        with codecs.open(os.path.join(raw_path, 'source_data', 'bids', test_pid), 'r', 'utf-8') as fp:
            for line in fp:
                bid = json.loads(line.strip())
                try:
                    assert (bid['signature'] in train_uid2ann_cands)
                except AssertionError:
                    authors_missing_in_train.add(bid['signature'])
                    continue
                try:
                    assert (bid['forum'] in test_pids)
                except AssertionError:
                    bids_missing_in_test.add(bid['forum'])
                    continue
                author2bids_norm[bid['signature']].append((bid['forum'], str2rating[bid['tag']]))
    author2bids_norm = dict(author2bids_norm)
    for auid in author2bids_norm:
        author2bids_norm[auid] = dict(author2bids_norm[auid])
    print(f'Bid missing in test: {len(bids_missing_in_test)}; '
          f'Authors missing in train: {len(authors_missing_in_train)}')
    print(f'Authors with bids: {len(author2bids_norm)}')
    return author2bids_norm


def read_assignments(raw_path, train_uid2ann_cands, test_pids):
    """
    All the assignments seem formated similarly.
    - todo: For now ignoring the "weight" attributes in the assignment.
    """
    bids_missing_in_test = set()
    authors_missing_in_train = set()
    with codecs.open(os.path.join(raw_path, 'source_data', 'assignments', 'assignments.json'), 'r', 'utf-8') as fp:
        author2assigns = json.load(fp)
        author2assigns_norm = {}
        for aname in author2assigns:
            try:
                assert (aname in train_uid2ann_cands)
            except AssertionError:
                authors_missing_in_train.add(aname)
                continue
            norm_bids = {}
            for assign in author2assigns[aname]:
                try:
                    assert (assign['head'] in test_pids)
                except AssertionError:
                    # print(aname, bid['forum'])
                    bids_missing_in_test.add(assign['head'])
                    continue
                norm_bids[assign['head']] = 1
            author2assigns_norm[aname] = norm_bids
    print(f'Assigns missing in test: {len(bids_missing_in_test)}; '
          f'Authors missing in train: {len(authors_missing_in_train)}')
    print(f'Authors with assigns: {len(author2assigns_norm)}')
    return author2assigns_norm
    

def openreview_to_json(raw_path, json_path, dataset):
    """
    Write out json files which I use in my pipelines.
    - Consider the reviewers papers as their query papers for each year.
    - Create one vocabulary of keyphrases from the authors declared interests.
    - Create one file which treats the bids as the ground truth relevances.
        (bids in ICLR2018 is formatted differently than the rest)
    - Create another file which treats assignments as ground truth.
        (assignments in ICLR2018 don't have "weight" which is maybe score from TPMS or ELMO?)
    """
    raw_path = os.path.join(raw_path, dataset)
    json_path = os.path.join(json_path, f'or{dataset.lower()}')
    # Read the papers by all the conference authors and the user2likedpids.
    author_names = os.listdir(os.path.join(raw_path, 'source_data', 'archives'))
    print(f'{dataset}')
    pid2abstract = {}
    train_pids = set()
    train_uid2ann_cands = {}  # The users authored pids are the training pids - the cold start split is natural.
    doc_missing_tags = set()
    doc_missing_content = set()
    doc_missing_year = set()
    tags_per_abs = {}
    tokens_per_abs = {}
    sents_per_abs = {}
    tag2count = collections.Counter()
    train_uq_per_user = []
    for aname in author_names:
        user_query_pids = []
        with codecs.open(os.path.join(raw_path, 'source_data', 'archives', aname), 'r', 'utf-8') as fp:
            for line in fp:
                doc_dict = json.loads(line.strip())
                paper_id = doc_dict['id']
                try:
                    abstract_text = doc_dict['content']['abstract']
                except KeyError:
                    doc_missing_content.add(paper_id)
                    continue
                try:
                    title_text = doc_dict['content']['title']
                except KeyError:
                    doc_missing_content.add(paper_id)
                    continue
                try:
                    paper_bibtex = doc_dict['content']['_bibtex']
                    year_cands = re.findall(r'year=\{([0-9]+)},', paper_bibtex)
                    if len(year_cands) > 1 or len(year_cands) < 1:
                        doc_missing_year.add(paper_id)
                        paper_year = -1
                    else:
                        paper_year = int(year_cands[0])
                except KeyError:
                    doc_missing_year.add(paper_id)
                    paper_year = -1
                try:  # I think only papers reviewed on OR have keywords.
                    doc_tags = [re.sub('[\-\\s]', ' ', tag).lower() for tag in doc_dict['content']['keywords']]
                    tag2count.update(doc_tags)
                except KeyError:
                    doc_tags = []
                    doc_missing_tags.add(paper_id)
                abstract_sents = scispacy_model(abstract_text,
                                                disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                         'lemmatizer', 'parser', 'ner'])
                tags_per_abs[paper_id] = len(doc_tags)
                tokens_per_abs[paper_id] = len(abstract_text.split())
                abstract_sents = [sent.text for sent in abstract_sents.sents]
                sents_per_abs[paper_id] = len(abstract_sents)
                if len(abstract_sents) == 0:
                    doc_missing_content.add(paper_id)
                    continue
                d = {
                    'paper_id': paper_id,
                    'title': title_text,
                    'abstract': abstract_sents,
                    'year': paper_year,
                    'tags': doc_tags
                }
                pid2abstract[paper_id] = d
                train_pids.add(paper_id)
                user_query_pids.append(paper_id)
            assert(len(user_query_pids) == len(set(user_query_pids)))  # There shouldn't be repetitions.
            if len(user_query_pids) > 0:  # Keep users only if they have atleast one paper.
                train_uq_per_user.append(len(user_query_pids))
                train_uid2ann_cands[aname[:-6]] = {'uquery_pids': list(user_query_pids)}
    print('Train set query articles per user:\n {:}'.format(pd.DataFrame(train_uq_per_user).describe()))
    print(f'Valid users: {len(train_uid2ann_cands)}')
    
    # Read the submissions.
    cand_fnames = os.listdir(os.path.join(raw_path, 'source_data', 'submissions'))
    test_pids = set()
    for cname in cand_fnames:
        with codecs.open(os.path.join(raw_path, 'source_data', 'submissions', cname)) as fp:
            for line in fp:
                doc_dict = json.loads(line.strip())
                abstract_text = doc_dict['content']['abstract']
                title_text = doc_dict['content']['title']
                paper_id = doc_dict['id']
                if len(abstract_text.split()) < 0:
                    raise AssertionError
                try:
                    doc_tags = [re.sub('[\-\\s]', ' ', tag).lower() for tag in doc_dict['content']['keywords']]
                    tag2count.update(doc_tags)
                except KeyError:
                    doc_tags = []
                    doc_missing_tags.add(paper_id)
                abstract_sents = scispacy_model(abstract_text,
                                                disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                         'lemmatizer', 'parser', 'ner'])
                tags_per_abs[paper_id] = len(doc_tags)
                tokens_per_abs[paper_id] = len(abstract_text.split())
                abstract_sents = [sent.text for sent in abstract_sents.sents]
                sents_per_abs[paper_id] = len(abstract_sents)
                if len(abstract_sents) == 0:
                    doc_missing_content.add(paper_id)
                    continue
                d = {
                    'paper_id': paper_id,
                    'title': title_text,
                    'abstract': abstract_sents,
                    'tags': doc_tags
                }
                pid2abstract[paper_id] = d
                test_pids.add(paper_id)
    print('Per-abstract; Tokens : {:.2f}; Sentences: {:.2f}; Tags: {:.2f}'.
          format(statistics.mean(list(tokens_per_abs.values())),
                 statistics.mean(list(sents_per_abs.values())),
                 statistics.mean(list(tags_per_abs.values()))))
    print(f'Valid test items: {len(test_pids)}; Total items: {len(pid2abstract)}')
    print(f'Doc missing tags: {len(doc_missing_tags)}; missing title/abstract: {len(doc_missing_content)};'
          f' missing year: {len(doc_missing_year)}')
    # Read the expertise profiles of authors - for now just collect it as doc tags, don't associate it with users.
    with codecs.open(os.path.join(raw_path, 'source_data', 'profiles_expertise',
                                  'profiles_expertise.json'), 'r', 'utf-8') as fp:
        author2expertise = json.load(fp)
        for author, temporal_exp in author2expertise.items():
            if temporal_exp:
                for exp in temporal_exp:
                    tags = [re.sub('[\-\\s]', ' ', tag).lower() for tag in exp['keywords']]
                    tag2count.update(tags)
    # Read the bids and assignments.
    if dataset in {'ICLR2018'}:
        author2bids_norm = read_bids_2019pre(raw_path=raw_path, train_uid2ann_cands=train_uid2ann_cands,
                                             test_pids=test_pids)
    else:
        author2bids_norm = read_bids_2019onward(raw_path=raw_path, train_uid2ann_cands=train_uid2ann_cands,
                                                test_pids=test_pids)
    author2assigns_norm = read_assignments(raw_path=raw_path, train_uid2ann_cands=train_uid2ann_cands,
                                           test_pids=test_pids)
    for rel_type, author2rels_norm in [('bids', author2bids_norm), ('assigns', author2assigns_norm)]:
        test_uid2ann_cands = {}
        test_pids = list(test_pids)
        positive_bids = 0
        paper_authorships = 0
        for aname in author2rels_norm:
            relevance = []
            for pid in test_pids:  # Unbid documents are neutral.
                if pid in author2rels_norm[aname]:
                    relevance.append(author2rels_norm[aname][pid])
                    if author2rels_norm[aname][pid] > 0:
                        positive_bids += 1
                else:
                    relevance.append(0)
            user_query_pids = train_uid2ann_cands[aname]['uquery_pids']
            paper_authorships += len(user_query_pids)
            test_uid2ann_cands[aname] = {
                'cands': test_pids,
                'relevance_adju': relevance,
                'uquery_pids': list(user_query_pids)
            }
        print(f'Positive {rel_type} interactions: {positive_bids}; Authorship interactions: {paper_authorships}; '
              f'Total: {positive_bids+paper_authorships}')
        int_sparsity = 1.0-(positive_bids+paper_authorships)/(len(pid2abstract)*len(train_uid2ann_cands))
        print('{:} sparsity: {:.4f}'.format(rel_type, int_sparsity))
        with codecs.open(os.path.join(json_path, f'test-uid2anns-or{dataset.lower()}-{rel_type}.json'), 'w', 'utf-8') as fp:
            json.dump(test_uid2ann_cands, fp)
            print(f'Wrote: {fp.name}')

    # Create a dev set in the same form as the train set; but with fixed sampled negative per positive.
    dev_uid2ann_cands = {}
    dev_frac = 0.2 if dataset in {'UAI2019'} else 0.1  # Use more dev users if its a small dataset.
    dev_uids = random.sample(list(train_uid2ann_cands.keys()), int(dev_frac * len(train_uid2ann_cands.keys())))
    print(f'Development UIDs: {len(dev_uids)}')
    for duid in dev_uids:
        ann_d = train_uid2ann_cands.pop(duid)
        cand_neg_pids = set.difference(set(train_pids), set(ann_d['uquery_pids']))
        neg_pids = random.sample(list(cand_neg_pids), len(ann_d['uquery_pids']))
        dev_uid2ann_cands[duid] = {
            'uquery_pids': ann_d['uquery_pids'],
            'neg_pids': neg_pids
        }

    with codecs.open(os.path.join(json_path, f'train-uid2anns-or{dataset.lower()}.json'), 'w', 'utf-8') as fp:
        json.dump(train_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'dev-uid2anns-or{dataset.lower()}.json'), 'w', 'utf-8') as fp:
        json.dump(dev_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'abstracts-or{dataset.lower()}.json'), 'w', 'utf-8') as fp:
        json.dump(pid2abstract, fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'tags-or{dataset.lower()}.json'), 'w', 'utf-8') as fp:
        json.dump(dict(tag2count), fp)
        print(f'Wrote: {fp.name}')
    with codecs.open(os.path.join(json_path, f'tags-or{dataset.lower()}.txt'), 'w', 'utf-8') as fp:
        for tag in sorted(tag2count, key=tag2count.get, reverse=True):
            fp.write(f'{tag}, {tag2count[tag]}\n')
    print('\n')


def split_dev_test_users(json_path, dataset, ann_suffix):
    """
    Read the test set and the big-test set and chunk some 10% of the users into a separate dev set.
    """
    assert (ann_suffix in {'bids', 'assigns'})
    random.seed(592)
    with codecs.open(os.path.join(json_path, f'test-uid2anns-{dataset}-{ann_suffix}.json'), 'r', 'utf-8') as fp:
        test_uid2ann_cands = json.load(fp)
        print(f'Read: {fp.name}')
        print(f'Len: {len(test_uid2ann_cands)}')
    
    all_user_ids = list(test_uid2ann_cands.keys())  # + list(test_uid2ann_cands_big.keys())
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


def get_abstract_kps_consent(trained_kpenc_path, json_path, dataset):
    """
    Read in abstracts and retrieve keyphrases for the abstracts sentences.
    """
    # Read in keyphrases.
    with codecs.open(os.path.join(json_path, f'tags-{dataset}.json'), 'r', 'utf-8') as fp:
        kp2count = json.load(fp)
        keyphrases = []
        for kp in kp2count:
            if len(kp.split()) <= 5:
                keyphrases.append(kp)
    print(f'Filtered KPs: {len(keyphrases)}')
    
    # Read in abstracts.
    with codecs.open(os.path.join(json_path, f'abstracts-{dataset}.json'), 'r', 'utf-8') as fp:
        pid2abstract = json.load(fp)
    abstract_stream = list(pid2abstract.items())
    print(f'Abstracts: {len(abstract_stream)}')
    
    # Get the abstract sentence embeddings.
    trained_sentenc_path = os.path.join(os.environ['CUR_PROJ_DIR'], 'model_runs',
                                        "s2orccompsci/miswordbienc/miswordbienc-2021_10_07-09_06_18-ot-best-rr1")
    pid2sentembeds = pre_proc_buildreps.get_wholeabs_sent_reps(doc_stream=abstract_stream,
                                                               model_name='miswordbienc',
                                                               trained_model_path=trained_sentenc_path)
    print(f'Encoded abstracts: {len(pid2sentembeds)}')
    
    # Initialize the kpencoder model and compute keyphrase representations.
    word_embedding_model = models.Transformer('allenai/scibert_scivocab_uncased', max_seq_length=512)
    trained_model_fname = os.path.join(trained_kpenc_path, 'kp_encoder_cur_best.pt')
    word_embedding_model.auto_model.load_state_dict(torch.load(trained_model_fname))
    pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    kp_enc_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
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
    outfile = codecs.open(os.path.join(json_path, f'abstracts-{dataset}-gold-consent.jsonl'), 'w', 'utf-8')
    outfile_readable = codecs.open(os.path.join(json_path, f'abstracts-{dataset}-gold-consent.txt'), 'w', 'utf-8')
    all_tags_per_paper = []
    truncated_tags_per_paper = []
    start_idx = 0
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
            # for skp in sent_kps:
            #     if skp in uniq_abs_sent_kps:
            #         continue
            #     else:
            #         abs_sent_kps.append(skp)
            #         uniq_abs_sent_kps.add(skp)
            abs_sent_kps.append(sent_kps[0])
            abs_sent_kps_readable.append([(keyphrases[ki],'{:.4f}'.format(d)) for
                                          ki, d in zip(sent_kpidxs, sent_kpdists)])
        all_tags_per_paper.append(len(abs_sent_kps))
        # Get at most 50 of the keyphrases.
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


def get_abstract_kps_tfidf_consent_doc(trained_kpenc_path, json_path, dataset):
    """
    Read in abstracts
    - retrieve keyphrases for the abstracts sentences using simple term matches.
    - then re-rank these with the contextual sentence encoder and use those.
    """
    with codecs.open(os.path.join(json_path, f'tags-{dataset}.json'), 'r', 'utf-8') as fp:
        kp2count = json.load(fp)
        keyphrases = []
        for kp in kp2count:
            if len(kp.split()) <= 5:
                keyphrases.append(kp)
    print(f'Filtered KPs: {len(keyphrases)}')
    
    # Read in abstracts.
    with codecs.open(os.path.join(json_path, f'abstracts-{dataset}.json'), 'r', 'utf-8') as fp:
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
        pid2sentembeds_sparse[pid] = flat_sent_reps_sparse[start_idx: start_idx + num_dense_sentence_reps, :]
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
    num_batches = all_num_sents // bsize
    start_i = 0
    for bi in range(num_batches):
        b_nearest_dists_sp, b_nearest_idxs = index.kneighbors(X=flat_sent_reps_sparse[start_i:start_i + bsize, :])
        b_nearest_kp_dense = kp_embeddings_dense[b_nearest_idxs, :]
        b_sents_dense = flat_sent_reps_dense[start_i:start_i + bsize, :]
        b_nearest_dists_dense = np.linalg.norm(np.repeat(b_sents_dense, repeats=n_neighbors, axis=0) -
                                               np.reshape(b_nearest_kp_dense, (bsize * n_neighbors, dense_enc_dim)),
                                               axis=1)
        assert (b_nearest_dists_dense.shape[0] == bsize * n_neighbors)
        all_nearest_dists_dense[start_i:start_i + bsize, :] = np.reshape(b_nearest_dists_dense, (bsize, n_neighbors))
        all_nearest_sparse_dists[start_i:start_i + bsize, :] = b_nearest_dists_sp
        all_nearest_sparse_idxs[start_i:start_i + bsize, :] = b_nearest_idxs
        if bi % 10 == 0:
            print(f'KNN batch: {bi}/{num_batches}')
        start_i += bsize
    if start_i < all_num_sents:
        b_nearest_dists_sp, b_nearest_idxs = index.kneighbors(X=flat_sent_reps_sparse[start_i:, :])
        b_nearest_kp_dense = kp_embeddings_dense[b_nearest_idxs, :]
        b_sents_dense = flat_sent_reps_dense[start_i:, :]
        end_bsize = b_sents_dense.shape[0]
        b_nearest_dists_dense = np.linalg.norm(np.repeat(b_sents_dense, repeats=n_neighbors, axis=0) -
                                               np.reshape(b_nearest_kp_dense, (end_bsize * n_neighbors, dense_enc_dim)),
                                               axis=1)
        assert (b_nearest_dists_dense.shape[0] == end_bsize * n_neighbors)
        all_nearest_dists_dense[start_i:, :] = np.reshape(b_nearest_dists_dense, (end_bsize, n_neighbors))
        all_nearest_sparse_dists[start_i:, :] = b_nearest_dists_sp
        all_nearest_sparse_idxs[start_i:, :] = b_nearest_idxs
    all_nearest_sparse_idxs = all_nearest_sparse_idxs.astype('int32')
    
    outfile = codecs.open(os.path.join(json_path, f'abstracts-{dataset}-gold-tfidfcsrr.jsonl'), 'w', 'utf-8')
    outfile_readable = codecs.open(os.path.join(json_path, f'abstracts-{dataset}-gold-tfidfcsrr.txt'), 'w',
                                   'utf-8')
    all_tags_per_paper = []
    start_idx = 0
    for pid, sentembeds_sparse in pid2sentembeds_sparse.items():
        paper_dict = pid2abstract[pid]
        num_sents = sentembeds_sparse.shape[0]
        nearest_idxs = all_nearest_sparse_idxs[start_idx:start_idx + num_sents, :]
        nearest_dists_dense = all_nearest_dists_dense[start_idx:start_idx + num_sents, :]
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
            for kpi, dense_dist in sorted(zip(sent_kpidxs, sent_kpdists_dense), key=lambda tu: tu[1]):
                # if (non_oov_keyphrases[kpi] in sent) and (len(non_oov_keyphrases[kpi]) > 4):
                rr_lexical_match_kps.append(non_oov_keyphrases[kpi])
            raw_match_kps = []
            for kpi in sent_kpidxs:
                raw_match_kps.append(non_oov_keyphrases[kpi])
            valid_abs_sent_kps.append(rr_lexical_match_kps)
            raw_abs_sent_kps.append(raw_match_kps)
        # If there are sentences missing matches then randomly use one of the other sentences.
        assert (sentembeds_sparse.shape[0] == len(valid_abs_sent_kps) == len(raw_abs_sent_kps))
        abs_sent_kps = [skp[0] for skp in valid_abs_sent_kps]
        assert (len(abs_sent_kps) == sentembeds_sparse.shape[0])
        all_tags_per_paper.append(len(abs_sent_kps))
        if len(abs_sent_kps) > 20:  # There are some papers with full-text. Truncate those right away.
            paper_dict['abstract'] = paper_dict['abstract'][:20]
            paper_dict['forecite_tags'] = abs_sent_kps[:20]
        else:
            paper_dict['forecite_tags'] = abs_sent_kps
        outfile.write(json.dumps(paper_dict) + '\n')
        if len(all_tags_per_paper) < 100000:
            outfile_readable.write(f'{pid}\n')
            outfile_readable.write('\n'.join([paper_dict['title']] + paper_dict['abstract']) + '\n')
            outfile_readable.write(f'{valid_abs_sent_kps}' + '\n')
            outfile_readable.write(f'{raw_abs_sent_kps}' + '\n')
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


def create_itemcoldstart_pairdoc_examples(json_path, proc_path, dataset, model_name, subsample_user=False):
    """
    Given the itemcoldstart splits write out train and dev files for the document
    level similarity model.
    """
    random.seed(720)
    # Make the output examples directory.
    out_path = os.path.join(proc_path, dataset, model_name, 'cold_start')
    data_utils.create_dir(out_path)
    
    with codecs.open(os.path.join(json_path, f'train-uid2anns-{dataset}.json'), 'r', 'utf-8') as fp:
        train_uid2anns = json.load(fp)
        print(f'Read: {fp.name}')
    with codecs.open(os.path.join(json_path, f'dev-uid2anns-{dataset}.json'), 'r', 'utf-8') as fp:
        dev_uid2anns = json.load(fp)
        print(f'Read: {fp.name}')
    
    with codecs.open(os.path.join(json_path, f'abstracts-{dataset}-gold.jsonl'), 'r', 'utf-8') as fp:
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
                cidxs = random.sample(cidxs, min(len(cidxs), 300))  # in OR datasets 75% of users are between 200-600.
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
        assert (ann_suffix in {'ccf'})
        train_ann_json = os.path.join(json_path, f'train-uid2anns-{dataset}-{ann_suffix}.json')
        dev_ann_json = os.path.join(json_path, f'dev-uid2anns-{dataset}-{ann_suffix}.json')
    else:  # This is cold-start for the proposed models - the default.
        train_ann_json = os.path.join(json_path, f'train-uid2anns-{dataset}.json')
        dev_ann_json = os.path.join(json_path, f'dev-uid2anns-{dataset}.json')
    
    if abstract_suffix:
        assert (abstract_suffix in {'tfidfcsrr', 'tfidf'})
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
    # Make cold start json splits.
    openreview_to_json(
        raw_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/openreview-raw/'),
        json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/'), dataset='ICLR2019')

    openreview_to_json(
        raw_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/openreview-raw'),
        json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw'), dataset='ICLR2020')

    openreview_to_json(
        raw_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/openreview-raw/'),
        json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/'), dataset='UAI2019')
    
    # Make splits for cold start eval.
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/oruai2019'),
                         dataset='oruai2019', ann_suffix='bids')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/oruai2019'),
                         dataset='oruai2019', ann_suffix='assigns')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/oriclr2019'),
                         dataset='oriclr2019', ann_suffix='bids')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/oriclr2019'),
                         dataset='oriclr2019', ann_suffix='assigns')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/oriclr2020'),
                         dataset='oriclr2020', ann_suffix='bids')
    split_dev_test_users(json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/oriclr2020'),
                         dataset='oriclr2020', ann_suffix='assigns')
    
    # Prefetch concepts for abstracts.
    get_abstract_kps_tfidf_consent_doc(
        trained_kpenc_path=os.path.join(os.environ['CUR_PROJ_DIR'],
                                        '/model_runs/gorccompscicit/kpencconsent/kpencconsent-2022_01_19-21_54_43-scib'),
        json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/oruai2019'),
        dataset='oruai2019')
    get_abstract_kps_tfidf_consent_doc(
        trained_kpenc_path=os.path.join(os.environ['CUR_PROJ_DIR'],
                                        '/model_runs/gorccompscicit/kpencconsent/kpencconsent-2022_01_19-21_54_43-scib'),
        json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/oriclr2019'),
        dataset='oriclr2019')
    get_abstract_kps_tfidf_consent_doc(
        trained_kpenc_path=os.path.join(os.environ['CUR_PROJ_DIR'],
                                        '/model_runs/gorccompscicit/kpencconsent/kpencconsent-2022_01_19-21_54_43-scib'),
        json_path=os.path.join(os.environ['CUR_PROJ_DIR'], '/datasets_raw/oriclr2020'),
        dataset='oriclr2020')
