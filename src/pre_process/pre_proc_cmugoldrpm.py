"""
Pre process the data from: https://github.com/niharshah/goldstandard-reviewer-paper-match for
predictions with LACE.
"""
import ast
import csv
import json, codecs
import collections
import os
import re
import statistics

import numpy as np
import pandas as pd
from sklearn import feature_extraction as sk_featext
import spacy
import torch
from sentence_transformers import SentenceTransformer, models
from sklearn import neighbors
from . import pre_proc_buildreps

scispacy_model = spacy.load("en_core_sci_sm")
scispacy_model.add_pipe('sentencizer')


def cmugoldrpm_to_json(raw_path, json_path):
    """
    - Read the evaluations.csv file and get the candidates for every user which we'll be scoring.
    - Read the participants folder to get the S2 profile of papers for the users.
    - Read the papers folder to get the text for the papers (candidates and historical papers?)
    """
    os.makedirs(json_path, exist_ok=True)
    # Read in candidates which need rating.
    rating_df = pd.read_csv(os.path.join(raw_path, 'evaluations.csv'), sep='\t')
    uid2cand_ratings = {}
    for idx, row in rating_df.iterrows():
        uid = str(row['ParticipantID'])
        uid2cand_ratings[uid] = {row[f'Paper{x}']: row[f'Expertise{x}'] for x in range(1, 11)
                                 if not pd.isna(row[f'Paper{x}'])}
    print(f'Users with ratings: {len(uid2cand_ratings)}')
    
    # Read in papers.
    paper_fnames = os.listdir(os.path.join(raw_path, 'papers'))
    pid2abstract = {}
    tokens_per_abs = {}
    sents_per_abs = {}
    doc_missing_content = set()
    for cname in paper_fnames:
        with codecs.open(os.path.join(raw_path, 'papers', cname), 'r', 'utf-8') as fp:
            doc_dict = json.load(fp)
        abstract_text = doc_dict['abstract']
        title_text = doc_dict['title']
        assert(isinstance(title_text, str))
        paper_id = doc_dict['ssId']
        if abstract_text == None or len(abstract_text.split()) <= 0:
            # print(cname)
            abstract_sents = ['No abstract']
        else:
            abstract_sents = scispacy_model(abstract_text,
                                            disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                     'lemmatizer', 'parser', 'ner'])
            tokens_per_abs[paper_id] = len(abstract_text.split())
            abstract_sents = [sent.text for sent in abstract_sents.sents]
        sents_per_abs[paper_id] = len(abstract_sents)
        if len(abstract_sents) == 0:
            doc_missing_content.add(paper_id)
        d = {
            'paper_id': paper_id,
            'title': title_text,
            'abstract': abstract_sents
        }
        pid2abstract[paper_id] = d
    pid2abstract['no_ss'] = {'paper_id': 'no_ss', 'title': 'No title', 'abstract': ['No abstract']}
    print('Per-abstract; Total docs: {:d}; Mean; tokens: {:.2f}; sentences: {:.2f}'.
          format(len(pid2abstract), statistics.mean(list(tokens_per_abs.values())),
                 statistics.mean(list(sents_per_abs.values()))))
    print(f'Docs missing abstract text: {len(doc_missing_content)}')
    with codecs.open(os.path.join(json_path, f'abstracts-cmugoldrpm.json'), 'w', 'utf-8') as fp:
        json.dump(pid2abstract, fp)
        print(f'Wrote: {fp.name}')
        
    # Read in the historic profiles for users.
    test_uid2ann_cands = {}
    for uid in uid2cand_ratings:
        with codecs.open(os.path.join(raw_path, 'participants', f'{uid}.json'), 'r', 'utf-8') as fp:
            author_profile = json.load(fp)
        user_query_pids = [d['paperId'] for d in author_profile['papers']]
        
        test_pids = []
        relevance = []
        for tpid in uid2cand_ratings[uid]:
            try:  # This happens once with the tpid called 'no_ss'; dk how to handle this.
                assert(tpid in pid2abstract)
            except AssertionError:
                print(tpid)
                # continue
            test_pids.append(tpid)
            relevance.append(uid2cand_ratings[uid][tpid])
        test_uid2ann_cands[uid] = {
            'cands': test_pids,
            'relevance_adju': relevance,
            'uquery_pids': list(user_query_pids)
        }
    with codecs.open(os.path.join(json_path, f'test-uid2anns-cmugoldrpm.json'), 'w', 'utf-8') as fp:
        json.dump(test_uid2ann_cands, fp)
        print(f'Wrote: {fp.name}')


def cmugoldrpm_to_json_repeats(raw_path, json_path):
    """
    - Read the repeatedly truncated archives of participants and make test files for them all.
    """
    os.makedirs(json_path, exist_ok=True)
    # Read in candidates which need rating.
    rating_df = pd.read_csv(os.path.join(raw_path, 'data',  'evaluations.csv'), sep='\t')
    uid2cand_ratings = {}
    for idx, row in rating_df.iterrows():
        uid = str(row['ParticipantID'])
        uid2cand_ratings[uid] = {row[f'Paper{x}']: row[f'Expertise{x}'] for x in range(1, 11)
                                 if not pd.isna(row[f'Paper{x}'])}
    print(f'Users with ratings: {len(uid2cand_ratings)}')
    
    # Read in the historic profiles for users.
    for i in range(1, 11):
        test_uid2ann_cands = {}
        for uid in uid2cand_ratings:
            user_query_pids = []
            with codecs.open(os.path.join(raw_path, 'evaluation_datasets', f'd_20_{i}', 'archives',
                                          f'~{uid}.jsonl'), 'r', 'utf-8') as fp:
                for line in fp:
                    paperd = json.loads(line.strip())
                    user_query_pids.append(paperd['id'])
            
            test_pids = []
            relevance = []
            for tpid in uid2cand_ratings[uid]:
                test_pids.append(tpid)
                relevance.append(uid2cand_ratings[uid][tpid])
            test_uid2ann_cands[uid] = {
                'cands': test_pids,
                'relevance_adju': relevance,
                'uquery_pids': list(user_query_pids)
            }
        with codecs.open(os.path.join(json_path, f'test-uid2anns-cmugoldrpm20s{i}.json'), 'w', 'utf-8') as fp:
            json.dump(test_uid2ann_cands, fp)
            print(f'Wrote: {fp.name}')


def get_abstract_kps_tfidf_consent_doc(json_path, concepts_path, dataset, sent_enc_name, kp_enc_name):
    """
    Read in abstracts
    - retrieve keyphrases for the abstracts sentences using simple term matches.
    - then re-rank these with the contextual sentence encoder and use those.
    """
    kp_file = codecs.open(os.path.join(concepts_path, 'gorccompscicit-keyphrases-forecite-filt-cul.csv'), 'r', 'utf-8')
    kp_csv = csv.DictReader(kp_file)
    kp2kpd = {}
    for kpd in kp_csv:
        kpd['mention_pids'] = set(ast.literal_eval(kpd['mention_pids']))
        kp2kpd[kpd['keyphrase']] = kpd
    print(f'Filtered KPs: {len(kp2kpd)}')
    keyphrases = list(kp2kpd.keys())
    outfile = codecs.open(os.path.join(json_path, f'abstracts-{dataset}-forecite-tfidfscrr.jsonl'), 'w', 'utf-8')
    outfile_readable = codecs.open(os.path.join(json_path, f'abstracts-{dataset}-forecite-tfidfscrr.txt'), 'w',
                                   'utf-8')
    
    # Read in abstracts.
    with codecs.open(os.path.join(json_path, f'abstracts-{dataset}.json'), 'r', 'utf-8') as fp:
        pid2abstract = json.load(fp)
    abstract_stream = list(pid2abstract.items())
    print(f'Abstracts: {len(abstract_stream)}')
    
    # Get the dense abstract sentence embeddings.
    pid2sentembeds_dense = pre_proc_buildreps.get_wholeabs_sent_reps(
        doc_stream=abstract_stream, model_name=sent_enc_name,
        trained_model_path=None)
        
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
    word_embedding_model = models.Transformer(kp_enc_name, max_seq_length=512)
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
    

def write_predictions_for_eval(run_path, full_profile=True):
    """
    Read the scores produced from lace using the full profile
    and write out the scores to files called lacefullprofile_d_20_{1-10}_ta.json
    - This is different from the paper which used 20 of the most recent papers
        for each reviewer.
    """
    if full_profile:
        out_path = os.path.join(run_path, 'lacefullprofile_predictions')
        ranked_file = os.path.join(run_path, f'test-pid2pool-cmugoldrpm-upnfconsent-ranked.json')
        outfile = os.path.join(out_path, 'lacefullprofile_d_20_{i}_ta.json')
    else:
        out_path = os.path.join(run_path, 'lace20profile_predictions')
        ranked_file = os.path.join(run_path, 'test-pid2pool-cmugoldrpm-upnfconsent20s{i}-ranked.json')
        outfile = os.path.join(out_path, 'lace20profile_d_20_{i}_ta.json')
    os.makedirs(out_path, exist_ok=True)
    # Write the 10 output files (which are identical here but ideally should use the 20 large profile)
    for i in range(1, 11):
        with codecs.open(re.sub('\{i\}', f'{i}', ranked_file), 'r','utf-8') as fp:
            uid2ranked_cands = json.load(fp)
            print(f'Read: {fp.name}')
            
        out_uid2preds = {}
        for uid, ranked_cands in uid2ranked_cands.items():
            pid2score = {}
            for pid, score in ranked_cands:
                pid2score[pid] = score
            out_uid2preds[uid] = pid2score
        with codecs.open(re.sub('\{i\}', f'{i}', outfile), 'w', 'utf-8') as fp:
            json.dump(out_uid2preds, fp)
            print(f'Wrote: {fp.name}')
        
        
if __name__ == '__main__':
    # Convert the dataset into jsons consumed by LACE.
    cmugoldrpm_to_json(raw_path='./data/datasets_raw/goldstandard-reviewer-paper-match/data',
                       json_path='./data/datasets_raw/cmugoldrpm')
    
    # Read the 20 item profiles.
    cmugoldrpm_to_json_repeats(
        raw_path=os.path.join('./data/datasets_raw/goldstandard-reviewer-paper-match'),
        json_path=os.path.join('./data/datasets_raw/cmugoldrpm'))
    
    # Pre-fetch keyphrases for the abstracts.
    get_abstract_kps_tfidf_consent_doc(
        sent_enc_name='allenai/aspire-contextualsentence-multim-compsci',
        kp_enc_name='Sheshera/lace-kp-encoder-compsci',
        json_path=os.path.join('./data/datasets_raw/cmugoldrpm'),
        concepts_path=os.path.join('./data'),
        dataset='cmugoldrpm')
    
    # # Write out predictions like the cmugoldrpm eval script likes it to be.
    write_predictions_for_eval(run_path=os.path.join(os.environ['CUR_PROJ_DIR'],
                                                     'datasets_raw/cmugoldrpm/upnfconsent/manual_run'),
                               full_profile=True)
    write_predictions_for_eval(run_path=os.path.join(os.environ['CUR_PROJ_DIR'],
                                                     'datasets_raw/cmugoldrpm/upnfconsent/manual_run'),
                               full_profile=False)
