"""
Prepare candidate papers to be used for ranking.
"""
import codecs
import json
import os
import pickle
import joblib
import random
import sys
import ast
import csv
import numpy as np
import pandas as pd

import torch
from sentence_transformers import SentenceTransformer, models

from . import pre_proc_build_reps


def sample_abstracts(metadata_csv_path, abstracts_path, out_path):
    """
    - Read the papers for which we have abstracts
    - Read the citation network metadata for the s2orccompsci set
    - Sample 50000 of the highest cited papers and 50000 random papers.
    :return:
    """
    # Read the metadata.
    meta_csv = pd.read_csv(os.path.join(metadata_csv_path, 'metadata-gorccompsci.tsv'),
                           delimiter='\t', error_bad_lines=False, engine='python', quoting=csv.QUOTE_NONE)
    pid2in_cite_count = {}
    for rid, row in meta_csv.iterrows():
        pid2in_cite_count[str(row['pid'])] = len(ast.literal_eval(row['inbound_citations']))
    
    # Read the papers into memory.
    with codecs.open(os.path.join(abstracts_path, 'abstracts-s2orccompsci.pickle'), 'rb') as fp:
        pid2abstract = pickle.load(fp)
    
    # Get the counts for the papers we have abstracts for.
    abs_pid2ccount = {}
    missing_pids = []
    for pid in pid2abstract:
        try:
            abs_pid2ccount[pid] = pid2in_cite_count[pid]
        except KeyError:
            missing_pids.append(pid)
    
    # Sort the papers by cite count and select 50k by cite count and 50k randomly.
    ccount_sorted_pids = list(sorted(abs_pid2ccount, key=abs_pid2ccount.get, reverse=True))
    selected_pids = ccount_sorted_pids[:50000]
    other_pids = set.difference(set(pid2abstract.keys()), set(selected_pids))
    random_selections = random.sample(other_pids, k=50000)
    selected_pids.extend(random_selections)
    
    # Write out the selected pids.
    outjsonl = codecs.open(os.path.join(out_path, 'abstracts-s2orccompsci-100k.jsonl'), 'w', 'utf-8')
    outf = open(os.path.join(out_path, 'abstracts-s2orccompsci-100k.pickle'), 'wb')
    subset_pid2abstract = {}
    count = 0
    for pid in selected_pids:
        subset_pid2abstract[pid] = pid2abstract[pid]
        outjsonl.write(json.dumps(pid2abstract[pid])+'\n')
        count += 1
    pickle.dump(subset_pid2abstract, outf)


def write_cand_doc_embeddings(embeddings_path, candidates_path):
    """
    - Read the pickle file with candidate documents.
    - Read the matrix with co-specter embeddings pre-written to disk
        and read the candidate paper embeddings.
    - Write candidate embeddings to disk.
    :return:
    """
    with open(os.path.join(candidates_path, 'abstracts-s2orccompsci-100k.pickle'), 'rb') as fp:
        pid2abstract = pickle.load(fp)
    
    all_doc_embeddings = np.load(os.path.join(embeddings_path, 's2orccompsci-doc.npy'))
    # For s2orccompsci the number of documents lengths were computed incorrectly.
    abstract_embeddings = all_doc_embeddings[:1479196, :]
    with open(os.path.join(embeddings_path, 'pid2idx-s2orccompsci-doc.json')) as fp:
        pid2idx = json.load(fp)
        
    candidate_embeddings = np.empty((100000, 768))
    candidate_pid2idx = {}
    for i, pid in enumerate(pid2abstract):
        idx = pid2idx[pid]
        candidate_embeddings[i] = abstract_embeddings[idx]
        candidate_pid2idx[pid] = i
    print(candidate_embeddings.shape)
    print(len(candidate_pid2idx))
    np.save(os.path.join(candidates_path, 'embeds-s2orccompsci-100k.npy'), candidate_embeddings)
    with open(os.path.join(candidates_path, 'pid2idx-s2orccompsci-100k.pickle'), 'wb') as fp:
        pickle.dump(candidate_pid2idx, fp)


def embed_candidates(candidates_path):
    """
    - Read in the candidate documents and embed them to get sentence reps.
    - These are used for ranking documents with LACE in the app.
    - The model used here is called: aspire-contextualsentence-multim-compsci on HF.
    :return:
    """
    # Read in abstracts.
    with open(os.path.join(candidates_path, f'abstracts-s2orccompsci-100k.pickle'), 'rb') as fp:
        pid2abstract = pickle.load(fp)
    abstract_stream = list(pid2abstract.items())
    print(f'Abstracts: {len(abstract_stream)}')

    # Get the abstract sentence embeddings.
    trained_sentenc_path = os.path.join(os.environ['CUR_PROJ_DIR'], 'model_runs',
                                        "s2orccompsci/miswordbienc/miswordbienc-2021_10_07-09_06_18-ot-best-rr1")
    pid2sentembeds = pre_proc_build_reps.get_wholeabs_sent_reps(doc_stream=abstract_stream,
                                                                model_name='miswordbienc',
                                                                trained_model_path=trained_sentenc_path)
    print(f'Encoded abstracts: {len(pid2sentembeds)}')
    
    # Save the embeddings to disk.
    outfname = os.path.join(candidates_path, 'embeds-sent-s2orccompsci-100k.pickle')
    joblib.dump(pid2sentembeds, outfname)
    print(f'Wrote: {outfname}')


def embed_keyphrases(forecite_path, trained_kpenc_path):
    """
    - Read keyphrases which have been interactively filtered further
        with cluster based exclusion.
    - Embed them and save them to disk; these are used for infering the keyphrase
        profile for every user in the study.
    :return:
    """
    # Read in keyphrases.
    kp_file = codecs.open(os.path.join(forecite_path,
                                       'gorccompscicit-keyphrases-forecite-laceustudy.csv'), 'r', 'utf-8')
    kp_csv = csv.DictReader(kp_file)
    kp2kpd = {}
    for kpd in kp_csv:
        kpd['mention_pids'] = set(ast.literal_eval(kpd['mention_pids']))
        kp2kpd[kpd['keyphrase']] = kpd
    print(f'Filtered KPs: {len(kp2kpd)}')
    keyphrases = list(kp2kpd.keys())
    
    # Load the model and embed the data.
    kp_enc_model = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')
    kp_embeddings = kp_enc_model.encode(keyphrases)
    
    # Save the keyphrases to disk.
    out_arrs = os.path.join(forecite_path, 'keyphrases-mpnet1b.npy')
    np.save(out_arrs, kp_embeddings)
    print(f'Wrote: {out_arrs}')
    with codecs.open(os.path.join(forecite_path, 'keyphrases-mpnet1b.txt'), 'w', 'utf-8') as fp:
        for kp in keyphrases:
            fp.write(kp+'\n')
        print(f'Wrote: {fp.name}')


if __name__ == '__main__':
    # Sample 100k abstracts for using as candidates.
    sample_abstracts(metadata_csv_path='datasets_raw/s2orc/hostservice_filt/metadata-gorccompsci.tsv',
                     abstracts_path='datasets_raw/s2orccompsci',
                     out_path='datasets_raw/s2orccompsci/user_study')
    
    # Embed the sentence embeddings and save them to disk.
    embed_candidates(candidates_path='/datasets_raw/s2orccompsci/user_study')
    
    # Write embeddings to pickle for easy reading.
    write_cand_doc_embeddings(
        embeddings_path='datasets_raw/s2orccompsci/cospecter/cospecter-2021_08_05-00_43_28-specter_init/',
        candidates_path='datasets_raw/s2orccompsci/user_study')
    
    # Embed keyphrases and save them to disk.
    embed_keyphrases(forecite_path='/datasets_raw/s2orccompsci/user_study/cands',
                     trained_kpenc_path='/model_runs/gorccompscicit/kpencconsent/kpencconsent-2022_01_19-21_54_43-scib')
    
