"""
Given the users papers create a seed set of papers.
"""
import collections
import os
import time
import codecs, json
import joblib
import argparse
import numpy as np
import seaborn as sns
from sklearn import neighbors
from scipy.spatial import distance
import ot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams['figure.dpi'] = 400
plt.rcParams.update({'axes.labelsize': 'small'})

from .pre_proc_build_reps import TrainedModel, get_wholeabs_sent_reps


def write_wholeabs_reps(out_path, username, model_name, condition):
    """
    Given a set of papers: read the abstract sentences and write out the bert representations of
    the abstracts. The entire abstract is passed through bert as one string with [SEP] tokens
    marking off sentences.
    :param data_path: base directory with abstract jsonl docs.
    :param out_path: directory to which bert reps, and maps of bert reps to strings will be written.
    :param username: string;
    :param model_name: string;
    :return: None. Writes to disk.
    """
    sent_enc_dim = 768
    in_fname = os.path.join(out_path, f'seedset-{username}-{condition}.json')
    with codecs.open(in_fname, 'r', 'utf-8') as fp:
        user_data = json.load(fp)
        user_papers = user_data['papers']
    cls_out_fname = os.path.join(out_path, 'embeds-{:s}-doc.npy'.format(username))
    out_map_fname = os.path.join(out_path, 'pid2idx-{:s}-doc.json'.format(username))
    num_docs = 0
    doc_stream = []
    for pid, paper_dict in enumerate(user_papers):
        num_docs += 1
        sents = paper_dict['abstract']
        abs_text = ' '.join(sents)
        ret_text = paper_dict['title'] + '[SEP]' + abs_text
        doc_stream.append((pid, ret_text))
    
    if model_name in {'cospecter'}:
        trained_model_path = os.path.join(os.environ['CUR_PROJ_DIR'],
                                          '/model_runs/s2orccompsci/cospecter/cospecter-2021_08_05-00_43_28-specter_init')
        model = TrainedModel(model_name=model_name, trained_model_path=trained_model_path)
        batch_size = 32
    start = time.time()
    print('Processing files in: {:s}'.format(in_fname))
    print('Num docs: {:d}'.format(num_docs))
    
    # Write out sentence reps incrementally.
    reps2idx = {}
    doc_reps_cls = np.empty((num_docs, sent_enc_dim))
    print('Allocated space for reps: {:}'.format(doc_reps_cls.shape))
    batch_docs = []
    batch_start_idx = 0
    for doci, (pid, abs_text) in enumerate(doc_stream):
        if doci % 1000 == 0:
            print('Processing document: {:d}/{:d}'.format(doci, num_docs))
        batch_docs.append(abs_text)
        reps2idx[pid] = len(reps2idx)
        if len(batch_docs) == batch_size:
            batch_reps_av, batch_reps_cls = model.predict(batch_docs)
            batch_docs = []
            doc_reps_cls[batch_start_idx:batch_start_idx + batch_size, :] = batch_reps_cls
            batch_start_idx = batch_start_idx + batch_size
    # Handle left over sentences.
    if len(batch_docs) > 0:
        batch_reps_av, batch_reps_cls = model.predict(batch_docs)
        final_bsize = batch_reps_cls.shape[0]
        doc_reps_cls[batch_start_idx:batch_start_idx + final_bsize, :] = batch_reps_cls
    print('Doc reps shape: {:}; Map length: {:d}'.format(doc_reps_cls.shape, len(reps2idx)))
    with codecs.open(out_map_fname, 'w', 'utf-8') as fp:
        json.dump(reps2idx, fp)
        print('Wrote: {:s}'.format(fp.name))
    with codecs.open(cls_out_fname, 'wb') as fp:
        np.save(fp, doc_reps_cls)
        print('Wrote: {:s}'.format(fp.name))
    print('Took: {:.4f}s'.format(time.time() - start))


def write_seedabs_sent_reps(out_path, username, condition):
    """
    - Write contextual aspire sentence embeddings for the users seed papers.
    """
    # Read in abstracts.
    with codecs.open(os.path.join(out_path, f'seedset-{username}-{condition}.json'), 'r', 'utf-8') as fp:
        seed_set = json.load(fp)
    abstract_stream = [(i, paperd) for i, paperd in enumerate(seed_set['papers'])]
    print(f'Abstracts: {len(abstract_stream)}')
    
    # Get the abstract sentence embeddings.
    trained_sentenc_path = os.path.join(os.environ['CUR_PROJ_DIR'],
                                        'model_runs/s2orccompsci/miswordbienc/miswordbienc-2021_10_07-09_06_18-ot-best-rr1')
    pid2sentembeds = get_wholeabs_sent_reps(doc_stream=abstract_stream,
                                            model_name='miswordbienc',
                                            trained_model_path=trained_sentenc_path)
    print(f'Encoded abstracts: {len(pid2sentembeds)}')
    outfname = os.path.join(out_path, f'embeds-{username}-sent.pickle')
    joblib.dump(pid2sentembeds, outfname)
    print('Wrote: {:s}'.format(outfname))


def get_user_paper_forecitekps_consent(data_path, user_path, username, condition, model_name='mpnet1b'):
    """
    - Read in keyphrases and the embeddings for the keyphrases
    - Read in the seed set of user papers
    - Embed the seed set of papers
    - Retrieve keyphrases for the seed papers and aggregate them across user papers
    """
    cand_path = os.path.join(data_path, 'cands')
    # Read in keyphrases.
    kp_file = codecs.open(os.path.join(cand_path, f'keyphrases-{model_name}.txt'), 'r', 'utf-8')
    keyphrases = []
    for line in kp_file:
        keyphrases.append(line.strip())
    kp_embeddings = np.load(os.path.join(cand_path, f'keyphrases-{model_name}.npy'))
    print(f'Keyphrases: {len(keyphrases)}; Encoded keyphrases: {kp_embeddings.shape}')
    # Read in abstracts.
    with codecs.open(os.path.join(user_path, f'seedset-{username}-{condition}.json'), 'r', 'utf-8') as fp:
        seed_set = json.load(fp)
    abstract_stream = [(i, paperd) for i, paperd in enumerate(seed_set['papers'])]
    pid2abstract = dict(abstract_stream)
    print(f'Abstracts: {len(abstract_stream)}')
    
    # Get the abstract sentence embeddings.
    if model_name == 'mpnet1b':
        pid2sentembeds = get_wholeabs_sent_reps(doc_stream=abstract_stream,
                                                model_name='sentence-transformers/all-mpnet-base-v2',
                                                trained_model_path=None)
    else:
        raise ValueError(f'Unknown model: {model_name}')
    print(f'Encoded abstracts: {len(pid2sentembeds)}')
    
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
    outfile = codecs.open(os.path.join(user_path, f'abstracts-{username}-forecite.jsonl'), 'w', 'utf-8')
    outfile_readable = codecs.open(os.path.join(user_path, f'abstracts-{username}-forecite.txt'), 'w', 'utf-8')
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
        for si in range(sentembeds.shape[0]):
            sent_kpidxs = nearest_idxs[si, :].tolist()
            sent_kpdists = nearest_dists[si, :].tolist()
            sent_kps = [keyphrases[ki] for ki in sent_kpidxs]
            abs_sent_kps.append(sent_kps[0])
            abs_sent_kps_readable.append([(keyphrases[ki],
                                           '{:.4f}'.format(d)) for
                                          ki, d in zip(sent_kpidxs, sent_kpdists)])
        all_tags_per_paper.append(len(abs_sent_kps))
        # Get at most 50 of the keyphrases.
        paper_dict['forecite_tags'] = abs_sent_kps[:50]
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


def aggregate_user_kps(data_path, user_path, username, condition):
    """
    - Read the user documents and their kps obtained from
        get_user_paper_forecitekps_consent. Read 5 kps per abstract.
    - Read the kp embeddings written by pre_proc_candidates.embed_keyphrases
        and get embeddings for above kps.
    - Read the user sentence embeddings written in write_seedabs_sent_reps.
    - Compute pairwise distances bw the keyphrases and sentences, for every kp
        get the minimum distance sentence, sort the kps by this distance,
        pick the top-20 keyphrases as the profile keyphrases.
    :return:
    """
    per_paper_kps = 5
    per_user_kps = 20
    cand_path = os.path.join(data_path, 'cands')
    # Read seed paper kps
    seed_papers_kps = []
    uniq_seed_kps = set()
    abstract_stream = []
    kp_counter = collections.Counter()
    with codecs.open(os.path.join(user_path, f'abstracts-{username}-forecite.jsonl'), 'r', 'utf-8') as fp:
        for pid, line in enumerate(fp):
            paperd = json.loads(line.strip())
            abstract_stream.append((pid, paperd))
            for kp in paperd['forecite_tags'][:per_paper_kps]:
                kp_counter.update([kp])
                if kp not in uniq_seed_kps:
                    uniq_seed_kps.add(kp)
                    seed_papers_kps.append(kp)
    # Get seed paper kp embeddings
    kp_embeddings = np.load(os.path.join(cand_path, 'keyphrases-kpencconsent.npy'))
    kp2embedding = dict()
    with codecs.open(os.path.join(cand_path, 'keyphrases-kpencconsent.txt'), 'r', 'utf-8') as fp:
        for i, line in enumerate(fp):
            kp = line.strip()
            kp2embedding[kp] = kp_embeddings[i, :]
    seed_kp_embeds = []
    for kp in seed_papers_kps:
        seed_kp_embeds.append(kp2embedding[kp])
    seed_kp_embeds = np.vstack(seed_kp_embeds)
    # Get user seed paper sentence embeddings.
    pid2sentembeds = joblib.load(os.path.join(user_path, f'embeds-{username}-sent.pickle'))
    user_seed_sentembeds = []
    for pid in sorted(pid2sentembeds):  # Get the pids in sorted order.
        user_seed_sentembeds.append(pid2sentembeds[pid])
    user_seed_sentembeds = np.vstack(user_seed_sentembeds)
    
    # Get the keyphrases based on frequency of a paper begin tagged with it.
    selected_kps = []
    selected_kp_i = []
    for kp in sorted(kp_counter, key=kp_counter.get, reverse=True):
        selected_kps.append(kp)
        selected_kp_i.append(seed_papers_kps.index(kp))
        if len(selected_kp_i) == per_user_kps:
            break
    print(selected_kps)
    # Sort the kps by lexical order to make it easier to exclude them.
    select_sorted_kps = []
    select_sorted_kp_i = []
    for kp, kpi in sorted(zip(selected_kps, selected_kp_i), key=lambda tu: tu[0]):
        select_sorted_kps.append(kp)
        select_sorted_kp_i.append(kpi)
    # Write the kps to disk.
    with codecs.open(os.path.join(user_path, f'seedset-{username}-{condition}.json'), 'r', 'utf-8') as fp:
        seed_set = json.load(fp)
        seed_set['user_kps'] = select_sorted_kps
    with codecs.open(os.path.join(user_path, f'seedset-{username}-{condition}.json'), 'w', 'utf-8') as fp:
        json.dump(seed_set, fp, indent=2)
        print(f'Wrote: {fp.name}')
        
    # Compute the transport plan from sents to kps and save to disk for examination.
    selected_kps_embeds = seed_kp_embeds[select_sorted_kp_i, :]
    pair_dists = distance.cdist(user_seed_sentembeds, selected_kps_embeds, 'euclidean')
    a_distr = [1/user_seed_sentembeds.shape[0]] * user_seed_sentembeds.shape[0]
    b_distr = [1/selected_kps_embeds.shape[0]] * selected_kps_embeds.shape[0]
    tplan = ot.partial.entropic_partial_wasserstein(a_distr, b_distr, pair_dists, 0.05, m=0.8)
    h = sns.heatmap(tplan.T, linewidths=.3, xticklabels=range(user_seed_sentembeds.shape[0]),
                    yticklabels=select_sorted_kps, cmap='Blues')
    h.tick_params('y', labelsize=5)
    h.tick_params('x', labelsize=2)
    plt.tight_layout()
    outfig_name = os.path.join(user_path, f'tplan-{username}.png')
    plt.savefig(outfig_name)
    print(f'Wrote: {outfig_name}')
    sent_i = 0
    outf = codecs.open(os.path.join(user_path, f'sents-{username}.txt'), 'w', 'utf-8')
    for pid, paperd in abstract_stream:
        outf.write(f'Title: {paperd["title"]}\n')
        encoded_sents = pid2sentembeds[pid].shape[0]
        for sent in paperd['abstract'][:encoded_sents]:
            outf.write('{:d}: {:s}\n'.format(sent_i, sent))
            sent_i += 1
        outf.write('\n')
    print(f'Wrote: {outf.name}')
    outf.close()
    

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # This creates a dummy user file manually.
    make_seed_papers = subparsers.add_parser('make_seed_paper_json')
    make_seed_papers.add_argument('--username', required=True,
                                  help='The user to build reps for.')
    make_seed_papers.add_argument('--data_path', required=True,
                                  help='Path to directory with jsonl data.')
    # Build embeddings for the seed papers submitted/expanded from user papers.
    build_vecs_args = subparsers.add_parser('build_seed_reps')
    build_vecs_args.add_argument('--model_name', required=True,
                                 choices=['cospecter', 'miswordbienc'],
                                 help='The name of the model to run.')
    build_vecs_args.add_argument('--username', required=True,
                                 help='The user to build reps for.')
    build_vecs_args.add_argument('--condition', required=True,
                                 help='Otter or Maple')
    build_vecs_args.add_argument('--data_path', required=True,
                                 help='Path to directory with jsonl data.')
    # Retrieve keyphrases for all the users papers.
    build_kps_args = subparsers.add_parser('get_seed_paper_kps')
    build_kps_args.add_argument('--username', required=True,
                                help='The user to build reps for.')
    build_kps_args.add_argument('--condition', required=True,
                                help='Otter or Maple')
    build_kps_args.add_argument('--data_path', required=True,
                                help='Path to directory with jsonl data.')
    # Aggregate the keyphrases across papers for a user.
    build_kps_args = subparsers.add_parser('aggregate_user_kps')
    build_kps_args.add_argument('--username', required=True,
                                help='The user to build reps for.')
    build_kps_args.add_argument('--condition', required=True,
                                help='Otter or Maple')
    build_kps_args.add_argument('--data_path', required=True,
                                help='Path to directory with jsonl data.')
    cl_args = parser.parse_args()
    if cl_args.subcommand == 'build_seed_reps':
        out_path = os.path.join(cl_args.data_path, 'users', cl_args.username, cl_args.condition)
        if cl_args.model_name in {'cospecter'}:
            write_wholeabs_reps(out_path=out_path,
                                username=cl_args.username,
                                model_name=cl_args.model_name, condition=cl_args.condition)
        elif cl_args.model_name in {'miswordbienc'}:
            write_seedabs_sent_reps(out_path=out_path,
                                    username=cl_args.username, condition=cl_args.condition)
    elif cl_args.subcommand == 'get_seed_paper_kps':
        out_path = os.path.join(cl_args.data_path, 'users', cl_args.username, cl_args.condition)
        get_user_paper_forecitekps_consent(data_path=cl_args.data_path, username=cl_args.username,
                                           user_path=out_path, condition=cl_args.condition)
    elif cl_args.subcommand == 'aggregate_user_kps':
        out_path = os.path.join(cl_args.data_path, 'users', cl_args.username, cl_args.condition)
        aggregate_user_kps(data_path=cl_args.data_path, username=cl_args.username,
                           user_path=out_path, condition=cl_args.condition)
        

if __name__ == "__main__":
    main()
