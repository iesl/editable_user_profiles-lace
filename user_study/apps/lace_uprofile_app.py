"""
Build an editable user profile based recommender.
- Read the users json and read their paper reps and keyphrases into memory.
- Read the candidates document (first stage retrieval) and
    sentence embeddings into memory (second stage retrieval).
- Display the keyphrases to users and ask them to check it.
- Use the keyphrases and sentence embeddings to compute keyphrase values.
- Display the keyphrase selection box to users for retrieval.
- Use the selected keyphrases for performing retrieval.
"""
import copy
import json
import pickle
import joblib
import os
import collections

import streamlit as st
import numpy as np
from scipy.spatial import distance
from scipy import special
from sklearn.neighbors import NearestNeighbors
from sentence_transformers import SentenceTransformer, models
import torch
import ot


in_path = './data'


########################################
#              BACKEND CODE            #
########################################
def read_user(seed_json):
    """
    Given the seed json for the user read the embedded
    documents for the user.
    :param seed_json:
    :return:
    """
    if 'doc_vectors_user' not in st.session_state:
        uname = seed_json['username']
        user_kps = seed_json['user_kps']
        # Read document vectors.
        doc_vectors_user = np.load(os.path.join(in_path, 'users', uname, f'embeds-{uname}-doc.npy'))
        with open(os.path.join(in_path, 'users', uname, f'pid2idx-{uname}-doc.json'), 'r') as fp:
            pid2idx_user = json.load(fp)
        # Read sentence vectors.
        pid2sent_vectors = joblib.load(os.path.join(in_path, 'users', uname, f'embeds-{uname}-sent.pickle'))
        pid2sent_vectors_user = collections.OrderedDict()
        for pid in sorted(pid2sent_vectors):
            pid2sent_vectors_user[pid] = pid2sent_vectors[pid]
        st.session_state['doc_vectors_user'] = doc_vectors_user
        st.session_state['pid2idx_user'] = pid2idx_user
        st.session_state['pid2sent_vectors_user'] = pid2sent_vectors_user
        st.session_state['user_kps'] = user_kps
        st.session_state['username'] = uname
        st.session_state['seed_titles'] = []
        for pd in seed_json['papers']:
            norm_title = " ".join(pd['title'].lower().strip().split())
            st.session_state.seed_titles.append(norm_title)
        return doc_vectors_user, pid2idx_user, pid2sent_vectors, user_kps
    else:
        return st.session_state.doc_vectors_user, st.session_state.pid2idx_user, \
               st.session_state.pid2sent_vectors_user, st.session_state.user_kps


def first_stage_ranked_docs(user_doc_queries, per_doc_to_rank, total_to_rank=2000):
    """
    Return a list of ranked documents given a set of queries.
    :param user_doc_queries: read the cached query embeddings
    :return:
    """
    if 'first_stage_ret_pids' not in st.session_state:
        # read the document vectors
        doc_vectors = np.load(os.path.join(in_path, 'cands', 'embeds-s2orccompsci-100k.npy'))
        with open(os.path.join(in_path, 'cands', 'pid2idx-s2orccompsci-100k.pickle'), 'rb') as fp:
            pid2idx_cands = pickle.load(fp)
            idx2pid_cands = dict([(v, k) for k, v in pid2idx_cands.items()])
        # index the vectors into a nearest neighbors structure
        neighbors = NearestNeighbors(n_neighbors=per_doc_to_rank)
        neighbors.fit(doc_vectors)
        st.session_state['neighbors'] = neighbors
        st.session_state['idx2pid_cands'] = idx2pid_cands
        
        # Get the dists for all the query docs.
        nearest_dists, nearest_idxs = neighbors.kneighbors(user_doc_queries, return_distance=True)
        
        # Get the docs
        top_pids = []
        uniq_top = set()
        for ranki in range(per_doc_to_rank):  # Save papers by rank position for debugging.
            for qi in range(user_doc_queries.shape[0]):
                idx = nearest_idxs[qi, ranki]
                pid = idx2pid_cands[idx]
                if pid not in uniq_top:  # Only save the unique papers. (ignore multiple retrievals of the same paper)
                    top_pids.append(pid)
                    uniq_top.add(pid)
        top_pids = top_pids[:total_to_rank]
        st.session_state['first_stage_ret_pids'] = top_pids
        return top_pids
    else:
        return st.session_state.first_stage_ret_pids


def read_kp_encoder(in_path):
    """
    Read the kp encoder model from disk.
    :param in_path: string;
    :return:
    """
    if 'kp_enc_model' not in st.session_state:
        word_embedding_model = models.Transformer(os.path.join(in_path, 'models', 'scibert_scivocab_uncased'),
                                                  max_seq_length=512)
        trained_model_fname = os.path.join(in_path, 'models', 'kp_encoder_cur_best.pt')
        if torch.cuda.is_available():
            saved_model = torch.load(trained_model_fname)
        else:
            saved_model = torch.load(trained_model_fname, map_location=torch.device('cpu'))
        word_embedding_model.auto_model.load_state_dict(saved_model)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        kp_enc_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        st.session_state['kp_enc_model'] = kp_enc_model
    else:
        return st.session_state.kp_enc_model
    
    
def read_candidates(in_path):
    """
    Read candidate papers into pandas dataframe.
    :param in_path:
    :return:
    """
    if 'pid2abstract' not in st.session_state:
        with open(os.path.join(in_path, 'cands', 'abstracts-s2orccompsci-100k.pickle'), 'rb') as fp:
            pid2abstract = pickle.load(fp)
        # read the sentence vectors
        pid2sent_vectors = joblib.load(os.path.join(in_path, 'cands', f'embeds-sent-s2orccompsci-100k.pickle'))
        st.session_state['pid2sent_vectors_cands'] = pid2sent_vectors
        st.session_state['pid2abstract'] = pid2abstract
        return pid2abstract, pid2sent_vectors
    else:
        return st.session_state.pid2abstract, st.session_state.pid2sent_vectors_cands


def get_kp_embeddings(profile_keyphrases):
    """
    Embed the passed profike keyphrases
    :param profile_keyphrases: list(string)
    :return:
    """
    kp_enc_model = st.session_state['kp_enc_model']
    if 'kp_vectors_user' not in st.session_state:
        kp_embeddings = kp_enc_model.encode(profile_keyphrases)
        kp_vectors_user = collections.OrderedDict()
        for i, kp in enumerate(profile_keyphrases):
            kp_vectors_user[kp] = kp_embeddings[i, :]
        st.session_state['kp_vectors_user'] = kp_vectors_user
        return kp_vectors_user
    else:
        uncached_kps = [kp for kp in profile_keyphrases if kp not in st.session_state.kp_vectors_user]
        kp_embeddings = kp_enc_model.encode(uncached_kps)
        for i, kp in enumerate(uncached_kps):
            st.session_state.kp_vectors_user[kp] = kp_embeddings[i, :]
        return st.session_state.kp_vectors_user
    

def generate_profile_values(profile_keyphrases):
    """
    - Read sentence embeddings
    - Read profile keyphrase embeddings
    - Compute alignment from sentences to keyphrases
    - Barycenter project the keyphrases to sentences to get kp values
    - Return the kp values
    :param profile_keyphrases: list(string)
    :return:
    """
    kp_embeddings = get_kp_embeddings(profile_keyphrases)
    # Read sentence embeddings.
    user_seed_sentembeds = np.vstack(list(st.session_state.pid2sent_vectors_user.values()))
    # Read keyphrase embeddings.
    kps_embeds_flat = []
    for kp in profile_keyphrases:
        kps_embeds_flat.append(kp_embeddings[kp])
    kps_embeds_flat = np.vstack(kps_embeds_flat)
    # Compute transport plan from sentence to keyphrases.
    pair_dists = distance.cdist(user_seed_sentembeds, kps_embeds_flat, 'euclidean')
    a_distr = [1 / user_seed_sentembeds.shape[0]] * user_seed_sentembeds.shape[0]
    b_distr = [1 / kps_embeds_flat.shape[0]] * kps_embeds_flat.shape[0]
    # tplan = ot.bregman.sinkhorn_epsilon_scaling(a_distr, b_distr, pair_dists, 0.05, numItermax=2000)
    tplan = ot.partial.entropic_partial_wasserstein(a_distr, b_distr, pair_dists, 0.05, m=0.8)
    # Barycenter project the keyphrases to the sentences: len(profile_keyphraases) x embedding_dim
    proj_kp_vectors = np.matmul(user_seed_sentembeds.T, tplan).T
    norm = np.sum(tplan, axis=0)
    kp_value_vectors = proj_kp_vectors/norm[:, np.newaxis]
    # Return as a dict.
    kp2valvectors = {}
    for i, kp in enumerate(profile_keyphrases):
        kp2valvectors[kp] = kp_value_vectors[i, :]
    return kp2valvectors, tplan


def second_stage_ranked_docs(selected_query_kps, first_stage_pids, pid2abstract, pid2sent_reps_cand, to_rank=30):
    """
    Return a list of ranked documents given a set of queries.
    :param first_stage_pids: list(string)
    :param pid2abstract: dict(pid: paperd)
    :param query_paper_idxs: list(int);
    :return:
    """
    if len(selected_query_kps) < 3:
        topk = len(selected_query_kps)
    else:  # Use 20% of keyphrases for scoring or 3 whichever is larger
        topk = max(int(len(st.session_state.kp2val_vectors)*0.2), 3)
    query_kp_values = np.vstack([st.session_state.kp2val_vectors[kp] for kp in selected_query_kps])
    pid2topkdist = dict()
    pid2kp_expls = collections.defaultdict(list)
    for i, pid in enumerate(first_stage_pids):
        sent_reps = pid2sent_reps_cand[pid]
        pair_dists = distance.cdist(query_kp_values, sent_reps)
        # Pick the topk unique profile concepts.
        kp_ind = np.argsort(pair_dists.min(axis=1))[:topk]
        sub_pair_dists = pair_dists[kp_ind, :]
        # sub_kp_reps = query_kp_values[kp_ind, :]
        a_distr = special.softmax(-1*np.min(sub_pair_dists, axis=1))
        b_distr = [1 / sent_reps.shape[0]] * sent_reps.shape[0]
        tplan = ot.bregman.sinkhorn_epsilon_scaling(a_distr, b_distr, sub_pair_dists, 0.05)
        wd = np.sum(sub_pair_dists * tplan)
        # topk_dist = 0
        # for k in range(topk):
        #     topk_dist += pair_dists[kp_ind[k], sent_ind[k]]
        #     pid2kp_expls[pid].append(selected_query_kps[kp_ind[k]])
        # pid2topkdist[pid] = topk_dist
        pid2topkdist[pid] = wd
    
    top_pids = sorted(pid2topkdist, key=pid2topkdist.get)
    
    # Get the docs
    retrieved_papers = collections.OrderedDict()
    for pid in top_pids:
        # Exclude papers from the seed set in the result set.
        norm_title = " ".join(pid2abstract[pid]['title'].lower().strip().split())
        if norm_title in st.session_state.seed_titles:
            continue
        retrieved_papers[pid2abstract[pid]['title']] = {
            'title': pid2abstract[pid]['title'],
            'kp_explanations': pid2kp_expls[pid],
            'abstract': pid2abstract[pid]['abstract']
        }
        if len(retrieved_papers) == to_rank:
            break
    return retrieved_papers
    

########################################
#              HELPER CODE             #
########################################
def parse_input_kps(unparsed_kps, initial_user_kps):
    """
    Function to parse the input keyphrase string.
    :return:
    """
    if unparsed_kps.strip():
        kps = unparsed_kps.split(',')
        parsed_user_kps = []
        uniq_kps = set()
        for kp in kps:
            kp = kp.strip()
            if kp not in uniq_kps:
                parsed_user_kps.append(kp)
                uniq_kps.add(kp)
    else:  # If its an empty string use the initial kps
        parsed_user_kps = copy.copy(initial_user_kps)
    return parsed_user_kps


# def plot_sent_kp_alignment(tplan, kp_labels, sent_labels):
#     """
#     Plot the sentence keyphrase alignment.
#     :return:
#     """
#     fig, ax = plt.subplots()
#     h = sns.heatmap(tplan.T, linewidths=.3, xticklabels=sent_labels,
#                     yticklabels=kp_labels, cmap='Blues')
#     h.tick_params('y', labelsize=5)
#     h.tick_params('x', labelsize=2)
#     plt.tight_layout()
#     return fig

    
def multiselect_title_formatter(title):
    """
    Format the multi-select titles.
    :param title: string
    :return: string: formatted title
    """
    ftitle = title.split()[:5]
    return ' '.join(ftitle) + '...'


def format_abstract(paperd, to_display=3, markdown=True):
    """
    Given a dict with title and abstract return
    a formatted text for rendering with markdown.
    :param paperd:
    :param to_display:
    :return:
    """
    if len(paperd['abstract']) < to_display:
        sents = ' '.join(paperd['abstract'])
    else:
        sents = ' '.join(paperd['abstract'][:to_display]) + '...'
    try:
        kp_expl = ', '.join(paperd['kp_explanations'])
    except KeyError:
        kp_expl = ''
    if markdown:
        par = '<p><b>Title</b>: <i>{:s}</i><br><b>Abstract</b>: {:s}<br><i>{:s}</i></p>'.\
            format(paper['title'], sents, kp_expl)
    else:
        par = 'Title: {:s}; Abstract: {:s}'.format(paper['title'], sents)
    return par


def perp_result_json():
    """
    Create a json with the results retrieved for each
    iteration and the papers users choose to save at
    each step.
    :return:
    """
    result_json = {}
    # print(len(st.session_state.i_selections))
    # print(len(st.session_state.i_resultps))
    # print(len(st.session_state.i_savedps))
    # print(st.session_state.tuning_i)
    assert(len(st.session_state.i_selections) == len(st.session_state.i_resultps)
           == len(st.session_state.i_savedps) == st.session_state.tuning_i)
    for tuning_i, i_pselects, (_, i_savedps) in zip(range(st.session_state.tuning_i), st.session_state.i_selections,
                                                    st.session_state.i_savedps.items()):
        iterdict = {
            'iteration': tuning_i,
            'profile_selections': copy.deepcopy(i_pselects),
            'saved_papers': copy.deepcopy(list(i_savedps.items()))
        }
        result_json[tuning_i] = iterdict
    result_json['condition'] = 'maple'
    result_json['username'] = st.session_state.username
    return json.dumps(result_json)


########################################
#              APP CODE                #
########################################
st.title('\U0001F341 Maple Paper Recommender \U0001F341')
st.markdown(
        '\U0001F341 Maple \U0001F341 makes controllable paper recommendations personalized to you based on a \U0001F331 seed set '
        '\U0001F331 of papers. The seed set of papers is used to build a \U0001F9D1 personalized profile \U0001F9D1 of keyphrases '
        'which describe the seed papers. These are your profile descriptors. You can change your recommendations by editing '
        'the list of descriptors, or including or excluding descriptors.')

# Load candidate documents and models.
pid2abstract_cands, pid2sent_vectors_cands = read_candidates(in_path)
kp_encoding_model = read_kp_encoder(in_path)

# Initialize the session state:
if 'tuning_i' not in st.session_state:
    st.session_state['tuning_i'] = 0
    # Save the profile keyphrases at every run
    # (run is every time the script runs, iteration is every time recs are requested)
    st.session_state['run_user_kps'] = []
    # Save the profile selections at each iteration
    st.session_state['i_selections'] = []
    # dict of dicts: tuning_i: dict(paper_title: paper)
    st.session_state['i_resultps'] = {}
    # dict of dicts: tuning_i: dict(paper_title: saved or not bool)
    st.session_state['i_savedps'] = collections.defaultdict(dict)

# Ask user to upload a set of seed query papers.
with st.sidebar:
    uploaded_file = st.file_uploader("\U0001F331 Upload seed papers",
                                     type='json',
                                     help='Upload a json file with titles and abstracts of the papers to '
                                          'include in your profile.')
    if uploaded_file is not None:
        user_papers = json.load(uploaded_file)
        # Read user data.
        doc_vectors_user, pid2idx_user, pid2sent_vectors_user, user_kps = read_user(user_papers)
        st.session_state.run_user_kps.append(copy.copy(user_kps))
        display_profile_kps = ', '.join(user_kps)
        # Perform first stage retrieval.
        first_stage_ret_pids = first_stage_ranked_docs(user_doc_queries=doc_vectors_user, per_doc_to_rank=500)
        with st.expander("Examine seed papers"):
            st.markdown(f'**Initial profile descriptors**:')
            st.markdown(display_profile_kps)
            st.markdown('**Seed papers**:')
            for paper in user_papers['papers']:
                par = format_abstract(paperd=paper, to_display=6)
                st.markdown(par, unsafe_allow_html=True)

    st.markdown('\u2b50 Saved papers')

if uploaded_file is not None:
    # Create a text box where users can see their profile keyphrases.
    st.subheader('\U0001F4DD Seed paper descriptors')
    with st.form('profile_kps'):
        input_kps = st.text_area('Edit seed descriptors:', display_profile_kps,
                                 help='Edit the profile descriptors if they are redundant, incomplete, nonsensical, '
                                      'or dont describe the seed papers. OR if you would like the descriptors to '
                                      'capture aspects of the seed papers that the descriptors dont currently capture.',
                                 placeholder='If left empty initial profile descriptors will be used...')
        input_user_kps = parse_input_kps(unparsed_kps=input_kps, initial_user_kps=user_kps)
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            generate_profile = st.form_submit_button('\U0001F9D1 Generate profile \U0001F9D1')

    if generate_profile:
        prev_run_input_kps = st.session_state.run_user_kps[-1]
        if set(prev_run_input_kps) == set(input_user_kps):  # If there is no change then use
            if 'kp2val_vectors' in st.session_state:  # This happens all the time except the first run.
                kp2val_vectors = st.session_state.kp2val_vectors
                user_tplan = st.session_state.user_tplan
            else:  # This happens on the first run.
                with st.spinner(text="Generating profile..."):
                    kp2val_vectors, user_tplan = generate_profile_values(profile_keyphrases=input_user_kps)
                st.session_state['kp2val_vectors'] = kp2val_vectors
                st.session_state['user_tplan'] = user_tplan
        else:
            with st.spinner(text="Generating profile..."):
                kp2val_vectors, user_tplan = generate_profile_values(profile_keyphrases=input_user_kps)
            st.session_state['kp2val_vectors'] = kp2val_vectors
            st.session_state['user_tplan'] = user_tplan
            st.session_state.run_user_kps.append(copy.copy(input_user_kps))
    
    # Create a multiselect dropdown
    if 'kp2val_vectors' in st.session_state:
        # with st.expander("Examine paper-descriptor alignment"):
        #     user_tplan = st.session_state.user_tplan
        #     fig = plot_sent_kp_alignment(tplan=user_tplan, kp_labels=input_user_kps,
        #                                  sent_labels=range(user_tplan.shape[0]))
        #     st.write(fig)
            
        st.subheader('\U0001F9D1 Profile descriptors for ranking')
        with st.form('profile_input'):
            profile_selections = st.multiselect(label='Add or remove profile descriptors to use for recommendations:',
                                                default=input_user_kps,  # Use all the values by default.
                                                options=input_user_kps,
                                                help='Items selected here will be used for creating your '
                                                     'recommended list')
            col1, col2, col3 = st.columns([1, 1, 1])
            with col2:
                generate_recs = st.form_submit_button('\U0001F9ED Recommend papers \U0001F9ED')
    
        # Use the uploaded files to create a ranked list of items.
        if generate_recs and profile_selections:
            # st.write('Generating recs...')
            st.session_state.tuning_i += 1
            st.session_state.i_selections.append(copy.deepcopy(profile_selections))
            with st.spinner(text="Recommending papers..."):
                top_papers = second_stage_ranked_docs(first_stage_pids=first_stage_ret_pids,
                                                      selected_query_kps=profile_selections,
                                                      pid2abstract=pid2abstract_cands,
                                                      pid2sent_reps_cand=pid2sent_vectors_cands,
                                                      to_rank=30)
            st.session_state.i_resultps[st.session_state.tuning_i] = copy.deepcopy(top_papers)
    
        # Read off from the result cache and allow users to save some papers.
        if st.session_state.tuning_i in st.session_state.i_resultps:
            # st.write('Waiting for selections...')
            cached_top_papers = st.session_state.i_resultps[st.session_state.tuning_i]
            for paper in cached_top_papers.values():
                # This statement ensures correctness for when users unselect a previously selected item.
                st.session_state.i_savedps[st.session_state.tuning_i][paper['title']] = False
                dcol1, dcol2 = st.columns([1, 16])
                with dcol1:
                    save_paper = st.checkbox('\u2b50', key=paper['title'])
                with dcol2:
                    plabel = format_abstract(paperd=paper, to_display=2, markdown=True)
                    st.markdown(plabel, unsafe_allow_html=True)
                    with st.expander('See more..'):
                        full_abstract = ' '.join(paper['abstract'])
                        st.markdown(full_abstract, unsafe_allow_html=True)
                if save_paper:
                    st.session_state.i_savedps[st.session_state.tuning_i].update({paper['title']: True})
    
        # Print the saved papers across iterations in the sidebar.
        with st.sidebar:
            with st.expander("Examine saved papers"):
                # st.write('Later write..')
                # st.write(st.session_state.i_savedps)
                for iteration, savedps in st.session_state.i_savedps.items():
                    st.markdown('Iteration: {:}'.format(iteration))
                    for papert, saved in savedps.items():
                        if saved:
                            fpapert = '<p style=color:Gray; ">- {:}</p>'.format(papert)
                            st.markdown('{:}'.format(fpapert), unsafe_allow_html=True)
            if st.session_state.tuning_i > 0:
                st.download_button('Download papers', perp_result_json(), mime='json',
                                   help='Download the papers saved in the session.')
                with st.expander("Copy saved papers to clipboard"):
                    st.write(json.loads(perp_result_json()))
            