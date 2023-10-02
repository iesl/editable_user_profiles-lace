"""
Build a simple item-knn recommender with tfidf based retrieval.
"""
import copy
import json
import pickle
import os
import collections

import streamlit as st
import numpy as np
from sklearn.neighbors import NearestNeighbors

in_path = './data'

to_rank_over = 35

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
        doc_vectors_user = np.load(os.path.join(in_path, uname, f'embeds-{uname}-doc.npy'))
        with open(os.path.join(in_path, uname, f'pid2idx-{uname}-doc.json'), 'r') as fp:
            pid2idx_user = json.load(fp)
        st.session_state['doc_vectors_user'] = doc_vectors_user
        st.session_state['pid2idx_user'] = pid2idx_user
        st.session_state['username'] = uname
        st.session_state['seed_titles'] = []
        for pd in seed_json['papers']:
            norm_title = " ".join(pd['title'].lower().strip().split())
            st.session_state.seed_titles.append(norm_title)
        return doc_vectors_user, pid2idx_user
    else:
        return st.session_state.doc_vectors_user, st.session_state.pid2idx_user
        
    
def read_candidates(in_path, count=10):
    """
    Read candidate papers into pandas dataframe.
    :param in_path:
    :return:
    """
    if 'pid2abstract' not in st.session_state:
        with open(os.path.join(in_path, 'abstracts-s2orccompsci-100k.pickle'), 'rb') as fp:
            pid2abstract = pickle.load(fp)
        st.session_state['pid2abstract'] = pid2abstract
        return pid2abstract
    else:
        return st.session_state.pid2abstract
    

def index_candidates(in_path):
    """
    - Read the embedded candidate documents.
    Index the candidates passed for nearest neighbor retrieval.
    :param in_path:
    :return:
    """
    if 'neighbors' not in st.session_state:
        doc_vectors = np.load(os.path.join(in_path, 'embeds-s2orccompsci-100k.npy'))
        with open(os.path.join(in_path, 'pid2idx-s2orccompsci-100k.pickle'), 'rb') as fp:
            pid2idx_cands = pickle.load(fp)
            idx2pid_cands = dict([(v, k) for k, v in pid2idx_cands.items()])
        # index the vectors into a nearest neighbors structure
        neighbors = NearestNeighbors(n_neighbors=to_rank_over)
        neighbors.fit(doc_vectors)
        st.session_state['neighbors'] = neighbors
        st.session_state['idx2pid_cands'] = idx2pid_cands
        
        return neighbors, idx2pid_cands
    else:
        return st.session_state.neighbors, st.session_state.idx2pid_cands


def generate_ranked_docs(index, doc_vectors_user, pid2abstract, idx2pid_cands, query_paper_idxs, to_rank=30):
    """
    Return a list of ranked documents given a set of queries.
    :param index: nearest neighbor index of candidate papers
    :param doc_vectors_user: read the cached query embeddings
    :param pid2abstract: dict(pid: paperd)
    :param query_paper_idxs: list(int);
    :return:
    """
    # Create the vectors for query docs.
    qvecs = doc_vectors_user[query_paper_idxs, :]
    # print(qvecs.shape)
    # Get the dists for all the query docs.
    nearest_dists, nearest_idxs = index.kneighbors(qvecs, return_distance=True)
    # Get the minimum distance from all the distances
    # First flatten; multiply by -1 since we'll get the largest values.
    nd_flat = -1*nearest_dists.flatten(order='C')
    ni_flat = nearest_idxs.flatten(order='C')
    # Then get topk indices.
    topk_ind = np.argpartition(nd_flat, -to_rank_over)[-to_rank_over:]
    sorted_topk_ind = topk_ind[np.argsort(-1*nd_flat[topk_ind])]
    # Get indices in the original space of documents.
    sorted_dists = nd_flat[sorted_topk_ind]
    # print(sorted_dists)
    sorted_indices = ni_flat[sorted_topk_ind]
    
    # Get the docs
    top_pids = []
    for i in sorted_indices.tolist():
        pid = idx2pid_cands[i]
        if pid not in top_pids:  # Only save the unique papers. (ignore multiple retrievals of the same paper)
            top_pids.append(pid)
    retrieved_papers = collections.OrderedDict()
    for pid in top_pids:
        # Exclude papers from the seed set in the result set.
        norm_title = " ".join(pid2abstract[pid]['title'].lower().strip().split())
        if norm_title in st.session_state.seed_titles:
            continue
        retrieved_papers[pid2abstract[pid]['title']] = {
            'title': pid2abstract[pid]['title'],
            'abstract': pid2abstract[pid]['abstract']
        }
        if len(retrieved_papers) == to_rank:
            break
    return retrieved_papers
    

########################################
#              HELPER CODE             #
########################################
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
    if markdown:
        par = '<p><b>Title</b>: <i>{:s}</i><br><b>Abstract</b>: {:s}</p>'.format(paper['title'], sents)
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
    result_json['condition'] = 'otter'
    result_json['username'] = st.session_state.username
    return json.dumps(result_json)


########################################
#              APP CODE                #
########################################
st.title('\U0001F9A6 Otter Paper Recommender \U0001F9A6')
st.markdown(
        '\U0001F9A6 Otter \U0001F9A6 makes controllable paper recommendations personalized to you based on a \U0001F331 seed set '
        '\U0001F331 of papers. The seed set of papers represents a \U0001F9D1 personalized profile \U0001F9D1 for you.'
        ' You can change your recommendations by including or excluding papers in the seed set of papers.')

# Load candidate documents.
pid2abstract_cands = read_candidates(in_path, count=10000)
nn_index_cands, idx2pid_cands = index_candidates(in_path=in_path)

# Initialize the session state:
if 'tuning_i' not in st.session_state:
    st.session_state['tuning_i'] = 0
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
        doc_vectors_user, pid2idx_user = read_user(user_papers)
        with st.expander("Examine seed papers"):
            for paper in user_papers['papers']:
                par = format_abstract(paperd=paper, to_display=6)
                st.markdown(par, unsafe_allow_html=True)
        seed_title2paper_idx = collections.OrderedDict([(d['title'], pidx) for pidx, d in enumerate(user_papers['papers'])])
        titles = [d['title'] for d in user_papers['papers']]

    st.markdown('\u2b50 Saved papers')

if uploaded_file is not None:
    # Create a multiselect dropdown
    st.subheader('\U0001F9D1 Profile items for ranking')
    with st.form('profile_input'):
        profile_selections = st.multiselect(label='Add or remove profile items to use for recommendations',
                                            default=titles,  # Use all the values by default.
                                            options=titles,
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
        query_title2paper_idx = [seed_title2paper_idx[title] for title in profile_selections]
        top_papers = generate_ranked_docs(query_paper_idxs=query_title2paper_idx, doc_vectors_user=doc_vectors_user,
                                          index=nn_index_cands, pid2abstract=pid2abstract_cands,
                                          idx2pid_cands=idx2pid_cands, to_rank=30)
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
            