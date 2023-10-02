"""
Use the Semantic Scholar API and access author data
and citation network data to expand the set of user
submitted papers.
"""
import argparse
import codecs
import collections
import os
import json
import sys
import time
import random
import requests
import spacy

scispacy_model = spacy.load("en_core_sci_sm")
scispacy_model.add_pipe('sentencizer')


def gather_author_papers(s2_authorid, username, data_path):
    """
    Given an author id download the title and abstracts for their papers.
    :param s2_authorid: string
    :param username: string; email name mostly.
    :param data_path: string; where to write the seed json set
    :return:
    """
    assert ('otter' in data_path or 'maple' in data_path)
    if 'otter' in data_path:
        condition = 'otter'
    else:
        condition = 'maple'
    # out_path = os.path.join(data_path, author_username)
    # du.create_dir(out_path)
    # Get the most recent 100 papers.
    result = requests.get("https://api.semanticscholar.org/graph/v1/author/"
                          f"{s2_authorid}/papers?fields=year,title,abstract&limit=100")
    # Check straightforward errors.
    if result.status_code != 200:
        print(f'{username} request failed with code: {result.status_code}')
        sys.exit()
    result_json = result.json()
    if len(result_json['data']) == 0:
        print(f'{username} request returned no data')
        sys.exit()
    print(f'{username} got papers: {len(result_json["data"])}')
    
    # Make the paper json.
    seed_set = {
        'username': username,
        's2_authorid': s2_authorid,
        'papers': []
    }
    papers_missing_data = 0
    for paper in result_json['data']:
        if paper['abstract'] and paper['title']:
            abstract_sents = scispacy_model(paper['abstract'],
                                            disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                     'lemmatizer', 'parser', 'ner'])
            abstract_sents = [sent.text for sent in abstract_sents.sents]
            paperd = {
                'title': paper['title'],
                'abstract': abstract_sents
            }
            seed_set['papers'].append(paperd)
            if len(seed_set['papers']) == 20:
                break
        else:
            papers_missing_data += 1
    print(f'{username} missing papers: {papers_missing_data}')
    print(f'{username} seed set: {len(seed_set["papers"])}')
    with codecs.open(os.path.join(data_path, f'seedset-{username}-{condition}.json'), 'w', 'utf-8') as fp:
        json.dump(seed_set, fp, indent=2)
        print(f'Wrote: {fp.name}')


def get_citing_paper_data(p_url, username):
    """
    Make a request and return the result json.
    :param rstring:
    :return:
    """
    rstring = f'https://api.semanticscholar.org/graph/v1/paper/URL:{p_url}/citations?fields=title,abstract,citationCount'
    # Get the most recent 100 papers.
    result = requests.get(rstring)
    # Check straightforward errors.
    if result.status_code != 200:
        print(f'{result.status_code} fail: {p_url}')
        sys.exit()
    result_json = result.json()
    if len(result_json['data']) == 0:
        print(f'Returned no data: {p_url}')
        return []
    ref_papers = []
    for paper in result_json['data']:
        paper = paper['citingPaper']
        if paper['abstract'] and paper['title']:
            if paper['citationCount'] > 5000:
                print(f'Skipping from refs: {paper["title"]}')
                continue
            abstract_sents = scispacy_model(paper['abstract'],
                                            disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                     'lemmatizer', 'parser', 'ner'])
            abstract_sents = [sent.text for sent in abstract_sents.sents]
            paperd = {
                'title': paper['title'],
                'abstract': abstract_sents
            }
            ref_papers.append(paperd)
    # Return in a deterministic title sort order.
    ref_papers = list(sorted(ref_papers, key=lambda d: d['title']))
    return ref_papers


def get_ref_paper_data(p_url, username):
    """
    Make a request and return the result json.
    :param rstring:
    :return:
    """
    rstring = f'https://api.semanticscholar.org/graph/v1/paper/URL:{p_url}/references?fields=title,abstract,citationCount'
    # Get the most recent 100 papers.
    result = requests.get(rstring)
    # Check straightforward errors.
    if result.status_code != 200:
        print(f'{result.status_code} fail: {p_url}')
        sys.exit()
    result_json = result.json()
    if len(result_json['data']) == 0:
        print(f'Returned no data: {p_url}')
        return []
    ref_papers = []
    for paper in result_json['data']:
        paper = paper['citedPaper']
        if paper['abstract'] and paper['title']:
            if paper['citationCount'] > 5000:
                print(f'Skipping from refs: {paper["title"]}')
                continue
            abstract_sents = scispacy_model(paper['abstract'],
                                            disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                     'lemmatizer', 'parser', 'ner'])
            abstract_sents = [sent.text for sent in abstract_sents.sents]
            paperd = {
                'title': paper['title'],
                'abstract': abstract_sents
            }
            ref_papers.append(paperd)
    # Return in a deterministic title sort order.
    ref_papers = list(sorted(ref_papers, key=lambda d: d['title']))
    return ref_papers


def get_paper_data(p_url, username):
    """
    Get a papers data given its url.
    :param rstring:
    :param username:
    :return:
    """
    rstring = f'https://api.semanticscholar.org/graph/v1/paper/URL:{p_url}?fields=title,abstract'
    # Get the most recent 100 papers.
    result = requests.get(rstring)
    # Check straightforward errors.
    if result.status_code != 200:
        print(f'{p_url} request failed with code: {result.status_code}')
        sys.exit()
    result_json = result.json()
    if result_json['abstract'] and result_json['title']:
        abstract_sents = scispacy_model(result_json['abstract'],
                                        disable=['tok2vec', 'tagger', 'attribute_ruler',
                                                 'lemmatizer', 'parser', 'ner'])
        abstract_sents = [sent.text for sent in abstract_sents.sents]
        paperd = {
            'title': result_json['title'],
            'abstract': abstract_sents
        }
        return paperd
        

def gather_citationnw_papers(username, data_path):
    """
    Given a text file with URLs for every paper expand the papers
    based on outgoing citations from the paper.
    - Assuming that there will be 2 files per user. One for Nomad and LACE each.
    :param username: string; email name mostly.
    :param data_path: string; where to write the seed json set
    :return:
    """
    assert('otter' in data_path or 'maple' in data_path)
    if 'otter' in data_path:
        condition = 'otter'
    else:
        condition = 'maple'
    random.seed(3230)
    # out_path = os.path.join(data_path, username)
    # du.create_dir(out_path)
    
    upaper_urls = []
    with codecs.open(os.path.join(data_path, f'{username}_links.txt')) as fp:
        for line in fp:
            upaper_urls.append(line.strip())
    # For the url papers get the title and abstract.
    papers = []
    for p_url in upaper_urls:
        paperd = get_paper_data(p_url, username)
        if paperd:
            papers.append(paperd)
        else:
            print(f'Submitted paper missing data: {p_url}')
    print(f'{username} URLS: {len(upaper_urls)}; Got data: {len(papers)}')
    seed_set = {
        'username': username,
        'papers': papers
    }
    print(f'{username} current seed set: {len(seed_set["papers"])}')
    
    # Get the papers references or citations.
    to_sample = int(20/len(upaper_urls))  # We want about 20 papers per user.
    uniq_titles = set()
    for i, p_url in enumerate(upaper_urls):
        reslist = get_ref_paper_data(p_url, username)
        if len(reslist) == 0:
            reslist = get_citing_paper_data(p_url, username)
        uniq_refs = []
        for paperd in reslist:
            if paperd['title'] not in uniq_titles:
                uniq_titles.add(paperd['title'])
                uniq_refs.append(paperd)
        if uniq_refs:
            # randomly sample some returned references.
            seed_set['papers'].extend(random.sample(uniq_refs, min(to_sample, len(uniq_refs))))
        else:
            print(f'Submitted paper missing refs: {p_url}')
        time.sleep(1)
    
    print(f'{username} seed set: {len(seed_set["papers"])}')
    with codecs.open(os.path.join(data_path, f'seedset-{username}-{condition}.json'), 'w', 'utf-8') as fp:
        json.dump(seed_set, fp, indent=2)
        print(f'Wrote: {fp.name}\n')
        

def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    make_seed_papers = subparsers.add_parser('download_author_seed_papers')
    make_seed_papers.add_argument('--author_uname', required=True,
                                  help='The user to build reps for.')
    make_seed_papers.add_argument('--author_s2id', required=True,
                                  help='The user to build reps for.')
    make_seed_papers.add_argument('--data_path', required=True,
                                  help='Path to directory with jsonl data.')
    seed_papers_refs = subparsers.add_parser('download_seed_paper_refs')
    seed_papers_refs.add_argument('--username', required=True,
                                  help='The user to build reps for.')
    seed_papers_refs.add_argument('--data_path', required=True,
                                  help='Path to directory with jsonl data.')
    cl_args = parser.parse_args()
    if cl_args.subcommand == 'download_author_seed_papers':
        gather_author_papers(username=cl_args.author_uname, data_path=cl_args.data_path,
                             s2_authorid=cl_args.author_s2id)
    elif cl_args.subcommand == 'download_seed_paper_refs':
        gather_citationnw_papers(username=cl_args.username, data_path=cl_args.data_path)


if __name__ == "__main__":
    main()
