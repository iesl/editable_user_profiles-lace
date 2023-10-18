"""
Implement baseline methods which are used for creating pooled papers.
"""
import os
import time
import codecs, json
import collections
import torch
from transformers import AutoModel, AutoTokenizer
import numpy as np

from . import data_utils as du
from ..learning.retrieval_models import content_profile_models, editable_profile_models
from ..learning import batchers


class TrainedModel:
    """
    Own trained model using which we want to build up document embeddings.
    """
    def __init__(self, model_name, trained_model_path, model_version='cur_best'):
        # Load label maps and configs.
        if trained_model_path:
            with codecs.open(os.path.join(trained_model_path, 'run_info.json'), 'r', 'utf-8') as fp:
                run_info = json.load(fp)
                all_hparams = run_info['all_hparams']
            # Init model:
            if model_name in {'cospecter'}:
                model = content_profile_models.MySPECTER(model_hparams=all_hparams)
            elif model_name in {'miswordbienc'}:
                model = content_profile_models.WordSentAlignBiEnc(model_hparams=all_hparams)
            elif model_name in {'upnfconsent'}:
                model = editable_profile_models.UPNamedFAspireKP(model_hparams=all_hparams)
            else:
                raise ValueError(f'Unknown model: {model_name}')
            model_fname = os.path.join(trained_model_path, 'model_{:s}.pt'.format(model_version))
            if torch.cuda.is_available():
                model.load_state_dict(torch.load(model_fname))
            else:
                model.load_state_dict(torch.load(model_fname, map_location=torch.device('cpu')))
            print(f'Loaded model: {model_fname}')
        else:
            # Models lile: 'sentence-transformers/all-mpnet-base-v2, sentence-transformers/bert-base-nli-mean-tokens',
            # allenai/aspire-contextualsentence-multim-compsci
            all_hparams = {
                'base-pt-layer': model_name,
                # Unnecessary but expected in model class.
                'score_aggregation': 'l2max'
            }
            model = content_profile_models.WordSentAlignBiEnc(model_hparams=all_hparams)
        # Move model to the GPU.
        if torch.cuda.is_available():
            model.cuda()
            print('Running on GPU.')
        model.eval()
        self.model_name = model_name
        self.model = model
        try:
            self.tokenizer_name = all_hparams['base-pt-layer']
        except KeyError:
            self.tokenizer_name = "allenai/specter"
        self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
    
    def predict(self, batch):
        """
        :param batch:
        :return:
        """
        with torch.no_grad():
            if self.model_name in {'cospecter'}:
                bert_batch, _, _ = batchers.SentTripleBatcher.prepare_bert_sentences(sents=batch, tokenizer=self.tokenizer)
                ret_dict = self.model.encode(batch_dict={'bert_batch': bert_batch})
                return ret_dict, ret_dict['doc_reps']
            # Go here: 'miswordbienc', 'upnfconsent', 'sentence-transformers/all-mpnet-base-v2', 'sentence-transformers/bert-base-nli-mean-tokens':
            else:
                batch_dict = batchers.AbsSentTokBatcher.make_batch(raw_feed={'query_texts': batch},
                                                                   pt_lm_tokenizer=self.tokenizer)
                ret_dict = self.model.encode(batch_dict=batch_dict)
                return ret_dict, ret_dict['sent_reps']
    
    
def get_wholeabs_sent_reps(doc_stream, model_name, trained_model_path, model_version='cur_best'):
    """
    Read the title abstract and get contextual sentence reps from a model. The model will clip] some
    abstracts because of a length restriction but this code will pad the reps with random uniform float
    vector.
    :param doc_stream: list(pid, abstract_dict)
    :param model_name: string; {'miswordbienc'}
    :param trained_model_path: string; run directory with saved model and hyperparams.
    :return: dict(pid: numpy.array(num_sents, encoding_dim)
    """
    encoding_dim = 768
    num_docs = len(doc_stream)
    if model_name in {'miswordbienc'}:
        trained_model = TrainedModel(model_name=model_name, trained_model_path=trained_model_path,
                                     model_version=model_version)
        batch_size = 32
    elif model_name in {'upnfconsent'}:
        trained_model = TrainedModel(model_name=model_name, trained_model_path=trained_model_path,
                                     model_version=model_version)
        batch_size = 32
    else:
        trained_model = TrainedModel(model_name=model_name, trained_model_path=trained_model_path,
                                     model_version=model_version)
        batch_size = 32
    start = time.time()
    print(f'Num docs: {num_docs}')
    
    # Write out sentence reps incrementally.
    pid2sentreps = collections.OrderedDict()
    batch_docs = []
    batch_pids = []
    for doci, (pid, abs_dict) in enumerate(doc_stream):
        if doci % 1000 == 0:
            print('Processing document: {:d}/{:d}'.format(doci, num_docs))
        batch_docs.append({'TITLE': abs_dict['title'],
                           'ABSTRACT': abs_dict['abstract']})
        batch_pids.append(pid)
        if len(batch_docs) == batch_size:
            # Returns a list of matrices with sentence reps.
            _, batch_doc_sentreps = trained_model.predict(batch_docs)
            assert(len(batch_pids) == len(batch_doc_sentreps))
            for bpid, bdoc, doc_sent_reps in zip(batch_pids, batch_docs, batch_doc_sentreps):
                assert(doc_sent_reps.shape[1] == encoding_dim)
                pid2sentreps[bpid] = doc_sent_reps
            batch_docs = []
            batch_pids = []
    # Handle left over documents.
    if len(batch_docs) > 0:
        _, batch_doc_sentreps = trained_model.predict(batch_docs)
        assert(len(batch_doc_sentreps) == len(batch_pids))
        assert(len(batch_pids) == len(batch_doc_sentreps))
        for bpid, bdoc, doc_sent_reps in zip(batch_pids, batch_docs, batch_doc_sentreps):
            assert(doc_sent_reps.shape[1] == encoding_dim)
            pid2sentreps[bpid] = doc_sent_reps
    del trained_model
    print('Took: {:.4f}s'.format(time.time() - start))
    return pid2sentreps

