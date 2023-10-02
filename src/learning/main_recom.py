"""
For the editable recommendation models:
Call code from everywhere, read data, initialize model, train model and make
sure training is doing something meaningful.
"""
import argparse, os, sys
import codecs, pprint, json

import comet_ml as cml
import logging
import torch
from . import rec_batchers, trainer, data_utils
from .retrieval_models import editable_profile_models, contentcf_models


def train_model(model_name, data_path, config_path, run_path, cl_args):
    """
    Read the int training and dev data, initialize and train the model.
    :param model_name: string; says which model to use.
    :param data_path: string; path to the directory with unshuffled data
        and the test and dev json files.
    :param config_path: string; path to the directory json config for model
        and trainer.
    :param run_path: string; path for shuffled training data for run and
        to which results and model gets saved.
    :param cl_args: argparse command line object.
    :return: None.
    """
    run_name = os.path.basename(run_path)
    # Load label maps and configs.
    with codecs.open(config_path, 'r', 'utf-8') as fp:
        all_hparams = json.load(fp)
    
    cml_experiment = cml.Experiment(project_name='2021-edit-expertise', display_summary_level=0)
    cml_experiment.log_parameters(all_hparams)
    cml_experiment.set_name(run_name)
    # Save the name of the screen session the experiment is running in.
    cml_experiment.add_tags([cl_args.dataset, cl_args.model_name, os.environ['STY']])
    
    # Unpack hyperparameter settings.
    logging.info('All hyperparams:')
    logging.info(pprint.pformat(all_hparams))
    
    # Save hyperparams to disk.
    run_info = {'all_hparams': all_hparams}
    with codecs.open(os.path.join(run_path, 'run_info.json'), 'w', 'utf-8') as fp:
        json.dump(run_info, fp)
    
    # Initialize model.
    if model_name in {'upnfconsent'} and all_hparams.get('barycenter_projection', False):
        model = editable_profile_models.UPNamedFBaryCProj(model_hparams=all_hparams)
        # Save an untrained model version.
        trainer.generic_save_function(model=model, save_path=run_path, model_suffix='init')
    elif model_name in {'upsentconsent'}:
        model = editable_profile_models.UPSentAspire(model_hparams=all_hparams)
        # Save an untrained model version.
        trainer.generic_save_function(model=model, save_path=run_path, model_suffix='init')
    elif model_name in {'upnfkpenc'}:
        model = editable_profile_models.UPNamedFKPCandSent(model_hparams=all_hparams)
        # Save an untrained model version.
        trainer.generic_save_function(model=model, save_path=run_path, model_suffix='init')
    elif model_name in {'contentcf'}:
        warm_start = all_hparams.get('warm_start', False)  # Default is cold-start.
        if warm_start:
            cold_warm_start = 'warm_start'
        else:
            simpair = all_hparams.get('simpair', False)  # For the editability eval.
            if simpair:
                cold_warm_start = 'simpair'
            else:
                cold_warm_start = 'cold_start'
        with open(os.path.join(data_path, cold_warm_start, 'uid2idx.json'), 'r') as fp:
            train_uid2idx = json.load(fp)
        # Save to run directory so inference code can access it.
        with codecs.open(os.path.join(run_path, 'uid2idx.json'), 'w') as fp:
            json.dump(train_uid2idx, fp)
            logging.info(f'Wrote: {fp.name}')
        contentcf_models.ContentCF.num_users = len(train_uid2idx)
        model = contentcf_models.ContentCF(model_hparams=all_hparams)
    else:
        logging.error('Unknown model: {:s}'.format(model_name))
        sys.exit(1)
    logging.info(f"GPU ids: {os.environ['CUDA_VISIBLE_DEVICES']}")
    # Only save an initialized model and exit if asked.
    only_init = all_hparams.get('only_init', False)
    if only_init:
        sys.exit()
        
    # Model class internal logic uses the names at times so set this here so it
    # is backward compatible.
    model.model_name = model_name
    logging.info(model)
    
    # Move model to the GPU.
    if torch.cuda.is_available():
        model.cuda()
        logging.info('Running on GPU.')
    
    # Initialize the trainer.
    ict_lossprop = all_hparams.get('ict_lossprop', 0)
    if model_name in ['upnfconsent', 'upsentconsent', 'upnfkpenc']:
        batcher_cls = rec_batchers.UserCandKPBatcher
        # todo: figure out how to get these automatically from the fine tuned model dir.
        if 'consent-base-pt-layer' in all_hparams and 's2orccompsci' in all_hparams['consent-base-pt-layer']:
            batcher_cls.context_bert_config_str = "allenai/specter"
            batcher_cls.kp_bert_config_str = "allenai/scibert_scivocab_uncased"
        else:  # this is in the case of tedrec more likely, but not exclusively.
            batcher_cls.context_bert_config_str = all_hparams['consent-base-pt-layer']
            batcher_cls.kp_bert_config_str = all_hparams['kp-base-pt-layer']
        # If using ict losses then return redundant kps per sentence.
        if ict_lossprop > 0:
            batcher_cls.return_uniq_kps = False
    elif model_name in ['contentcf']:
        batcher_cls = rec_batchers.UserIDCandBatcher
        if cl_args.dataset in {'citeulikea', 'citeuliket', 'oriclr2019', 'oriclr2020', 'oruai2019'}:
            batcher_cls.context_bert_config_str = "allenai/specter"
        elif cl_args.dataset in {'tedrec'}:
            batcher_cls.context_bert_config_str = 'sentence-transformers/all-mpnet-base-v2'
        batcher_cls.user_ids = train_uid2idx
    else:
        logging.error('Unknown model: {:s}'.format(model_name))
        sys.exit(1)
    
    if model_name in ['upnfconsent', 'upsentconsent', 'upnfkpenc']:
        if ict_lossprop > 0:
            model_trainer = trainer.InMemBasicRankingTrainerSepLosses(cml_exp=cml_experiment,
                                                                      model=model, batcher=batcher_cls,
                                                                      data_path=data_path,
                                                                      model_path=run_path,
                                                                      early_stop=True, dev_score='loss',
                                                                      train_hparams=all_hparams)
            model_trainer.save_function = trainer.generic_save_function
        else:
            model_trainer = trainer.InMemBasicRankingTrainer(cml_exp=cml_experiment,
                                                             model=model, batcher=batcher_cls,
                                                             data_path=data_path,
                                                             model_path=run_path,
                                                             early_stop=True, dev_score='loss',
                                                             train_hparams=all_hparams,
                                                             epoch_save=True)
            model_trainer.save_function = trainer.generic_save_function
    # Train and save the best model to model_path.
    model_trainer.train()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subcommand',
                                       help='The action to perform.')
    # Train the model.
    train_args = subparsers.add_parser('train_model')
    # Where to get what.
    train_args.add_argument('--model_name', required=True,
                            choices=['upnfconsent', 'upsentconsent', 'upnfkpenc', 'contentcf'],
                            help='The name of the model to train.')
    train_args.add_argument('--dataset', required=True,
                            choices=['citeulikea', 'citeuliket', 'tedrec', 'oriclr2019', 'oriclr2020', 'oruai2019'],
                            help='The dataset to train and predict on.')
    train_args.add_argument('--num_gpus', required=True, type=int,
                            help='Number of GPUs to train on/number of processes running parallel training.')
    train_args.add_argument('--data_path', required=True,
                            help='Path to the jsonl dataset.')
    train_args.add_argument('--run_path', required=True,
                            help='Path to directory to save all run items to.')
    train_args.add_argument('--config_path', required=True,
                            help='Path to directory json config file for model.')
    cl_args = parser.parse_args()
    # If a log file was passed then write to it.

    try:
        logging.basicConfig(level='INFO', format='%(message)s',
                            filename=cl_args.log_fname)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))
    # Else just write to stdout.
    except AttributeError:
        logging.basicConfig(level='INFO', format='%(message)s',
                            stream=sys.stdout)
        # Print the called script and its args to the log.
        logging.info(' '.join(sys.argv))

    if cl_args.subcommand == 'train_model':
        if cl_args.num_gpus > 1:
            raise NotImplementedError
        else:
            train_model(model_name=cl_args.model_name, data_path=cl_args.data_path,
                        run_path=cl_args.run_path, config_path=cl_args.config_path, cl_args=cl_args)


if __name__ == '__main__':
    main()
