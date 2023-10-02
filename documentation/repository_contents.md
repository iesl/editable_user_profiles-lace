## Repository contents

    ├── README.md
    ├── bin
    │   ├── evaluation
    │   ├── learning
    │   └── pre_process
    ├── config
    │   └── models_config
    ├── documentation
    │   └── repository_contents.md
    ├── requirements.txt
    ├── src
    │   ├── __init__.py
    │   ├── evaluation
    │   ├── learning
    │   └── pre_process
    └── user_study
        ├── apps
        ├── requirements.txt
        └── src

- `src`: This contains all the python source files for pre-processing datasets, model and training, and evaluation. Here are the most important pieces:

    - `src/pre_process`: Contains somewhat identical scripts for pre-processing different datasets, this is all the files with names `pre_proc_<dataset>.py`. The functions in the files contain code to: convert the originally distributed dataset into json files which my pipelines can work with, split the data into train/dev/test, code to convert the splits into pickle files which my modeling code loads into memory for training. The creation of splits for ItemKNN models, LACE models, and Hybrid models is slightly different so that is implemented here. The creation of splits for cold start and warm-start recommendation is also implemented here, the splits for cold-start and zero-shot are identical.

        - `src/pre_process/pp_gen_nearest.py`: This is an important file, and contains the inference code for various models. The functions load the test data, load the trained/pre-trained models from disk, and run the model on every user in the test sets -- the models provide functions for inference. This file contains different classes for model types: ItemKNN and LACE models.

    - `src/learning`: This directory contains the code for implementing models (`learning/retrieval_models`), initializing and training the models (`main_fsim.py`, `main_mfbpr.py`, `main_recom.py`), boiler plate training loops (`trainer.py`), and batching code (`rec_batchers.py`, `batchers.py`).
        - `main_fsim.py`, `main_recom.py`: The two kinds of models implemented are ItemKNN models (which use doc-doc similarity) and LACE models (which use user docs-doc similarity) so the two files here initialize and train those models respectively. The ItemKNN models are a holdover from a older (more mature) project where the training supports distributed training over multiple GPUs and you will see this in `main_fsim.py`. The `main_recom.py` however does not currently implement multi GPU training though it contains some (likely broken) code for it.

        - `rec_batchers.py`: This is the batching code for LACE models. This code loads a pickle file of the whole dataset into memory so scaling it to very large datasets will need some changes. In a high level: the batcher chunks the training set so that similarly sized users are present in a single batch, creates "leave one out" training examples paired with a random negative for every user (i.e if a user has 10 docs then it yields 10 training samples), tokenizes text for LMs, pads the text, and feeds it to the model.

        - `batchers.py`: This is the batching code for the ItemKNN/doc-doc similarity models. Here the code is more scalable and streams jsonl examples from disk into the model.

        - `main_mfbpr.py`: Here, [implicit](https://github.com/benfred/implicit/), is used to train matrix factorization based recommender baselines. Here, the code to load data, train the model, and make predictions is present in a single script.

        - `retrieval_models/contentcf_models.py`: Hybrid models which learn a fixed vector per user and use an LM for the item content.

        - `retrieval_models/disent_models.py`: ItemKNN models. Learn a single vector per document or multiple vectors per document.

        - `retrieval_models/editie_models.py`: The LACE models. There are also a number of different models which are variants of LACE which did not make it into any published work.

        - `retrieval_models/pair_distances.py`: Implements optimal transport for document-concept assignment and profile-values to document distance computation. Also implements variants for this such as attention.

    - `src/evaluation/ranking_eval.py`: The functions to load predictions, gold data, and print metrics. `rank_metrics.py` is a generic script implementing various ranking metrics. Ideally in future we should use [`trec_eval`](https://trec.nist.gov/trec_eval/).

- `config`: This contains `json` per dataset, per experiment configuration files for the models called from `main_fsim.py` and `main_recom.py`. The names kind of have a mnemonic for what the model represents but I don't expect it to be clear without documentation - ask me if you need to know! My protocol is that every model that gets run (trained or a zero-shot model) must have a config here. The exception to this are the models in `main_mfbpr.py`.

- `bin`: This contains bash script to launch various python scripts in the `src` directory. Many of these accept command line arguments which indicate the dataset, experiment name, experimental conditions and so on. Which are then passed to the python scripts. Here they all expect a environment variable called `CUR_PROJ_DIR` to be set to the current project directory.

- `user_study`: Contains code used for running the user study reported in the paper. `documentation/user_study_usage.md` contains some usage instructions.

    - `user_study/apps` contains code for the streamlit apps for ItemKNN and LACE, and their respective requirement files.

    - `user_study/src` contains code to pre-process candidate papers recommended to users and user seed papers from which their profiles are built. `user_study/src/pre_process` contains pre-processing code. `user_study/src/learning` is mostly copied from `src/learning` above with unused code deleted.