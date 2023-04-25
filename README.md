### Editable User Profiles for Controllable Text Recommendations

Repository accompanying paper: 

**Title**: "Editable User Profiles for Controllable Text Recommendations"

**Authors**: Sheshera Mysore, Mahmood Jasim, Andrew McCallum, and Hamed Zamani at the University of Massachusetts Amherst, USA.

**Abstract**: Methods for making high-quality recommendations often rely on learning latent representations from interaction data. These methods, while performant, do not provide ready mechanisms for users to control the recommendation they receive. Our work tackles this problem by proposing LACE, a novel concept value bottleneck model for controllable text recommendations. LACE represents each user with a succinct set of human-readable concepts through retrieval given user-interacted documents and learns personalized representations of the concepts based on user documents. This concept based user profile is then leveraged to make recommendations. The design of our model affords control over the recommendations through a number of intuitive interactions with a transparent user profile. We first establish the quality of recommendations obtained from LACE in an offline evaluation on three recommendation tasks spanning six datasets in warm-start, cold-start, and zero-shot setups. Next, we validate the controllability of LACE under simulated user interactions. Finally, we implement LACE in an interactive controllable recommender system and conduct a user study to demonstrate that users are able to improve the quality of recommendations they receive through interactions with an editable user profile.

:page_facing_up: Arxiv pre-print: https://arxiv.org/abs/2304.04250

:memo: A twitter thread: https://twitter.com/msheshera/status/1645837725516300308?s=46 

:blue_book: Paper as published by ACM: TODO

## Repository contents

    ├── README.md
    ├── bin
    │   ├── evaluation
    │   ├── learning
    │   └── pre_process
    ├── config
    │   └── models_config
    ├── ipynb
    ├── scripts
    └── src
        ├── __init__.py
        ├── __pycache__
        ├── evaluation
        ├── learning
        ├── pre_process
        └── tests

- `src`: This contains all the python source files for pre-processing datasets, model and training, and evaluation. Here are the most important/less-decipherable pieces:

    - `src/pre_process`: Contains somewhat identical scripts for pre-processing different datasets, this is all the files with names `pre_proc_<dataset>.py`. The functions in the files contain code to: convert the originally distributed dataset into json files which my pipelines can work with, split the data into train/dev/test, code to convert the splits into pickle files which my modeling code loads into memory for training. The creation of splits for ItemKNN models, LACE models, and Hybrid models is slightly different so that is implemented here. The creation of splits for cold start and warm-start recommendation is also implemented here, the splits for cold-start and zero-shot are identical. There are many datasets here which were explored over various projects but only some of them made it into papers so you will find many older functions here.

        - `src/pre_process/pp_gen_nearest.py`: This is an important file, and contains the inference code for various models. The functions load the test data, load the trained/pre-trained models from disk, and run the model on every user in the test sets -- the models provide functions for inference. This file contains different classes for model types: ItemKNN and LACE models.

    - `src/learning`: This directory contains the code for implementing models (`learning/facetid_models`, `learning/models_common`), initializing and training the models (`main_fsim.py`, `main_mfbpr.py`, `main_recom.py`), boiler plate training loops (`trainer.py`), and batching code (`rec_batchers.py`, `batchers.py`).
        - `main_fsim.py`, `main_recom.py`: The two kinds of models implemented are ItemKNN models (which use doc-doc similarity) and LACE models (which use user docs-doc similarity) so the two files here initialize and train those models respectively. The ItemKNN models are a holdover from a older (more mature) project where the training supports distributed training over multiple GPUs and you will see this in `main_fsim.py`. The `main_recom.py` however does not currently implement multi GPU training though it contains some (likely broken) code for it.

        - `rec_batchers.py`: This is the batching code for LACE models. This code loads a pickle file of the whole dataset into memory so scaling it to very large datasets will need some changes. In a high level the batcher: chunks the training set so that similarly sized users are present in a single batch, creates "leave one out" training examples paired with a random negative for every user (i.e if a user has 10 docs then it yields 10 training samples), tokenizes text for LMs, pads the text, and feeds it to the model.

        - `batchers.py`: This is the batching code for the ItemKNN/doc-doc similarity models. Here the code is more scalable and streams jsonl examples from disk into the model.

        - `main_mfbpr.py`: This is a bit of an exception from the above where an external library, [implicit](https://github.com/benfred/implicit/), is used to train matrix factorization based recommender baselines. Here, the code to load data, train the model, and make predictions is present in a single script.

        - `facetid_models/contentcf_models.py`: Hybrid models which learn a fixed vector per user and use an LM for the item content.

        - `facetid_models/disent_models.py`: ItemKNN models. Learn a single vector per document or multiple vectors per document.

        - `facetid_models/editie_models.py`: The LACE models. There are also a number of different models which are variants of LACE which did not make it into any published work.

        - `facetid_models/pair_distances.py`: Implements optimal transport for document-concept assignment and profile-values to document distance computation. Also implements variants for this such as attention.


    - `src/evaluation/ranking_eval.py`: The functions to load predictions, gold data, and print metrics. The metrics are manually copied to dataset specific sheets in this google drive: [2021-edit_expertise](https://drive.google.com/drive/folders/1bOf6GUwqvrgOJailFjr3SpBQt9FYw9VG?usp=sharing). `rank_metrics.py` is a generic script implementing various ranking metrics. Ideally in future we should use [`trec_eval`](https://trec.nist.gov/trec_eval/).

- `config`: This contains `json` per dataset, per experiment configuration files for the models called from `main_fsim.py` and `main_recom.py`. Currently I do hyper-parameter tuning manually from these configs. The names kind of have a mnemonic for what the model represents but I don't expect it to be clear without documentation - ask me if you need to know! My protocol is that every model that gets run (trained or a zero-shot model) must have a config here. The exception to this are the models in `main_mfbpr.py`.

- `bin`: This contains bash script to launch various python scripts in the `src` directory. Many of these accept command line arguments which indicate the dataset, experiment name, experimental conditions and so on. Which are then passed to the python scripts. Here they all expect a environment variable called `CUR_PROJ_DIR` to be set, this is: `/work/smysore_umass_edu/2021-edit_expertise` -- this is where this project lives on Unity.


### TODOs

1. Release documented code for experiments.
2. Release processed training and evaluation data.
3. Release trained model parameters and instructions to use released models.
4. Release instructions to run training scripts.