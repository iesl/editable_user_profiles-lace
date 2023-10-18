## Usage for reviewer-paper matching data

These are instructions to run LACE inference on an openly available reviewer-paper matching [dataset](https://github.com/niharshah/goldstandard-reviewer-paper-match). The code refers to this dataset as `cmugoldrpm` and the paper (Appendix A) refers to it as RAPRatings.


1. Clone this repository.
```bash
git clone git@github.com:iesl/editable_user_profiles-lace.git
cd editable_user_profiles-lace
```

2. Install the requirements.txt (I havent tested this code starting from a fresh install, so getting things to run may need some hacking)

3. Install the scispacy model for sentence segmentation:
```bash
pip3 install https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.0/en_core_sci_sm-0.5.0.tar.gz
```

4. Pre-process the data for consumption by LACE models - this converts datasets into a json format and pre-fetches keyphrases to the documents in the dataset. The `./data` directory in the repository contains a pre-processed version of `cmugoldrpm`. `src/pre_process/pre_proc_cmugoldrpm.py` contains code on how `cmugoldrpm` was processed and other datasets should be formatted similarly. The necessary files for inference are: `abstracts-cmugoldrpm-forecite-tfidfcsrr.jsonl` and `test-uid2anns-cmugoldrpm.json`. The first contains abstracts split into sentences and with keyphrases pre-fetched for all the dataset documents and the second contains candidate documents per user and user profile documents.

5. Assuming that the data has been processed for LACE run the below command for running inference with LACE. The function initializes LACE models (document and concept encoders) from a huggingface repo, reads the processed data, and produces scores for users and candidate items. It writes a readable output and a machine readable json file with the output.
```python
python3 -um src.pre_process.pp_gen_nearest rank_pool --root_path data/cmugoldrpm --dataset cmugoldrpm --rep_type upnfconsent --config_path config/models_config/cmugoldrpm/upnfconsent-rrtkp-baryc-init.json --caching_scorer
```
