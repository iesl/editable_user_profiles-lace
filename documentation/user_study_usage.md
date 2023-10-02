To conduct the user study a set of 100k candidate documents are sampled and processed in `user_study/src/pre_process/pre_proc_candidates.py`.

To pre-process the documents for a participant:

1. Download the participants seed papers from S2 either through references of submitted papers or submitted author ids of interest. This script expects participant directories to be created and contain a text file with the participant submitted paper links:
```commandline
# If participant submits papers of interest.
./src/pre_process/get_seed_papers.sh -a download_seed_paper_refs -u <participant_name>
# If participant submits authors of interest.
./src/pre_process/get_seed_papers.sh -a download_author_seed_papers -u <partipant_name> -i <integer author id1> -j <integer author id2>
```

2. Embed the papers.

```commandline
./src/pre_process/run_pp_seedset.sh -a user_preproc -u <participant_name>
```
