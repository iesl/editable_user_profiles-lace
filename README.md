### Editable User Profiles for Controllable Text Recommendations

Repository accompanying paper: 

**Title**: "Editable User Profiles for Controllable Text Recommendations"

**Authors**: Sheshera Mysore, Mahmood Jasim, Andrew McCallum, and Hamed Zamani at the University of Massachusetts Amherst, USA.

**Abstract**: Methods for making high-quality recommendations often rely on learning latent representations from interaction data. These methods, while performant, do not provide ready mechanisms for users to control the recommendation they receive. Our work tackles this problem by proposing LACE, a novel concept value bottleneck model for controllable text recommendations. LACE represents each user with a succinct set of human-readable concepts through retrieval given user-interacted documents and learns personalized representations of the concepts based on user documents. This concept based user profile is then leveraged to make recommendations. The design of our model affords control over the recommendations through a number of intuitive interactions with a transparent user profile. We first establish the quality of recommendations obtained from LACE in an offline evaluation on three recommendation tasks spanning six datasets in warm-start, cold-start, and zero-shot setups. Next, we validate the controllability of LACE under simulated user interactions. Finally, we implement LACE in an interactive controllable recommender system and conduct a user study to demonstrate that users are able to improve the quality of recommendations they receive through interactions with an editable user profile.

:page_facing_up: Arxiv pre-print: https://arxiv.org/abs/2304.04250

:memo: A twitter thread: https://twitter.com/msheshera/status/1645837725516300308?s=46 

:blue_book: Paper as published by ACM: https://dl.acm.org/doi/10.1145/3539618.3591677

## Repository contents

    ├── README.md
    ├── bin
    ├── config
    ├── data
    ├── documentation
    ├── requirements.txt
    ├── src
    └── user_study

- `bin`: This contains bash script to launch various python scripts in the `src` directory.

- `src`: This contains all the python source files for pre-processing datasets, model and training, and evaluation reported in experiments for the paper.

- `config`: This contains `json` configs containing hyperparameters for models per dataset, evaluation setup, and model for experiments reported in the paper.

- `data`: Contains concepts extracted with [Forecite](https://dl.acm.org/doi/10.1145/3397271.3401235) that are used in LACE. The directory also contains a pre-processed copy of an openly available reviewer-paper matching [dataset](https://github.com/niharshah/goldstandard-reviewer-paper-match) as demo data for showing how to run LACE.

- `documentation`: More documentation + usage instructions for demo data and models hosted on Huggingface.

- `user_study`: Contains streamlit app code for the LACE and ItemKNN models run in the user_study reported in the paper as well as pre-processing for the (non-personal) data used in the experiments.

### Citation

Please cite the LACE paper as:  

```bibtex
@inproceedings{mysore2023lace,
    author = {Mysore, Sheshera and Jasim, Mahmood and Mccallum, Andrew and Zamani, Hamed},
    title = {Editable User Profiles for Controllable Text Recommendations},
    year = {2023},
    isbn = {9781450394086},
    publisher = {Association for Computing Machinery},
    address = {New York, NY, USA},
    url = {https://doi.org/10.1145/3539618.3591677},
    doi = {10.1145/3539618.3591677},
    booktitle = {Proceedings of the 46th International ACM SIGIR Conference on Research and Development in Information Retrieval},
    pages = {993–1003},
    numpages = {11},
    keywords = {concept bottleneck models, pre-trained language models, interactive recommendation systems, text recommendations},
    location = {Taipei, Taiwan},
    series = {SIGIR '23}
}
```
