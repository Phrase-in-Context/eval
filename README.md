<div id="top"></div>

<!--
*** Thanks for checking out the Best-README-Template. If you have a suggestion
*** that would make this better, please fork the repo and create a pull request
*** or simply open an issue with the tag "enhancement".
*** Don't forget to give the project a star!
*** Thanks again! Now go create something AMAZING! :D
-->



<!-- PROJECT SHIELDS -->
<!--
*** I'm using markdown "reference style" links for readability.
*** Reference links are enclosed in brackets [ ] instead of parentheses ( ).
*** See the bottom of this document for the declaration of the reference variables
*** for contributors-url, forks-url, etc. This is an optional, concise syntax you may use.
*** https://www.markdownguide.org/basic-syntax/#reference-style-links
-->

<!--
[![Contributors][contributors-shield]][contributors-url]
[![Forks][forks-shield]][forks-url]
[![Stargazers][stars-shield]][stars-url]
[![Issues][issues-shield]][issues-url]
[![MIT License][license-shield]][license-url]
[![LinkedIn][linkedin-shield]][linkedin-url]
-->


<!-- PROJECT LOGO -->
<br />

<div align="center">
  <!--
  <a href="https://github.com/anguyen8/im">
    <img src="images/logo.png" alt="Logo" width="80" height="80">
  </a>
  -->

  <h1 align="center">PiC: A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search</h1>
  <p align="center">
    Joint work between Adobe Research and Auburn University
  </p>
  
  [Thang Pham](https://twitter.com/pmthangxai), [Seunghyun Yoon](https://david-yoon.github.io), [Trung Bui](https://research.adobe.com/person/trung-bui/), and [Anh Nguyen](https://anhnguyen.me).
</div>

<!-- TABLE OF CONTENTS -->

<details open>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    </li>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#reproduce-results-for-benchmark">Reproduce results for benchmark</a></li>
    <li><a href="#license">License</a></li>
    <li><a href="#contact">Contact</a></li>
    <li><a href="#references">References</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->

## About The Project

Phrase in Context is a curated benchmark for phrase understanding and semantic search, consisting of three tasks of increasing difficulty: Phrase Similarity (PS), Phrase Retrieval (PR) and Phrase Sense Disambiguation (PSD). The datasets are annotated by 13 linguistic experts on Upwork and verified by two groups: ~1000 AMT crowdworkers and another set of 5 linguistic experts. PiC benchmark is distributed under [CC-BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/).

[![PiC example][pic-sample]](https://github.com/Phrase-in-Context/eval/)

:star2: Official implementation to reproduce most of the main results in our paper [PiC: A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search](https://arxiv.org/abs/2207.09068).

:sun_with_face: **Project Link**: [https://phrase-in-context.github.io/](https://phrase-in-context.github.io/)

:fire: **Online Web Demo**: https://aub.ie/phrase-search

**If you use our PiC dataset or software, please consider citing:**

    @article{pham2022PiC,
      title={PiC: A Phrase-in-Context Dataset for Phrase Understanding and Semantic Search},
      author={Pham, Thang M and Yoon, Seunghyun and Bui, Trung and Nguyen, Anh},
      journal={arXiv preprint arXiv:2207.09068},
      year={2022}
    }

<!-- GETTING STARTED -->

## Getting Started

### Prerequisites

* Anaconda 4.10 or higher
* Python 3.9 or higher
* pip version 21 or higher

### Installation

1. Create a new folder and clone the repo

   ```sh
   mkdir phrase-in-context && cd "$_"
   git clone https://github.com/Phrase-in-Context/eval.git && cd eval
   ```

2. Create and activate a Conda environment

   ```sh
   conda create -n pic_eval python=3.9
   conda activate pic_eval
   ```

3. Install required libraries

   ```sh
   pip install -r requirements.txt
   bash extra_requirements.sh
   ```

<!-- USAGE EXAMPLES -->

## Reproduce results for benchmark

### 1. PS: Phrase Similarity

Change directory to the `similarity` folder
```
cd similarity/
```

#### Approach 1: Cosine Similarity

```
bash run_eval_ranking.sh
```

#### Approach 2: BERT-based classifiers

```
bash run_eval_cls.sh
```

* Please note that the default setting for both approaches is **non-contextualized** phrase embeddings. For the **contextualized** setting, we need to uncomment the argumnet `--contextual` in the script.
* The results are stored under the folder `../results/phrase_similarity/ranking` for the first approach and `../results/phrase_similarity/classification` for the second approach.

### 2. PR: Phrase Retrieval

#### Approach 1: Similarity-based Ranking

Change directory to the `retrieval_ranking` folder
```
cd retrieval_ranking/
```

For this approach, we do not train the models. Instead, we evaluate them by comparing the phrase representations of a query with that of phrase candidates encoded by the models' encoders.
```
export DATASET="phrase_retrieval"
export DATASET_CONFIG="PR-pass"
export MODEL="BERT-base"
export CONTEXTUAL=False     # Set it to True for contextualized setting

bash run_eval.sh evaluate_model "${DATASET}" "${DATASET_CONFIG}" "${MODEL}" "${CONTEXTUAL}"
```

* Note that the default setting is **non-contextualized** phrase embeddings. For the **contextualized** setting, we need to change the export value of `CONTEXTUAL` to **True**.
* `DATASET_CONFIG` can be exported to the followings: `PR-pass` or `PR-page`
* `MODEL` can be exported to the followings: : `BERT-base`, `BERT-large`, `PhraseBERT`, `SpanBERT`, `SentenceBERT`, `SimCSE`, and `USE`.
* The **results** and **log file** are stored under the folder `../results/phrase_retrieval/${DATASET_CONFIG}/ranking`

#### Approach 2: Question Answering retrieval system 

Change directory to the `retrieval_qa` folder
```
cd retrieval_qa/
```

First, we train a Q/A model (e.g., BERT-base) on one of two versions of PR dataset: PR-pass or PR-page
```
export DATASET="phrase_retrieval"
export DATASET_CONFIG="PR-pass"
export MODEL="BERT-base"

bash train_qa.sh finetune_model "${DATASET}" "${DATASET_CONFIG}" "${MODEL}"
```

Then, we evaluate the newly trained Q/A model as follows
```
bash eval_qa.sh evaluate_model "${DATASET}" "${DATASET_CONFIG}" "${MODEL}"
```

* `DATASET_CONFIG` can be exported to the followings: `PR-pass` or `PR-page`
* `MODEL` can be exported to the followings: : `BERT-base`, `BERT-large`, `PhraseBERT`, `SpanBERT`, `SentenceBERT`, `SimCSE`, `Longformer-base` and `Longformer-large`.
* The **results** and **log file** of training and evaluation are stored under the folder `../results/phrase_retrieval/${DATASET_CONFIG}/qa`

### 3. PSD: Phrase Sense Disambiguation

#### Approach 1: Question Answering retrieval system 

Training and evaluating a Q/A model on PSD dataset is quite similar to PR's approach 2. All we need is to update the dataset and its config as follows and keep the rest part unchanged.
```
export DATASET="phrase_sense_disambiguation"
export DATASET_CONFIG=""
export MODEL="BERT-base"

# For training
bash train_qa.sh finetune_model "${DATASET}" "${DATASET_CONFIG}" "${MODEL}"

# For evaluation
bash eval_qa.sh evaluate_model "${DATASET}" "${DATASET_CONFIG}" "${MODEL}"
```
* `DATASET_CONFIG` can only be exported to an empty string `""` since PSD has only one version. Regarding `MODEL`, you can follow the provided list in the PR section to train and evaluate other models.
* The **results** and **log file** of training and evaluation are stored under the folder `../results/phrase_sense_disambiguation/qa`

<!-- ### 2. Evaluate your own models -->



<!--

- [] Analysis of attribution maps
  - [] Out-of-distribution issue (Sec. 5.1)
  - [] BERT often replaces a word by itself (Sec. 5.2)
  - [] Attribution magnitude (Sec. 5.2)
    -->

See the [open issues](https://github.com/Phrase-in-Context/eval/issues) for a full list of proposed features (and
known issues).


<!-- CONTRIBUTING -->

<!--

## Contributing

Contributions are what make the open source community such an amazing place to learn, inspire, and create. Any
contributions you make are **greatly appreciated**.

If you have a suggestion that would make this better, please fork the repo and create a pull request. You can also
simply open an issue with the tag "enhancement". Don't forget to give the project a star! Thanks again!

1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request
   -->

<!-- LICENSE -->

## License

Distributed under the MIT License.


<!-- CONTACT -->

## Contact

The entire code was done and maintained by Thang Pham, [@pmthangxai](https://twitter.com/pmthangxai) - tmp0038@auburn.edu.
Contact us via email or create github issues if you have any questions/requests. Thanks!


<!-- ACKNOWLEDGMENTS -->

## References

* Huggingface. 2022. transformers/examples/pytorch/question-answering at main · huggingface/transformers. [https://github.com/huggingface/transformers/tree/main/](https://github.com/huggingface/transformers/tree/main/) examples/pytorch/question-answering.
* Shufan Wang, Laure Thompson, and Mohit Iyyer. 2021. [Phrase-BERT](https://github.com/sf-wa-326/phrase-bert-topic-model): Improved Phrase Embeddings from BERT with an Application to Corpus Exploration. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 10837–10851, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
* Jinhyuk Lee, Mujeen Sung, Jaewoo Kang, and Danqi Chen. 2021. Learning Dense Representations of Phrases at Scale. In Association for Computational Linguistics (ACL) [DensePhrase](https://github.com/princeton-nlp/DensePhrases)
* Tianyu Gao, Xingcheng Yao, and Danqi Chen. 2021. [SimCSE](https://github.com/princeton-nlp/SimCSE): Simple Contrastive Learning of Sentence Embeddings. In Proceedings of the 2021 Conference on Empirical Methods in Natural Language Processing, pages 6894–6910, Online and Punta Cana, Dominican Republic. Association for Computational Linguistics.
* Iz Beltagy, Matthew E Peters, and Arman Cohan. 2020. [Longformer](https://arxiv.org/abs/2004.05150): The long-document transformer.

<p align="right">&#40;<a href="#top">back to top</a>&#41;</p>

<!-- MARKDOWN LINKS & IMAGES -->
<!-- https://www.markdownguide.org/basic-syntax/#reference-style-links -->

[contributors-shield]: https://img.shields.io/github/contributors/Phrase-in-Context/eval.svg?style=for-the-badge
[contributors-url]: https://github.com/Phrase-in-Context/eval/graphs/contributors
[forks-shield]: https://img.shields.io/github/forks/Phrase-in-Context/eval.svg?style=for-the-badge
[forks-url]: https://github.com/Phrase-in-Context/eval/network/members
[stars-shield]: https://img.shields.io/github/stars/Phrase-in-Context/eval.svg?style=for-the-badge
[stars-url]: https://github.com/Phrase-in-Context/eval/stargazers
[issues-shield]: https://img.shields.io/github/issues/Phrase-in-Context/eval.svg?style=for-the-badge
[issues-url]: https://github.com/Phrase-in-Context/eval/issues
[license-shield]: https://img.shields.io/github/license/Phrase-in-Context/eval.svg?style=for-the-badge
[license-url]: https://github.com/Phrase-in-Context/eval/blob/master/LICENSE.txt
[linkedin-shield]: https://img.shields.io/badge/-LinkedIn-black.svg?style=for-the-badge&logo=linkedin&colorB=555
[linkedin-url]: https://linkedin.com/in/thangpm
[pic-sample]: images/pic_construction.png
