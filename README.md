# thaiperformancearts_RS
# Semantic Artifacts and Eligibility Gating for Hybrid Recommendation in Thai Performing Arts

## Repository purpose

This repository contains the research code and selected input artifacts for the paper titled **"Semantic Artifacts and Eligibility Gating for Hybrid Recommendation in Thai Performing Arts"**.

The repository implements an experimental pipeline for context-constrained recommendation in Thai performing arts. The main objective is to separate **cultural eligibility** from **preference-based ranking**. This separation ensures that recommendation models rank only items that are valid for the requested occasion or usage context.

This repository is a reproducible research artifact. It is not a deployed recommendation application. The code supports data preparation, semantic artifact construction, keyword-to-item grounding, eligibility-gated candidate construction, content-based filtering, collaborative filtering, hybrid ranking, and result aggregation for paper reporting.

---

## Relationship to the previous study

This repository extends a previous Thai performing arts text-clustering study titled **"Enhancing Thai Text Clustering for Performing Arts Through Domain-Specific NLP and Clustering Algorithms"**.

The previous study focused on building a domain-specific Thai NLP pipeline for Thai performing arts vocabulary. It included word segmentation, part-of-speech tagging, stopword removal, noun extraction, text embedding, dimensionality reduction, and clustering. The study compared embedding models such as FastText, BERT, WangchanBERTa, and BGE-M3, as well as clustering algorithms such as k-means, DBSCAN, hierarchical clustering, spectral clustering, and Gaussian mixture models. The best reported clustering configuration used k-means with UMAP and produced semantically coherent clusters for Thai performing arts vocabulary.

The current repository uses the output of that prior study as an upstream lexical input. In particular, `cluster_results.csv` represents the clustered vocabulary produced by the previous work. The current repository does not treat `cluster_results.csv` as raw data. Instead, it treats this file as a prior research artifact and uses it as the starting point for the semantic artifact construction layer.

The relationship between the two studies can be summarized as follows.

```text
Previous study
Thai performing arts corpus
→ domain-specific Thai text preprocessing
→ noun extraction
→ embedding and dimensionality reduction
→ clustering
→ cluster_results.csv

Current study
cluster_results.csv
→ LLM-based semantic stopword filtering
→ LLM-based taxonomy induction
→ keyword-to-item grounding
→ eligibility-gated recommendation evaluation
→ hybrid recommendation results
```

Therefore, the current work builds on the previous clustering study by transforming clustered Thai performing arts vocabulary into reusable, item-linked semantic artifacts and then using those artifacts in an eligibility-gated hybrid recommendation experiment.

---

## Research problem

Thai performing arts recommendation differs from ordinary entertainment recommendation because a performance may be suitable for one occasion but unsuitable for another. Ceremonies, festivals, rituals, institutional events, and social occasions impose cultural constraints that should be checked before ranking.

This project addresses this issue through a two-layer design.

**Layer A: Semantic Artifact Construction Layer**

This layer transforms Thai performing arts vocabulary into reusable semantic evidence. It includes preservation-oriented semantic stopword filtering, hierarchical taxonomy construction, and keyword-to-item grounding.

**Layer B: Eligibility-Gated Recommendation Evaluation Layer**

This layer evaluates recommendation models inside a context-valid candidate space. The eligibility gate first restricts candidate items by occasion context. Then popularity-based, content-based, collaborative filtering, and hybrid models rank only the context-valid items. This design allows recommendation performance to reflect ranking quality rather than accidental exposure to culturally invalid candidates.

---

## Repository structure

The current codebase is organized as follows.

```text
.
|-- config.py
|   Shared configuration file for the recommendation pipeline.
|   It defines file paths, hyperparameters, random seeds, rating thresholds,
|   text cleaning utilities, metric functions, score normalization,
|   negative-penalty handling, and reproducibility utilities.
|
|-- stopword_filter.py
|   LLM-based semantic stopword filtering script.
|   It analyzes clustered Thai performing arts vocabulary and identifies
|   low-semantic-value terms using multiple LLMs, including Gemini, GPT,
|   Qwen, and DeepSeek. It supports batch processing, retry handling,
|   run logs, partial outputs, and generation of stopword and non-stopword
|   lists. The non-stopword output becomes an input to the semantic artifact
|   construction layer.
|
|-- taxonomy.py
|   Semantic taxonomy construction script.
|   It implements a primary-critic-architect workflow. Gemini is used as
|   the primary model, GPT, Qwen, and DeepSeek act as critic models, and
|   Claude acts as the final architect model. The output is a hierarchical
|   semantic taxonomy for retained Thai performing arts vocabulary.
|
|-- keywordmapping.py
|   Keyword-to-item grounding script for Thai performing arts.
|   It maps classified keywords to performance items using exact matching,
|   fuzzy token-set matching, partial matching, optional semantic matching,
|   threshold sweeps, ablation analysis, and evaluation against a gold standard.
|
|-- data_loader.py
|   Data loading and preparation module.
|   It loads item metadata, user interaction logs, keyword mappings,
|   and taxonomy labels. It also builds reproducible train, validation,
|   and test splits for recommendation evaluation.
|
|-- eligibility_gate.py
|   Eligibility-gated candidate construction module.
|   It matches the requested context to item metadata, builds a context-valid
|   candidate pool, prioritizes keyword-hit items inside that pool, and applies
|   candidate-size control without adding context-invalid items.
|
|-- cbf.py
|   Content-based filtering module.
|   It encodes item text and query evidence using sentence embedding models
|   and scores candidate items using cosine similarity with a validation-tuned
|   keyword-hit boost.
|
|-- cf.py
|   Collaborative filtering and popularity baseline module.
|   It implements EASE_R, BiasedMF, BiasedMF-BPR, ItemKNN, SimpleX,
|   POP-Global, and POP-Context.
|
|-- hybrid.py
|   Hybrid fusion module.
|   It implements weighted-sum fusion, weighted-product fusion,
|   reciprocal rank fusion, and reliability-gated fusion between
|   content-based and collaborative filtering scores.
|
|-- run_pipeline.py
|   Main entry point for the recommendation evaluation pipeline.
|   It orchestrates data preparation, content-based filtering,
|   collaborative filtering, hybrid fusion, result aggregation,
|   paper-ready table generation, contamination checks, and supplementary
|   analyses.
|
|-- requirements.txt
|   Minimal Python package list for the core recommendation pipeline.
|
|-- prior_study.pdf
|   Documentation of the previous Thai performing arts text-clustering study.
|   This file explains the upstream study that produced the clustered
|   vocabulary used by the current repository.
|
|-- cluster_results.csv
|   Upstream clustered vocabulary artifact from the previous study.
|   This file is used as the starting lexical input for LLM-based semantic
|   stopword filtering in the current study.
|
|-- input/
|   Expected folder for CSV input files used by the executable pipeline.
|
|-- output/
|   Default folder for recommendation experiment outputs.
|
|-- result/
|   Default folder for semantic artifact outputs from stopword filtering,
|   keyword mapping, and taxonomy construction scripts.
