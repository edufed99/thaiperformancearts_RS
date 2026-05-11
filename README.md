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
```

---

## Expected input files

The recommendation pipeline expects the following files under the `input` directory.

### Required input files for the recommendation pipeline

1. `all_item_130868.csv`  
   Item catalog for Thai performing arts. The file should contain item names, item descriptions, main contexts, and sub-contexts. Duplicate item rows are aggregated by performance name during loading.

2. `user_log_with_keywords_only_list.csv`  
   User interaction log. The file should contain user IDs, item names, ratings, usage contexts, and keyword lists. Ratings use a five-point scale. Ratings of 4 or higher are treated as positive interactions.

3. `mapped_words_to_items95_default.csv`  
   Keyword-to-item mapping output. This file links retained Thai performing arts keywords to item names and is used as semantic evidence during candidate prioritization and content-based ranking.

4. `consensus_classification.csv`  
   Taxonomy classification output. This file stores hierarchical semantic labels for retained vocabulary and supports the semantic artifact layer.

### Input files for semantic artifact construction

1. `cluster_results.csv`  
   Clustered vocabulary artifact from the previous study. This file is the starting point for LLM-based semantic stopword filtering.

2. `semantic_artifact_master.csv`  
   Retained vocabulary or classified keyword file used by `keywordmapping.py`.

3. `non_stopwords_gemini.csv`  
   Gemini-based retained vocabulary used as the primary lexical input for `taxonomy.py`.

4. `gold_keywords_114.csv`  
   Expert or manually annotated gold standard for keyword-to-item mapping evaluation.

5. `prior_study.pdf`  
   Supporting document that describes the upstream clustering study. This file is included for traceability and research context. It is not required for executing the Python pipeline.

---

## Installation

Create a Python environment and install the required packages.

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The minimal requirements file contains:

```text
numpy
pandas
torch
scipy
tqdm
```

Some semantic artifact scripts require additional optional packages depending on the selected mode. Install these packages when running taxonomy construction, semantic matching, Thai tokenization, or visualization functions.

```bash
pip install pythainlp rapidfuzz sentence-transformers transformers matplotlib requests python-dotenv
```

---

## API key setup for LLM-based semantic artifact construction

`stopword_filter.py` and `taxonomy.py` use OpenRouter-compatible model calls. If these scripts are executed, create a `.env` file in the same directory as the scripts and define the required API keys.

```text
OPENROUTER_API_KEY_GEMINI=your_key_here
OPENROUTER_API_KEY_GPT=your_key_here
OPENROUTER_API_KEY_QWEN=your_key_here
OPENROUTER_API_KEY_DEEPSEEK=your_key_here
OPENROUTER_API_KEY_CLAUDE=your_key_here
```

Do not commit real API keys to GitHub.

---

## How to run the semantic artifact construction layer

### 1. Run LLM-based semantic stopword filtering

Use `stopword_filter.py` to identify low-semantic-value vocabulary from the clustered vocabulary produced by the previous study.

```bash
python stopword_filter.py --model gemini
```

To run all supported LLMs:

```bash
python stopword_filter.py --model all
```

Typical outputs include stopword analysis files, non-stopword files, run logs, partial outputs, and metadata files under the `result` or `result_eval` directories. The retained non-stopword vocabulary becomes the main input for taxonomy construction and keyword-to-item grounding.

### 2. Run taxonomy construction

Use `taxonomy.py` to construct and evaluate a hierarchical semantic taxonomy from the retained vocabulary.

```bash
python taxonomy.py
```

To resume an existing run:

```bash
python taxonomy.py --resume RUN_ID
```

Typical outputs are written under `result/run_RUN_ID/final` and may include taxonomy consensus files, classification outputs, agreement tables, model comparison tables, semantic evidence tables, and paper-ready summaries.

### 3. Run keyword-to-item grounding

Use `keywordmapping.py` when the keyword-to-item mapping file has not yet been created or when threshold selection needs to be reproduced.

```bash
python keywordmapping.py --threshold 95 --evaluate
```

Useful alternatives include:

```bash
python keywordmapping.py --sweep
python keywordmapping.py --ablation-eval
python keywordmapping.py --train-val-test
python keywordmapping.py --cross-val
```

The script writes mapping and evaluation outputs under the `result` directory. Typical outputs include `mapped_words_to_items95.csv`, `unmapped_words95.csv`, `unmapped_items95.csv`, `sweep_table.csv`, `evaluation_summary.csv`, ablation tables, and error reports.

---

## How to run the recommendation evaluation pipeline

### 1. Prepare input files

Place the required CSV files in the `input` directory.

```text
input/all_item_130868.csv
input/user_log_with_keywords_only_list.csv
input/mapped_words_to_items95_default.csv
input/consensus_classification.csv
```

### 2. Run the full recommendation evaluation pipeline

```bash
python run_pipeline.py
```

This command runs the full eligibility-gated recommendation evaluation pipeline. It loads data, builds train-validation-test splits, constructs context-valid candidates, evaluates content-based filtering, evaluates collaborative filtering, selects validation-tuned models, runs hybrid fusion, aggregates results, and exports paper-ready tables.

### 3. Run selected phases only

```bash
python run_pipeline.py --phase cbf
python run_pipeline.py --phase cf
python run_pipeline.py --phase hybrid
```

### 4. Run selected seeds or candidate sizes

```bash
python run_pipeline.py --seeds 42 123
python run_pipeline.py --max-cands 20 40 80
```

### 5. Run selected CBF or CF models

```bash
python run_pipeline.py --cbf-models bge-m3
python run_pipeline.py --cf-models EASE_R BiasedMF-BPR ItemKNN
```

---

## Core experimental protocol

The main recommendation evaluation follows these settings.

1. Ratings use a five-point scale.
2. Ratings of 4 or higher are treated as positive interactions.
3. Users with at least three positive interactions are retained.
4. Each seed creates a random leave-two-out split.
5. One positive interaction is used for validation.
6. One positive interaction is used for testing.
7. Remaining positive interactions are used for training.
8. Negative interactions below the positive threshold are retained as negative evidence.
9. The default random seeds are 42, 123, 999, 2024, and 555.
10. The default top-k value is 10.
11. Candidate-size settings are 20, 40, 80, and all eligible items.
12. Model selection is performed on the validation split.
13. Final evaluation is reported on feasible test cases where the held-out item appears in the gated candidate set.

---

## Main methodological flow

The implemented workflow follows the paper methodology.

### Step 1. Upstream clustered vocabulary from the previous study

The previous study prepares Thai performing arts vocabulary using a domain-specific Thai NLP and clustering pipeline. The current repository receives the resulting `cluster_results.csv` file as the upstream lexical artifact.

### Step 2. Semantic stopword filtering

The current project filters low-semantic-value terms from clustered Thai performing arts vocabulary using a preservation-oriented LLM-based protocol. This step aims to remove broad, abstract, redundant, or low-discriminative terms while retaining culturally meaningful vocabulary.

### Step 3. Taxonomy construction

The retained vocabulary is organized into a hierarchical semantic taxonomy using a primary-critic-architect LLM workflow. The taxonomy provides a reusable semantic structure for Thai performing arts vocabulary.

### Step 4. Keyword-to-item grounding

The retained and classified keywords are linked to Thai performing arts items. This step produces item-linked semantic evidence that supports candidate prioritization and content-based ranking.

### Step 5. Data loading and split construction

The system loads item metadata, user interaction logs, keyword mappings, and taxonomy outputs. It then creates reproducible train-validation-test splits for each random seed.

### Step 6. Eligibility-gated candidate construction

For each user query and context, the eligibility gate performs context matching against item metadata. It builds a context-valid candidate pool, prioritizes items that match selected keywords, and controls candidate size. The adaptive minimum candidate logic does not add items from outside the context-valid pool.

### Step 7. Content-based filtering

The content-based model encodes item names and descriptions as item embeddings. It also encodes selected context and selected keywords as query evidence. Candidate items are scored using cosine similarity and a validation-tuned keyword-hit boost.

### Step 8. Collaborative filtering

Collaborative filtering models learn from user-item interactions. The evaluated models include EASE_R, BiasedMF, BiasedMF-BPR, ItemKNN, and SimpleX. POP-Global and POP-Context are used as non-personalized baselines.

### Step 9. Hybrid ranking

Hybrid models combine content-based and collaborative filtering scores. The main hybrid method uses weighted-sum fusion after score calibration. Additional fusion strategies are included for comparison.

### Step 10. Evaluation and reporting

The pipeline computes HR@10, MRR@10, and nDCG@10. It also exports overall results, per-user metrics, candidate contamination checks, context-policy summaries, inference-time summaries, and paper-ready tables.

---

## Main outputs

`run_pipeline.py` writes recommendation outputs to the `output` directory by default. Important files include:

1. `experiment_config.json`  
   Configuration record for reproducibility.

2. `results_cbf.csv`  
   Content-based filtering results.

3. `results_cf.csv`  
   Collaborative filtering and popularity baseline results.

4. `results_hybrid.csv`  
   Hybrid fusion results.

5. `results_overall.csv`  
   Aggregated comparison across model families.

6. `results_hybrid_methods.csv`  
   Summary of hybrid fusion methods.

7. `per_user_metrics.csv`  
   Per-user evaluation records.

8. `results_contamination.csv`  
   Cross-context contamination check for gated candidates.

9. `results_context_policy.csv`  
   Context policy and candidate construction report.

10. `results_inference_time.csv`  
    Inference-time benchmark summary when benchmarking is enabled.

11. `results_by_user_bin.csv`  
    Results by user activity group.

12. `results_significance_paired_user.csv`  
    Paired significance analysis when enough per-user pairs are available.

---

## Implementation notes

1. All recommendation models use the same eligibility gate.  
   This ensures fair comparison because CBF, CF, popularity baselines, and hybrid models receive the same candidate set for each user, seed, phase, and candidate-size setting.

2. Candidate construction checks context validity before scoring.  
   Context filtering occurs before any ranking model computes scores. This prevents high preference scores or high similarity scores from overriding cultural appropriateness.

3. Keyword evidence only prioritizes items inside the valid context pool.  
   Selected keywords adjust the order of items within the context-valid candidate pool. They do not introduce context-invalid items.

4. Target item names are filtered from selected keywords during validation and testing.  
   This reduces keyword leakage when the held-out item name appears in the user keyword log.

5. Negative interactions are retained as negative evidence.  
   Interactions below the positive threshold are not used as held-out positives, but they can reduce the influence of disliked items during scoring or training.

6. Multi-seed evaluation supports reproducibility.  
   The default experiment uses five deterministic seeds to reduce dependence on a single random split.

7. CPU mode is used by default.  
   The current configuration sets the device to CPU. This is sufficient for the small item catalog used in the experiment, although embedding models can run faster on a supported GPU.

---

## Privacy and GitHub release notes

Before uploading this repository to a public GitHub repository, remove personally identifiable information, confidential procurement data, private client or institutional information, and real API keys.

If the full dataset cannot be released, provide anonymized sample files under `input/sample` and describe the full data schema in the paper. The code can remain reproducible by specifying the required schema and sharing non-sensitive processed artifacts where possible.

---

## Suggested citation placeholder

If this repository is cited as a paper artifact, replace the placeholder below with the final bibliographic information.

```text
Author Name. Year. Semantic Artifacts and Eligibility Gating for Hybrid Recommendation in Thai Performing Arts. Journal or Conference Name. Repository URL.
```

The previous study should also be cited when explaining the origin of `cluster_results.csv`.

```text
Dumnil, P., and Buranasaksee, U. Year. Enhancing Thai Text Clustering for Performing Arts Through Domain-Specific NLP and Clustering Algorithms. Conference or Journal Name.
```

---

## License

Add the intended license before publishing the repository publicly. If the repository is released only as a paper artifact, clearly state whether the code, sample data, and derived artifacts may be reused.
