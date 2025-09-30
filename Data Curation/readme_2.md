 ### Downloading and Reproducibility Links Coming within 24 hours
 
---

# Data Curation

This section describes our **curation pipeline** and the **ablation experiment** conducted to measure its effectiveness.

---

## Folder Structure

```
iclr-submission/
 ├── data_curation/
 │    ├── curation/
 │    │    ├── cc_curator.py
 │    │    ├── language_detector.py
 │    │    ├── nemo_curator.py
 │    │    ├── streaming_regex.py
 │    ├── deduplication/
 │    │    ├── deduplication.py
 │    │    └── deduplication.sh
 │    ├── toxic_filter/
 │    │    ├── sample_toxic_words.txt
 │    │    ├── toxic_filter_inference.py
 │    │    └── toxic_filter_rule.py
 │    ├── quality_filter/
 │    │    ├── quality_filter.py
 │    └── README.md
```
---

### Curation Pipeline

---

## Curation Pipeline Overview

**Pipeline Diagram**
![Curation Pipeline](/readme-resources/data-curation.png)

**Description of Stages**

1. **Raw Corpus Construction**

   * Aggregate large-scale text from publicly available sources (English + Hindi).
2. **Deduplication and Cleaning**

   * NeMo Curator deduplication (DCLM)
   * Filter by word/character distribution statistics (FWE filtering)
3. **Quality Filtering**

   * Document-level filtering: removal of boilerplate, low-information pages, and noise
   * Language ID filtering (restricting to English + Hindi)
4. **Indian Language Adaptation**

   * Language-specific tokenization and normalization for Hindi
   * Script unification for consistent Unicode handling
5. **Final Curated Dataset**

   * High-quality, domain-diverse corpus passed to training

---

## Base Pretraining Checkpoint

For Ablation study purposes we chose the 2.9B param range of model to test it on.
Based on our internal experimentatin , we have empirically observed a scaling effect applicable from small models to typical medium sized and large size models.
https://ar5iv.labs.arxiv.org/html/2001.08361
https://arxiv.org/abs/2403.06563
https://arxiv.org/html/2412.01505
https://www.emergentmind.com/papers/2403.08540

* **Model**: Param-1 PT (2.9B parameters)
* **Checkpoint Source**: [Hugging Face: Param-1 PT1](https://huggingface.co/bharatgenai/Param-1)
* **Training Recipe**: As described in [Param Paper](https://arxiv.org/pdf/2507.13390)
* **Tokens Trained**: 5T (before this ablation experiment)

---

## Training Data Composition

We extended training with **2T tokens** under two conditions:

1. **Without Curation** (Conventional Corpus)

   * Raw text with only basic preprocessing
   * Download: `https://example.com/datasets/param_ablation/english_hindi_noncurated`

2. **With Curation** (Curated Corpus via Pipeline)

   * Same sources passed through the NeMo Curator pipeline
   * Download: `https://example.com/datasets/param_ablation/english_hindi_curated`

Both datasets are matched in size (**2T tokens**) to ensure comparability.

---

## Curation Scripts and Codebase

All scripts are provided under [`iclr-submission/Data_Curation/`](experiments/data_curation/).

### Key Components

* `curation/curator.py` → Curation Pipeline (Cleaning, Heuristic Filters, Redact PII etc.)
* `deduplication/deduplciation.sh` → Bash file for global deduplication
* `quality_filter/quality_filter.py` → Quality Filter (Low, Medium, High)
* `toxic_filter/toxic_filter_rule.py` → Rule Based Toxic Filtering (word list included for 1 language)

---

## Steps to Run

1. **Setup Environment**

   ```bash
   conda create -n curator python=3.10 -y
   conda activate curator
   pip install nemo_toolkit[all] transformers datasets
   ```

2. **Download Raw Corpus**

   ```bash
   bash scripts/download_raw.sh --languages en,hi --output data/raw/
   ```

3. **Run Curation Pipeline**

   ```bash
   bash scripts/run_curator.sh --config configs/curator.yaml \
                               --input data/raw/ \
                               --output data/curated/
   ```

4. **Inspect Curated Data**

   ```bash
   jupyter notebook notebooks/quality_checks.ipynb
   ```

5. **Use in Pretraining**

   * Curated and non-curated corpora are directly pluggable into the NeMo pretraining recipes.
   * Training scripts are provided under [`experiments/pretraining/`](experiments/pretraining/).

---

### Results Obtained

### Ablation Experiment 1: Benchmark Results: Conventional vs Curated

### Conventional vs Curated Data Sample

![Curation Sample](/readme-resources/curation.png)

### Benchmark Table
| **Model**      | **ARC Challenge** | **ARC Easy** | **Hella Swag** | **Hella Swag Hi** | **MMLU** | **MMLU Hi** |
|----------------|------------------:|--------------:|----------------:|------------------:|---------:|------------:|
| Conventional   | 46.5              | 73.6          | 73.5            | 28.9              | 41.3     | 26.2        |
| Curated        | 53.6              | 74.2          | 73.8            | 41.4              | 46.2     | 34.6        |

### Ablation Experiment 2: Toxicity Comparison

![Toxicity Sample](/readme-resources/toxic-comparison)

### Observation of the Resutls

We observe that after performing curation increase in the score on the various benhcmarks are obtained.
