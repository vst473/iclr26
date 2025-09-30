Got it 👍 — the merged one is too dense right now, so the GitHub preview won’t feel as polished as your original READMEs. Let’s fix that.

To make it look **clean, structured, and professional** in GitHub’s markdown preview, we should:

* Keep **short sections with clear headers**.
* Use **emoji headers sparingly** for readability.
* Place **diagrams, tables, and results** in collapsible `<details>` blocks so the front page isn’t overloaded.
* Separate **practical pipeline usage** and **research results** — so developers can quickly run the code, and researchers can expand the details if they want.

Here’s a **cleaned-up and better-styled final `README.md`** you can use:

````markdown
# 📘 Common Crawl Indic Language Processing & Data Curation

This repository provides:
- A **four-stage pipeline** for downloading, filtering, classifying, and curating Indic language text data from **Common Crawl** WET dumps.  
- **Curation experiments & results** from our research paper, showing the benefits of high-quality data curation.

---

## ⚡ Pipeline Overview

1. **Download & Stream Filtering** – fetch WET dumps, unzip with `pigz`, stream regex filters.  
2. **Regex Filtering** – Unicode-based matching for Indic scripts (Hindi, Bengali, Tamil, Telugu, etc.).  
3. **Language Classification** – strict FastText filtering with script verification.  
4. **Data Curation** – NeMo-Curator filters for cleaning, deduplication, heuristic quality checks, PII redaction, toxic filtering.

---

## 🛠️ Setup

### System Dependencies
- Linux environment
- Python 3.9+
- [pigz](https://zlib.net/pigz/) (parallel gzip decompression)
- `wget`

### Installation
```bash
pip install -r requirements.txt
````

Or with Conda:

```bash
conda create -n curator python=3.10 -y
conda activate curator
pip install nemo_toolkit[all] transformers datasets
```

---

## 🚀 Usage

### 1. Download & Process Dumps

```bash
python3 process_final.py \
  https://data.commoncrawl.org/CC-MAIN-2025-08/wet.paths.gz \
  /path/to/output \
  --workers 12 \
  --batch-size 20
```

### 2. Regex Filtering (Standalone)

```bash
cat file.wet | pigz -d | \
python3 streaming_regex.py \
  --output-dir ./regex_output \
  --output-file filtered.jsonl \
  --workers 16
```

### 3. Language Classification

```bash
python3 classify_fasttext.py \
  --input ./regex_output/hin/regex_file.txt \
  --output ./classified/hindi.txt \
  --model ./lid.176.bin \
  --languages hindi bengali tamil telugu \
  --strict
```

### 4. Data Curation

```bash
python3 curate_data.py \
  --input ./classified/hindi.txt \
  --output ./curated/hindi_cleaned.jsonl
```

Or use the provided script:

```bash
bash scripts/run_curator.sh --config configs/curator.yaml \
                            --input data/raw/ \
                            --output data/curated/
```

---

## 📂 Output Structure

```
output/
 ├── tel/         # Telugu regex matches
 ├── hin/         # Hindi regex matches
 ├── ben/         # Bengali regex matches
classified/
 ├── hindi.txt
 ├── bengali.txt
 ├── tamil.txt
curated/
 ├── hindi_cleaned.jsonl
 ├── bengali_cleaned.jsonl
 ├── tamil_cleaned.jsonl
```

---

## 📊 Research & Ablation Experiments

<details>
<summary>🔎 Curation Pipeline & Experiments (click to expand)</summary>

### Pipeline Diagram

![Curation Pipeline](/readme-resources/data-curation.png)

### Stages

1. Raw corpus construction (English + Hindi).
2. Deduplication & cleaning (NeMo Curator, FWE filtering).
3. Quality filtering (boilerplate removal, noise filtering, strict LangID).
4. Indic adaptation (Hindi tokenization + Unicode normalization).
5. Final curated dataset (domain-diverse, high-quality).

### Base Pretraining Setup

* **Model**: Param-1 PT (2.9B parameters)
* **Checkpoint**: [Hugging Face: Param-1 PT1](https://huggingface.co/bharatgenai/Param-1)
* **Tokens Trained**: 5T baseline + 2T extension
* **Recipe**: [Param Paper](https://arxiv.org/pdf/2507.13390)

### Data Conditions

* **Non-curated**: [Download](https://example.com/datasets/param_ablation/english_hindi_noncurated)
* **Curated**: [Download](https://example.com/datasets/param_ablation/english_hindi_curated)

Both matched at **2T tokens**.

### Results – Benchmark Scores

| **Model**    | **ARC Challenge** | **ARC Easy** | **Hella Swag** | **Hella Swag Hi** | **MMLU** | **MMLU Hi** |
| ------------ | ----------------: | -----------: | -------------: | ----------------: | -------: | ----------: |
| Conventional |              46.5 |         73.6 |           73.5 |              28.9 |     41.3 |        26.2 |
| Curated      |              53.6 |         74.2 |           73.8 |              41.4 |     46.2 |        34.6 |

**Observation**: Curated data consistently improves performance, especially in Hindi benchmarks.

### Toxicity Reduction

![Toxicity Sample](/readme-resources/toxic-comparison)

</details>

---

## 📂 Repository Layout

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

## ⚠️ Notes

* Use **10–12 workers max** during download to avoid throttling.
* Regex filtering is broad; FastText + script checks ensure strict accuracy.
* Use a **Dask cluster** for large-scale curation.
* Both curated and non-curated datasets plug directly into NeMo pretraining.

---

## 📌 References

* [Common Crawl](https://commoncrawl.org/)
* [FastText Language ID](https://fasttext.cc/docs/en/language-identification.html)
* [NVIDIA NeMo Curator](https://github.com/NVIDIA/NeMo-Curator)
* [Param Paper](https://arxiv.org/pdf/2507.13390)
* [https://ar5iv.labs.arxiv.org/html/2001.08361](https://ar5iv.labs.arxiv.org/html/2001.08361)
* [https://arxiv.org/abs/2403.06563](https://arxiv.org/abs/2403.06563)
* [https://arxiv.org/html/2412.01505](https://arxiv.org/html/2412.01505)
* [https://www.emergentmind.com/papers/2403.08540](https://www.emergentmind.com/papers/2403.08540)

