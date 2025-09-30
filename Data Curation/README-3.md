Perfect 👌 Thanks for sharing both. Since both READMEs are about the same project but highlight different angles — the **technical pipeline (first one)** and the **research/experiments + results (second one)** — I’ll carefully merge them into one clean `README.md` without losing any information.

Here’s the **final merged README** you can directly use in your GitHub repo:

````markdown
# 📘 Common Crawl Indic Language Processing & Data Curation Pipeline

This repository provides a **complete pipeline** for downloading, filtering, classifying, and curating Indic language text data from **Common Crawl** WET dumps.  
It also includes **curation experiments** conducted for research (published in our paper), demonstrating the impact of curated vs. conventional datasets on pretraining performance.

---

## ⚡ Pipeline Overview

### 1. Download & Stream Filtering
* Download each WET file from Common Crawl using `wget` (10–12 workers to avoid IP blocking).
* Unzip in memory with `pigz`.
* Stream through regex filters to separate Indic language candidates.

### 2. Regex Filtering (Streaming)
* Apply Unicode regex patterns for Devanagari, Bengali, Tamil, Telugu, etc.
* Write matched lines into separate JSONL files for each language.

### 3. Strict Language Classification
* Use **FastText** (`lid.176.bin`) to refine classification.
* Disambiguates languages where Unicode ranges overlap (e.g., Hindi vs. Marathi).
* Supports **strict mode**: ensures script-based character verification.

### 4. Data Curation
* Apply multiple filters using **NeMo-Curator**:
  * `clean_and_unify` (basic cleaning, normalization)
  * `heuristic_filter` (quality-based filtering)
  * `dedupe` (exact & fuzzy deduplication)
  * `redact_pii` (removes personally identifiable information)
  * Toxic filtering (rule-based wordlist, provided in repo)
  * Quality filtering (document-level, boilerplate removal, word/symbol checks)

---

## 🛠️ Setup & Requirements

### System Dependencies
* Linux environment
* Python 3.9+
* [pigz](https://zlib.net/pigz/) (parallel gzip decompression)
* wget

### Python Libraries
Install requirements:
```bash
pip install -r requirements.txt
````

Typical requirements:

* `tqdm`
* `fasttext`
* `nemo-curator`
* `dask[distributed]`
* `regex`
* `transformers`
* `datasets`

Or create an environment:

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

### 2. Regex Filtering (Standalone Option)

```bash
cat file.wet | pigz -d | \
python3 streaming_regex.py \
  --output-dir ./regex_output \
  --output-file filtered.jsonl \
  --workers 16
```

### 3. Language Classification with FastText

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

Or via provided scripts:

```bash
bash scripts/run_curator.sh --config configs/curator.yaml \
                            --input data/raw/ \
                            --output data/curated/
```

Inspect curated data:

```bash
jupyter notebook notebooks/quality_checks.ipynb
```

---

## 📂 Output Structure

```
output/
 ├── tel/         # Telugu regex matches
 ├── hin/         # Hindi regex matches
 ├── ben/         # Bengali regex matches
 ...
classified/
 ├── hindi.txt
 ├── bengali.txt
 ├── tamil.txt
 ...
curated/
 ├── hindi_cleaned.jsonl
 ├── bengali_cleaned.jsonl
 ├── tamil_cleaned.jsonl
```

---

## 📊 Research & Ablation Experiments

Our **ablation study** evaluated the effect of curation on pretraining outcomes.

### Curation Pipeline Diagram

![Curation Pipeline](/readme-resources/data-curation.png)

### Stages of Curation

1. **Raw Corpus Construction** (aggregate large-scale text, English + Hindi).
2. **Deduplication & Cleaning** (NeMo Curator DCLM, FWE filtering).
3. **Quality Filtering** (remove boilerplate/noise, enforce language ID).
4. **Indian Language Adaptation** (Hindi tokenization + Unicode unification).
5. **Final Curated Dataset** (domain-diverse, high-quality corpus).

### Base Pretraining Checkpoint

* **Model**: Param-1 PT (2.9B parameters)
* **Checkpoint**: [Hugging Face: Param-1 PT1](https://huggingface.co/bharatgenai/Param-1)
* **Tokens Trained**: 5T (baseline) + 2T (experiment extension)
* **Training Recipe**: [Param Paper](https://arxiv.org/pdf/2507.13390)

### Training Data Conditions

1. **Without Curation**: [Download Non-Curated](https://example.com/datasets/param_ablation/english_hindi_noncurated)
2. **With Curation**: [Download Curated](https://example.com/datasets/param_ablation/english_hindi_curated)

Both datasets had **2T tokens** for fair comparison.

### Results – Benchmark Performance

| **Model**    | **ARC Challenge** | **ARC Easy** | **Hella Swag** | **Hella Swag Hi** | **MMLU** | **MMLU Hi** |
| ------------ | ----------------: | -----------: | -------------: | ----------------: | -------: | ----------: |
| Conventional |              46.5 |         73.6 |           73.5 |              28.9 |     41.3 |        26.2 |
| Curated      |              53.6 |         74.2 |           73.8 |              41.4 |     46.2 |        34.6 |

Curated data shows **consistent improvements**, especially in Indic language benchmarks.

### Toxicity Reduction

![Toxicity Sample](/readme-resources/toxic-comparison)

Curated pipeline significantly reduces toxic/low-quality content.

---

## 📂 Repository Structure

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

* Run the download step with **10–12 workers max** to avoid IP throttling from Common Crawl.
* Regex filtering is **broad**; FastText + script verification ensures accuracy.
* For **large-scale runs**, use a **Dask cluster** for curation.
* Curated and non-curated corpora are both compatible with **NeMo pretraining recipes**.

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

---