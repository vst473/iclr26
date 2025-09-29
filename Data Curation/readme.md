---

# 📘 Common Crawl Indic Language Processing Pipeline

This repository provides a **four-stage pipeline** for downloading, filtering, classifying, and curating Indic language text data from **Common Crawl** WET dumps.

The pipeline is optimized for **scalable processing** with parallel workers, regex-based streaming filters, language classification using **FastText**, and curation using **NeMo-Curator** filters.

---

## ⚡ Pipeline Overview

1. **Download & Stream Filtering**

   * Download each WET file from Common Crawl using `wget` (10–12 workers to avoid IP blocking).
   * Unzip in memory with `pigz`.
   * Stream through regex filters to separate Indic language candidates.

2. **Regex Filtering (Streaming)**

   * Apply Unicode regex patterns for Devanagari, Bengali, Tamil, Telugu, etc.
   * Write matched lines into separate JSONL files for each language.

3. **Strict Language Classification**

   * Use **FastText** (`lid.176.bin`) to refine classification.
   * Disambiguates languages where Unicode ranges overlap (e.g., Hindi vs. Marathi).
   * Supports **strict mode**: ensures script-based character verification.

4. **Data Curation**

   * Apply multiple filters using **NeMo-Curator**:

     * `clean_and_unify` (basic cleaning, normalization)
     * `heuristic_filter` (quality-based filtering)
     * `dedupe` (exact & fuzzy deduplication)
     * `redact_pii` (removes personally identifiable information)

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
```

Typical requirements:

* `tqdm`
* `fasttext`
* `nemo-curator`
* `dask[distributed]`
* `regex`

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

* Downloads and streams WET files.
* Filters with regex (via `streaming_regex.py`).
* Saves per-language intermediate files in `output/{lang}/regex_*.txt`.

---

### 2. Regex Filtering (Streaming)

Called internally by the first script, but can also be run standalone:

```bash
cat file.wet | pigz -d | \
python3 streaming_regex.py \
  --output-dir ./regex_output \
  --output-file filtered.jsonl \
  --workers 16
```

---

### 3. Language Classification with FastText

```bash
python3 classify_fasttext.py \
  --input ./regex_output/hin/regex_file.txt \
  --output ./classified/hindi.txt \
  --model ./lid.176.bin \
  --languages hindi bengali tamil telugu \
  --strict
```

* Loads FastText language ID model.
* Applies **script verification** in strict mode.
* Produces per-language clean text files.

---

### 4. Data Curation

```bash
python3 curate_data.py \
  --input ./classified/hindi.txt \
  --output ./curated/hindi_cleaned.jsonl
```

Curation includes:

* Cleaning (unifying quotes, removing boilerplate HTML).
* Heuristic filtering (length, repetition, word/symbol ratios).
* Deduplication (exact + fuzzy).
* PII redaction.

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

## ⚠️ Notes

* Run the download step with **10–12 workers max** to avoid IP throttling from Common Crawl.
* Regex-based filtering is **broad** (may catch noise). The FastText step is **strict** for accuracy.
* For **large-scale runs**, it’s recommended to use a Dask cluster for the curation step.

---

## 📌 References

* [Common Crawl](https://commoncrawl.org/)
* [FastText Language ID](https://fasttext.cc/docs/en/language-identification.html)
* [NVIDIA NeMo Curator](https://github.com/NVIDIA/NeMo-Curator)

---
