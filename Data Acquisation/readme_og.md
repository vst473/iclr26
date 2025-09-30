### Downloading and Reproducibility Links Coming within 24 hours.
---

# Data Acquisition

This section documents the **acquisition pipeline** for our multilingual pretraining corpus. It covers the **sources of data**, the **collection methodology**, and the **organization schema** applied to ensure reproducibility, integrity, and curriculum-aware structuring.

---

## Overview

We aggregate large-scale corpora from **multi-source web crawling**, **curated open datasets**, and **book/academic repositories**. Our acquisition emphasizes **authentic, curriculum-aligned content** to mitigate linguistic and cultural gaps commonly present in purely synthetic or translated datasets.

**Key Sources**
- **Web Crawl + Open Datasets**
  - Common Crawl, multilingual websites, forums, academic repositories
  - >1700 datasets hosted on [Hugging Face](https://huggingface.co)
- **Book Collections**
  - ~1M books from [Archive.org](https://archive.org)
  - 28,500 curriculum-aligned documents from the [National Digital Library of India (NDLI)](https://ndl.iitkgp.ac.in)

---

## Acquisition Methodology

1. **Source Integration**
   - Inspired by *Pile* [1], *RedPajama* [2], and *C4* [3].
   - Unified crawl + dataset ingestion pipeline with provenance tracking.

2. **Schema-First Cataloging**
   - Every acquired item is labeled along **orthogonal axes**:
     - Language  
     - Grade level (for school curricula)  
     - Content provider / institution  
     - Subject domain (for higher education)  
   - Multi-labeled items (e.g., bilingual, multi-grade) are preserved with **multiple metadata tags**.

3. **Deduplication & Integrity**
   - Deduplication performed at the **item-ID level**, preventing loss of cross-listed data.
   - Metadata normalization ensures **cross-source compatibility**.

4. **Distribution Reporting**
   - Tabular distributions provided at multiple levels:
     - NDLI **school-level content** (languages, classes, providers)
     - NDLI **higher education** (providers, levels, subjects)
   - Enables **coverage quantification** and **balanced sampling**.

5. **Scalable Processing**
   - Designed for **millions of documents**, including OCR + post-correction workflows for Indic scripts.
   - Robust **checkpointing + shard-based retries** for week-scale ingestion tasks.

---

## Steps to Reproduce

*To be added: detailed code instructions.*

---

## Folder Structure

```
experiments/
 ├── data_acquisition/
 │    ├── configs/
 │    │    └── schema.yaml
 │    ├── scripts/
 │    │    ├── download_books.sh
 │    │    ├── download_ndli.sh
 │    │    ├── download_hf.sh
 │    │    └── catalog_items.py
 │    ├── notebooks/
 │    │    └── distribution_analysis.ipynb
 │    └── README.md
 ├── data_curation/
 └── pretraining/
```

---

## Observations

The dataset statistics across different languages are summarized below:

| **Language** | **# PDFs** | **# Pages** | **Word Count** |
|--------------|-----------:|------------:|---------------:|
| Hindi        | 396.12 K   | 7.53 M      | 4.15 B         |
| Marathi      | 124.22 K   | 3.02 M      | 1.26 B         |
| Malayalam    | 65.03 K    | 2.18 M      | 1.06 B         |
| Telugu       | 77.86 K    | 5.93 M      | 1.53 B         |
| Tamil        | 43.59 K    | 5.28 M      | 1.44 B         |
| Kannada      | 41.71 K    | 4.08 M      | 1.01 B         |
| Sanskrit     | 44.49 K    | 10.09 M     | 2.68 B         |
| Bengali      | 41.25 K    | 10.95 M     | 3.10 B         |
| Urdu         | 126.03 K   | 32.15 M     | 10.03 B        |
| English      | 45.10 K    | 2.57 M      | 0.89 B         |
| **Total**    | **1.00 M** | **84.00 M** | **27.15 B**    |

---
