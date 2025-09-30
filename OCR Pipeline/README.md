 ### Downloading and Reproducibility Links Coming within 24 hours.

---
# Indic OCR Pipeline

> A robust multi-stage pipeline for optical character recognition of low-resource Indian languages, designed to handle complex scripts and degraded documents.

---

## Overview

Indic languages face significant digital scarcity, particularly low-resource languages like Maithili and Sindhi. We developed a comprehensive OCR pipeline to process **5–6 million pages** from print materials and scanned books, converting inaccessible texts into machine-readable format.

### Key Challenges

- **Poor scan quality** — 37% of materials exhibited faded ink, irregular printing, and evolving orthographies
- **Complex scripts** — Ligatures, conjunct consonants, and diacritics across Devanagari, Bengali, Tamil, Telugu, and more
- **Layout variation** — Multi-column formats, newspapers, manuscripts, tables, and figures

---

## Pipeline Architecture

![OCR Pipeline](/readme-resources/ocr-pipeline.png)

### 1. Pre-Processing

- Denoising, binarization, and contrast enhancement
- Super-resolution (SRGAN) for readability improvement
- VLM-based artifact detection (orientation, blur, stamps, illustrations)

**Example prompt:**
```
You are a document analysis system. Identify:
1. Orientation (normal, rotated, upside-down)
2. Noise, blur, watermarks
3. Non-text regions
4. Overall readability score
Return structured JSON.
```

### 2. OCR Generation

- Ensemble of script-specific OCR models
- Generalist VLMs for layout interpretation
- Human-in-the-loop calibration on sampled pages

**Layout detection prompt:**
```
You are an OCR layout assistant. Output:
1. Document layout (columns, tables, figures)
2. Logical reading order
3. Script/language hints
Return as structured JSON.
```

### 3. Post-OCR Enhancement

- Rule-based normalization (dictionary checks, Unicode repair, spacing fixes)
- LLM-based postcorrection for grammar and semantic alignment

### 4. Validation

- hOCR reconstruction with style transfer
- Embedding similarity between original and reconstructed images
- Reasoning-based validation with VLMs
- Trajectory comparison using LLMs for semantic similarity scoring
- Human expert review for flagged cases

---

## ISOB: Indic Synthetic OCR Benchmark

Due to copyright constraints on scanned materials, we created **ISOB-Small**, a synthetic benchmark covering **22 Indian languages**.

### Features

- **110 synthetic pages** with diverse layouts and degradations
- Multi-column layouts, tables, equations, blur, shadows, watermarks, folds, font variation
- Stress-tests for ligatures, conjunct consonants, and diacritics

### Generation Pipeline

1. **Seed Corpus** — Initialize with OCR'd hOCR pages
2. **Hard Page Selection** — Identify difficult pages using confidence scores + VLM classifiers
3. **Language Sampling** — Random selection of 3–10 languages
4. **Artifact Taxonomy** — Extract complex layouts via LLMs
5. **Synthetic Augmentation** — Add multilingual + artifact-rich structures
6. **Visual Rendering** — Convert to images/PDFs
7. **Style Transformation** — Simulate manuscript/book styles
8. **Degradation** — Apply noise, blur, distortions
9. **Annotation** — Store with ground truth hOCR, language tags, metadata

---

## Folder Structure

```
experiments/
 ├── data_curation/
 │    ├── configs/
 │    │    └── curator.yaml
 │    ├── scripts/
 │    │    ├── run_curator.sh
 │    │    ├── download_raw.sh
 │    ├── notebooks/
 │    │    └── quality_checks.ipynb
 │    └── README.md
 ├── pretraining/
 │    ├── train_curated.sh
 │    ├── train_noncurated.sh
 │    └── configs/
 │         └── param_ablation.yaml
```

---

## Evaluation

### Metrics

- Character Error Rate (CER)
- Word Error Rate (WER)
- Position-Independent WER (PI-WER)
- Char3-gram F1

### LLM-Assisted Quality Evaluation

Traditional metrics are insufficient post-enhancement. Our framework includes:

- LLM-based quality judging (OCR vs. ground truth)
- Multilingual embedding similarity for semantic fidelity
- Back-translation to English for cross-lingual alignment
- BLEU, ROUGE, CHRF++ for translation-level evaluation
- Image embedding validation for reconstructed pages

### Experimental Results

#### Without Preprocessing and Postprocessing

| Model                                   | bashini CER | bashini WER | bashini PI-WER | bashini Char3gram-F1 | mozhi CER | mozhi WER | mozhi PI-WER | mozhi Char3gram-F1 | iiit CER | iiit WER | iiit PI-WER | iiit Char3gram-F1 |
|----------------------------------------|------------|------------|----------------|---------------------|-----------|-----------|--------------|------------------|----------|----------|-------------|-----------------|
| dotsOCR2                                | 0.168      | 0.253      | 0.23           | 0.801               | 0.12      | 0.19      | 0.9          | 0.88              | 0.15     | 0.22     | 0.91        | 0.86             |
| Surya                                   | 0.2        | 0.28       | 0.138          | 0.867               | 0.14      | 0.21      | 0.91         | 0.89              | 0.17     | 0.25     | 0.92        | 0.87             |
| Llama-4-Scout-17B-16E-Instruct         | 0.259      | 0.445      | 0.398          | 0.672               | 4.35      | 1.38      | 0.619        | 0.31              | 3.64     | 1.71     | 0.734       | 0.279            |
| NuMarkdown-8B-Thinking                  | 0.361      | 0.537      | 0.508          | 0.556               | 53.31     | 9.21      | 0.677        | 0.168             | 59.45    | 18.33    | 0.955       | 0.039            |
| Llama-4-Maverick-17B-128E-Instruct_final | 0.4       | 0.58       | 0.418          | 0.645               | 12        | 4         | 0.72         | 0.22              | 11.33    | 3.49     | 0.778       | 0.238            |
| Qwen2.5-VL-72B-Instruct                | 0.676      | 0.847      | 0.45           | 0.613               | 18.22     | 4.16      | 0.677        | 0.266             | 0.72     | 0.95     | 0.5         | 0.6              |
| Qwen2.5-VL-72-Instruct                  | 1.026      | 1.048      | 0.489          | 0.571               | 6.89      | 2.6       | 0.662        | 0.076             | 4.55     | 2.41     | 0.867       | 0.047            |
| SmolDocling-256M-preview                | 1.235      | 1.4        | 0.988          | 0.016               | 137.66    | 55.57     | 0.946        | 0.0001            | 116.04   | 66.68    | 1           | 0                |
| RolmOCR                                 | 1.938      | 2.019      | 0.498          | 0.552               | 986.91    | 263.08    | 0.692        | 0.111             | 501.2    | 176.7    | 0.884       | 0.069            |
| olmOCR-7B-0825                          | 2.068      | 1.842      | 0.516          | 0.531               | 28.53     | 6.48      | 0.704        | 0.126             | 12.19    | 5.61     | 0.89        | 0.072            |
| Nanonets-OCR-s                          | 3.573      | 2.318      | 0.568          | 0.471               | 305.27    | 42.57     | 0.685        | 0.161             | 102.62   | 31.15    | 0.891       | 0.089            |
| GLM-4.1V-9B-Thinking                    | 4.384      | 3.88       | 0.893          | 0.08                | 755.1     | 321.92    | 0.985        | 0.0001            | 488.06   | 257.78   | 0.984       | 0.011            |
| MinerU2.5-2509-1.2B                     | 5.176      | 3.214      | 0.906          | 0.095               | 180       | 55        | 0.91         | 0.1               | 160      | 60       | 0.93        | 0.08             |
| pixtral-12B                             | 5.847      | 4.86       | 0.893          | 0.163               | 1.47      | 0.999     | 0.941        | 0.0039            | 1.4      | 1        | 1           | 0.0001            |
| InternVL3_5-GPT-OSS-20B-A4B-Preview-HF | 38.87      | 4.537      | 0.994          | 0.0029              | 195.85    | 240.2     | 0.919        | 0                 | 400      | 500      | 0.95        | 0.05             |


#### With Preprocessing and Postprocessing

| Model | CER | WER | PI-WER | Char3-gram F1 |
|-------|-----|-----|--------|---------------|
| dots.OCR | 0.168 | 0.253 | 0.23 | 0.801 |
| dots.OCR - postcorrected | 0.085 | 0.145 | 0.12 | 0.91 |
| Surya | 0.2 | 0.28 | 0.138 | 0.867 |
| Surya - postcorrected | 0.095 | 0.16 | 0.11 | 0.925 |

### Key Findings

- VLMs provide coverage but hallucinate in Indic scripts
- Specialist OCR models outperform after preprocessing + postcorrection
- Postcorrection improves CER by ~50% and WER by ~40% on average
- Pretraining on processed OCR text yields smoother convergence vs. raw OCR

---

## Dataset Release

**ISOB-Small** (22 languages, 110 pages) will be released publicly for research purposes.

Includes:
- Synthetic generation recipes
- Augmentation scripts
- Ground truth annotations
- Reproducibility documentation

Future releases planned for **Indic-Real-OCR** and **Indic-Synthetic-OCR** variants.


| Model                                   | isob CER  | isob WER  | isob PI-WER | isob Char3gram-F1 |
|----------------------------------------|-----------|-----------|-------------|-----------------|
| dotsOCR2                                | 0.73919   | 0.86158   | 0.72613     | 0.30264         |
| Surya                                   | 0.88765   | 0.89821   | 0.88141     | 0.14812         |

---

## Key Insights

1. OCR quality depends critically on data availability and processing pipelines
2. Iterative refinement (preprocessing → OCR → postcorrection → validation) is essential
3. Synthetic benchmarks fill critical gaps in Indic OCR research
4. Human-in-the-loop validation ensures quality for low-resource scenarios

---

*This work addresses the digital divide for low-resource Indian languages through innovative OCR techniques and reproducible benchmarking.*
