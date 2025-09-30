# Multi-Stage OCR Processing and Validation Pipeline

**Input:** Digitized or undigitized document images (low-quality, noisy, or artifact-rich)
**Output:** Validated OCR text with associated confidence and quality labels

---

## Step 1: Pre-Processing

### 1.1 Error Identification using VLMs

**Model:** Lightweight VLM (e.g., Qwen-VL-7B)

**Task:** Detect artifacts, orientation, and readability issues in scanned pages.

**Prompt (VLM):**

```text
You are a document analysis system.  
Given this scanned page, identify the following issues:  
1. Page orientation (normal, rotated, upside-down).  
2. Presence of noise, blur, or watermarks.  
3. Regions of non-text (stamps, illustrations, smudges).  
4. Overall readability score (High, Medium, Low).  

Return the issues as a structured JSON object.
```

---

### 1.2 Artifact Removal & Enhancement

**Techniques:**

* Orientation correction, noise/blur removal
* Super-Resolution (SRGAN) for low-resolution pages
* Qwen Image Edit for severe degradation

**Prompt (Qwen Image Edit):**

```text
You are an image enhancement system.  
Task: Improve readability of this scanned document.  

Instructions:  
- Sharpen text edges.  
- Remove background noise and smudges.  
- Correct orientation if tilted.  
- Increase resolution while preserving textual structure.  

Return the enhanced page without altering the content.
```

---

## Step 2: OCR Generation with Human-in-the-Loop

**Approach:**

* Ensemble of specialist OCR models (per script/language)
* Generalist VLMs for layout interpretation
* Periodic human calibration on sampled pages

**Prompt (Generalist VLM for layout):**

```text
You are an OCR layout assistant.  
Given this scanned page, output:  
1. The document layout (columns, tables, figures).  
2. Logical reading order of text blocks.  
3. Any script/language hints detected.  

Return in structured JSON to assist OCR alignment.
```

---

## Step 3: Post-OCR Quality Enhancement

**Techniques:**

* Rule-based filters (dictionary constraints, script normalization)
* LLM-based post-correction for grammar, consistency, semantic alignment

**Prompt (LLM Post-Correction):**

```text
You are an expert in correcting OCR output for printed text.  
The OCR output may contain:
- Misrecognized characters (e.g., language-specific confusions)
- Broken, merged, or incomplete words
- Incorrect line or paragraph order due to multi-column layouts
- Stray punctuation, symbols, or other OCR artifacts

Your task is to:
1. Correct all character-level and word-level OCR errors.  
2. Restore proper reading order and logical sentence/paragraph flow.  
3. Remove meaningless OCR noise while preserving all valid content.  
4. Keep the text in the original script.  
5. Output only clean, coherent text in proper writing.

Now correct this OCR output accordingly:
```

---

## Step 4: Validation & Consistency Checking

### 4.1 Reconstruction with hOCR + Style Transfer

**Prompt (Qwen Image Edit - Style Transfer):**

```text
You are an image reconstruction system.  
Task: Using the provided hOCR, generate an image that visually resembles the original scanned document.  

Constraints:  
- Preserve layout, fonts, and formatting.  
- Apply natural degradation styles common in manuscripts (faded ink, paper texture).  

Return the synthetic page.
```

---

### 4.2 Embedding Similarity Check

* Compute cosine similarity between embeddings of original vs reconstructed images

---

### 4.3 Reasoning-Based Validation with VLMs

**Prompt (Reasoning VLM):**

```text
You are a reasoning OCR evaluator.  
Given this scanned page, describe step by step:  
1. What the page contains (title, paragraphs, tables, etc.).  
2. The key semantic content (entities, topics, structure).  

Return a reasoning trajectory of what you understand from this page.
```

---

### 4.4 Trajectory Comparison with LLMs

**Prompt (Text LLM Comparison):**

```text
You are a similarity evaluator.  
Input: Two reasoning trajectories of the same page (original vs reconstructed).  

Task:  
1. Compare their semantic overlap.  
2. Assign a similarity score (0–100).  
3. Provide justification for the score.  
4. Bucket the result into {Good, Acceptable, Bad}.  

Return output as JSON: {score, justification, bucket}.
```

---

### 4.5 Score-Reason Consistency Check

**Prompt (Consistency Check LLM):**

```text
You are a validation assistant.  
Input: {score, justification} from similarity evaluator.  

Task: Verify if the justification logically supports the score.  
Return either "Consistent" or "Inconsistent".
```

---

## Step 5: Human Expert Review and Manual Post-Correction

* Pages flagged as low similarity or inconsistent are routed to human linguists for manual validation.

---

**Final Result:**
An orchestrated OCR pipeline integrating:

* Artifact detection
* Super-resolution
* Ensemble OCR
* Post-correction
* Reconstruction-based validation
* Reasoning-VLM alignment
* Human-in-the-loop oversight

All model interactions are guided by instruction-grade prompts for reproducibility.

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

- VLMs cover Indic scripts but hallucinate extra text, inflating CER and WER beyond 1.
- Specialist OCR models outperform after preprocessing + postcorrection
- Postcorrection improves CER by ~50% and WER by ~40% on average
- Pretraining on processed OCR text yields smoother convergence vs. raw OCR
