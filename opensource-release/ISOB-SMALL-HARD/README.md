# ISOB: Indian Synthetic OCR Benchmark – Small-Hard

Since a significant portion of our corpus originates from copyrighted materials obtained under formal MoUs and partnerships, we cannot release those pages directly as benchmarks.

However, recognizing the complexity and challenges inherent in such offline digitized documents—and to support the research community—we are introducing the **first version of the Indian Synthetic OCR Benchmark – Small-Hard**.

Future releases will expand to include:

* **Indic-Real-OCR Benchmarks** (licensed for public release)

  * Difficulty levels: Easy, Medium, Hard
  * Dataset sizes: Small, Medium, Large

* **Indic-Synthetic-OCR Benchmarks**

  * Difficulty levels: Easy, Medium, Hard
  * Dataset sizes: Small, Medium, Large

---

## Input & Output

* **Input:** Seed corpus of OCR’d pages in hOCR format
* **Output:** Synthetic benchmark dataset of hard-to-OCR images with:

  * Ground truth hOCR
  * Language tags

---

## Steps to Build the Benchmark

### Step 0: Initialize Seed Corpus

* Use existing OCR’ed pages (JSON/hOCR format) as the starting corpus.

### Step 1: Hard Page Identification

* Filter pages using OCR confidence scores:

  * Discard pages with very low confidence (e.g., whitespace, empty, or low-text pages).
* Use **Qwen-VL-7B** grounded with the page hOCR to predict difficulty.
* Select pages predicted as **hard-to-OCR**.

**Result:** `hOCR_1` (hard page set)

### Step 2: Language Selection

* Randomly choose **3–10 languages** from a pool of 22 Indian languages.

### Step 3: Artifact Taxonomy Extraction

* Build a taxonomy of “hard artifacts” from the seed hOCR corpus.
* Use an **LLM** to generate this list to stay grounded in real document complexities.

**Examples of Hard Artifacts:**

* Multi-column layouts
* Dense tables
* Handwriting inserts
* Overlapping scripts
* Reading order complexities
* Equations, figures, pie charts
* Complex tables

### Step 4: Synthetic hOCR Augmentation

* Augment each `hOCR_1` using:

  * Selected languages (Step 2)
  * Artifact templates (Step 3)
* Produce enriched hOCR documents → `hOCR_2`
* **hOCR_2 serves as the ground truth**

### Step 5: hOCR to Visual Conversion

* Render each `hOCR_2` into **PDF/image format**.

### Step 6: Style Transformation (Prompt Pool + Image Editing)

* Construct a **prompt pool** describing Indian manuscript styles, books, literature, and other domain-specific styles.
* For each page:

  * Sample a prompt from the pool
  * Apply **Qwen image editing** with the prompt to transform visual style

### Step 7: Image Processing Augmentation

* Apply low-level transformations to further increase difficulty:

  * Orientation changes
  * Contrast/brightness shifts
  * Noise, blur, distortions, etc.

### Step 8: Storage & Annotation

* Save final images with metadata:

  * Associated ground truth `hOCR_2`
  * Language tags (Step 2)
  * Style/augmentation metadata

---

## End Result

A structured, multilingual, style-rich, **artifact-heavy synthetic OCR benchmark** that systematically captures hard-to-recognize text cases.
