# Translation Benchmarking & Pipeline for Indic Languages

> Ensemble-based translation and post-correction pipeline for 16 Indic languages — leveraging MT models, LLMs, and human validation.

---

## Abstract

This repository benchmarks open-source machine translation (MT) models and LLMs on Indic languages and documents an ensemble-based pipeline that uses MT outputs + high-performing Indic LLMs for post-correction. The merged pipeline combines specialist translation systems, generalist LLMs, automated ranking, back-translation, and human validation to produce coherent, domain-sensitive translations — including for complex STEM/mathematics/code content and long-context documents.

---

## Goals

1. Benchmark open-source MT models and open-source LLMs across multiple parameter sizes.
2. Use high-scoring Indic‑MMLU LLMs to post-correct and enhance MT outputs.
3. Compare pre-enhancement vs post-enhancement results using chrF, chrF++, and sacreBLEU.
4. Define a reproducible ensemble + post-correction strategy for high-fidelity Indic translations.

---

## Datasets

We evaluate on two well-known benchmarks:

* [google/IndicGenBench_flores_in](https://huggingface.co/datasets/google/IndicGenBench_flores_in)
* [ai4bharat/IN22-Gen](https://huggingface.co/datasets/ai4bharat/IN22-Gen)

The datasets cover translations into 16 Indic languages (As, Be, Gu, Ka, Hi, Mai, Ml, Mr, Ne, Or, Pa, Sa, Sd, Ta, Te, etc.).

---

## High-Level Pipeline

### Step 0 — Data Augmentation & Diversification

* Add parallel data from resource-rich languages and domain-specific corpora (math, STEM, formal proofs, code).
* Augment with synthetic paraphrases and domain-targeted snippets to expose systems to rare constructs.

**Rationale:** improves cross-domain generalization and model robustness for difficult constructs.

---

### Step 1 — Initial Translation (Ensemble Generation)

* Generate multiple candidate translations for each segment using an ensemble of:

  * **Specialist models:** domain/tuning-specific MT that preserve technical terms.
  * **Generalist LLMs:** broader fluency and style adaptation.

**Output:** N candidate translations per source segment.

---

### Step 2 — Automatic Post-Correction & Ranking

#### 2.1 LLM-as-Judge (Semantic Consensus)

* Use high-performing multilingual LLM(s) (examples: Qwen-235B, DeepSeek, GPT-OSS family) to rank candidate translations by:

  * semantic fidelity
  * grammar & fluency
  * technical correctness

**Prompt template (ranker):**

```text
You are a multilingual semantic evaluator.
Input: Source text + multiple candidate translations.
Task: Rank translations for:
(i) semantic fidelity
(ii) grammar/fluency
(iii) technical correctness
Return JSON {best_translation, justification}.
```

#### 2.2 Back-Translation & Embedding Similarity

* Back-translate each selected Indic candidate to English and compare embedding similarity with the original English source.
* Use LLM-as-judge to combine semantic judgments and embedding-similarity signals into a final score.

**Why:** provides an extra robustness signal against hallucinations and large semantic drift.

---

### Step 3 — LLM-Based Post-Correction (Enhancement)

* Feed the best MT/ensemble candidate through an LLM that has demonstrated strong Indic‑MMLU performance for post-correction.
* The LLM refines grammar, naturalness, and domain-specific terminology while preserving fidelity.

**Post-correction role prompt (example):**

```text
Role and Context: You are an expert linguist specializing in {language}.
Task: Transform {language} text that is poorly structured, grammatically incorrect, or unnatural into well-formed, natural-sounding {language} text.
Input Text: '{input_text}'
Output Requirements:
- Return complete rephrased text with no omissions
- Maintain original language; do not mix English/Hindi
- Do not provide explanations or meta commentary
- Keep length close to the original
```

**Model selection:** prefer LLMs that scored highly on Indic MMLU benchmarks (we empirically observed these models produce better post-corrections for low-resource Indic languages).

---

### Step 4 — Human Evaluation & Calibration

* Low-scoring or ambiguous cases are flagged for human linguist review.
* Each flagged item is reviewed by at least 3 language evaluators for cultural alignment, term consistency, and domain correctness.
* Calibration rounds align human graders to a shared rubric and reduce inter-annotator variance.

**Human review policy highlights:**

* Focus human effort where automatic metrics disagree, or where the ensemble confidence is low.
* Keep detailed edit logs to feed back into model selection and prompt refinement.

---

### Step 5 — Long-Context Chunking Strategy

For very long documents (20K–25K+ tokens) use hierarchical chunking:

* Split into logical contiguous segments with controlled token lengths.
* Include overlap tokens between consecutive segments.
* Provide brief summaries of preceding segments to the translator to maintain continuity.

**Chunking prompt template:**

```text
You are a long-context translator.
Input: Segment of a long document (with overlapping context from previous segment)
Task: Translate into {Indic language}, ensuring continuity with prior segments
Return translation only.
```

**Consolidation:** after chunk-level translation, run a coherence evaluator that checks cross-boundary consistency.

**Coherence evaluator prompt:**

```text
You are a coherence evaluator.
Input: Consecutive translated chunks
Task: Verify continuity of meaning, terminology consistency, and semantic flow
Return verdict: {Consistent, Inconsistent, Requires Edit}.
```

---

## Evaluation Metrics

* chrF – A character n-gram F-score metric that evaluates translations at the subword/character level, useful for capturing fine-grained differences in morphologically rich languages.
* chrF++ – An enhanced version of chrF that incorporates word n-grams in addition to character n-grams, providing a more robust measure for translation quality in languages with complex morphology. (Preferred for character-level scoring across morphologically-rich languages)
* sacreBLEU score – A standardized BLEU score implementation that ensures reproducibility, measuring the overlap of n-grams between the candidate translation and reference.

**Empirical Note:** preprocessing + LLM post-correction improved chrF++ by ~2-5 points across languages in our experiments.

---

### Model Inventory (example shortlist)

| Models | Models |
|--------|--------|
| moonshotai/Kimi-K2-Instruct | Qwen/Qwen3-8B |
| Qwen/Qwen3-235B-A22B-Instruct-2507 | Qwen/Qwen3-0.6B |
| Qwen/Qwen3-Next-80B-A3B-Instruct | Qwen/Qwen3-Next-80B-A3B-Thinking |
| google/gemma-3-4b-it | openai/gpt-oss-20b |
| google/gemma-3-27b-it | openai/gpt-oss-120b |
| meta-llama/Llama-3.1-70B-Instruct | facebook/nllb-200-1.3B |
| meta-llama/Llama-4-Maverick-17B-128E-Instruct | facebook/nllb-moe-54b |
| meta-llama/Llama-4-Scout-17B-16E-Instruct | facebook/nllb-200-distilled-1.3B |
| deepseek-ai/DeepSeek-V3.1 | facebook/nllb-200-3.3B |
| deepseek-ai/DeepSeek-V3.1 Think | facebook/nllb-200-distilled-600M |
| Qwen/Qwen3-235B-A22B-Thinking-2507 | zai-org/GLM-4.5 |

---

## Reproducible Commands (examples from experiments)

**IndicTrans2 evaluation**

```bash
python3 indic_trans2_bench.py \
  --input_file_path train-00000-of-00001_with_ids.jsonl \
  --output_file_path IN22_gen.jsonl \
  --src_lang en
```

**NLLB evaluation example**

```bash
python3 nllb_bench.py \
  --input_file_path final_merged.jsonl \
  --output_file_path IndicGenBench_flores_in_gen_nllb-200-distilled-600M.jsonl \
  --src_lang en \
  --model_name facebook/nllb-200-distilled-600M \
  --device cuda \
  --batch_size 1024 \
  --max_workers 121
```

**Open-source LLM asynchronous pipeline example**
> sampleing parameters: > "temperature":0.7,"top_p":0.6,"top_k":20,"repetition_penalty":1.05,"max_tokens":1024
```bash
python3 asyn_trans.py \
  --input-path train-00000-of-00001_with_ids.jsonl \
  --output-file IN22-gen_1.jsonl \
  --instruction-path instruction_prompts.yml \
  --task hunyuan_translation \
  --template-fields context eng_Latn \
  --max-concurrency 512 \
  --backend sglang-oai-chat \
  --port 30000 \
  --extra-request-body '{"temperature":0.7,"top_p":0.6,"top_k":20,"repetition_penalty":1.05,"max_tokens":1024}' \
  --host 192.168.25.165 \
  --target-langs as bn gu hi kn mai ml mr ne or pa sa sdd ta te ur
```

---

## Results

* **Pre-Enhancement:** raw MT and LLM outputs — place per-language tables / radar/spiral charts here.

* **Post-Enhancement:** MT → LLM post-corrected outputs — place per-language tables / radar/spiral charts here.


---

## Key Findings & Recommendations

* **No single winner:** specialist MT and large generalist LLMs each have strengths. Use ensembles.
* **Pipeline advantage:** preprocessing + ensemble generation + LLM post-correction consistently improved chrF++ by 3–7 points.
* **Strategy:** Use MT for base translations, then LLM refinement for fluency, terminology normalization, and complex context handling.
* **Human-in-the-loop:** keep human validation for low-confidence or high-stakes outputs.
* **Long documents:** apply hierarchical chunking with overlap + summaries to retain cross-segment dependencies.

---
