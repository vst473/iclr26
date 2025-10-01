# Translation Benchmarking & Pipeline for Indic Languages

> Ensemble-based translation and post-correction pipeline for 16 Indic languages — leveraging specialist MT models, high-performing Indic LLMs, and human validation.

---

## Abstract

This repository benchmarks open-source machine translation (MT) models and LLMs on Indic languages and documents a multi-stage pipeline that combines specialist translation systems with generalist LLMs for post-correction. The pipeline integrates automated ranking using LLMs as judges, back-translation validation, and human evaluation to produce coherent, domain-sensitive translations across STEM, mathematics, code, and general content for 16 Indic languages.

---

## Goals

1. Benchmark open-source specialist MT models and generalist LLMs across multiple parameter sizes.
2. Leverage high-scoring Indic-MMLU LLMs for translation refinement and post-correction.
3. Evaluate translation quality using **chrF**, **chrF++**, and **sacreBLEU** metrics.
4. Define a reproducible ensemble + post-correction strategy for high-fidelity Indic translations.

---

## Datasets

We evaluate on two widely used benchmarks:

* [google/IndicGenBench_flores_in](https://huggingface.co/datasets/google/IndicGenBench_flores_in)
* [ai4bharat/IN22-Gen](https://huggingface.co/datasets/ai4bharat/IN22-Gen)

The datasets cover translations into 16 Indic languages: As, Be, Gu, Ka, Hi, Mai, Ml, Mr, Ne, Or, Pa, Sa, Sd, Ta, Te, Ur.

---

## Translation Pipeline

### Step 1 — Source Data Selection

* Select parallel data from resource-rich language pairs (primarily English → Indic).
* Cover diverse domains: general text, STEM, mathematics, code, and formal documentation.
* Ensure representation across different text lengths and complexity levels.

---

### Step 2 — Specialist Model Translation & Generalist Refinement

* Initial translations generated using **two or more specialist models**:

  * **IndicTrans2:** Domain-optimized for Indic languages.
  * **NLLB variants:** Multilingual neural models (200M–54B parameters).
* Specialist outputs are refined by high-performing generalist LLMs, correcting inconsistencies and improving fluency.

**Generalist Refinement Prompt:**

```text
Role: Expert translator for {target_language}.
Input: Two translation candidates from specialist systems.
Candidate 1: {translation_1}
Candidate 2: {translation_2}
Task: Produce a refined translation that:
- Combines the strengths of both candidates
- Ensures grammatical correctness and natural flow
- Preserves technical terminology
- Maintains semantic fidelity
Output: Final refined translation only.
```

---

### Step 3 — LLM-as-Judge Semantic Consensus

Deploy 2+ multilingual LLMs (e.g., Qwen-235B, DeepSeek-V3.1, GPT-OSS-120B) to independently evaluate and rank translations based on:

**Evaluation Criteria:**

**Semantic Fidelity**: Meaning preservation, idiomatic expressions, completeness.
**Grammar & Fluency**: Correctness, readability, style.
**Technical Correctness**: Domain terminology, code/mathematical syntax, consistency.

**Prompt Template for Judges:**

```text
You are a multilingual quality evaluator for {target_language}.

Source Text (English): {source_text}
Translation Candidate: {translation}

Evaluate on a 1-10 scale:
1. Semantic Fidelity
2. Grammar & Fluency
3. Technical Correctness

Return JSON:
{
  "semantic_fidelity": <score>,
  "grammar_fluency": <score>,
  "technical_correctness": <score>,
  "overall_score": <average>,
  "justification": "<reason>",
  "flags": ["<concerns>"]
}
```

**Back-Translation Validation (Optional):**

* Back-translate low-confidence outputs to English.
* Compare embedding similarity to original.
* Flag significant semantic drift for human review.

---

### Step 4 — Human Evaluation & Finalization

**Low-Score Flagging:**

* Translations with `overall_score < 7.0` or high judge disagreement are prioritized for review.

**Expert Review Process:**

* Reviewed by 2–3 native evaluators for cultural, contextual, and domain accuracy.
* Corrections logged with rationales for feedback loops.

**Calibration & Consistency:**

* Inter-annotator agreement (Cohen’s κ > 0.75)
* Edge cases discussed to refine evaluation guidelines.

---

### Step 5 — Long-Context Translation

For documents exceeding standard context windows (>8K tokens):

**Chunking Strategy:**

* Translate segments hierarchically with 150–200 token overlaps.
* Include concise summaries of previous segments for contextual continuity.

**Post-Processing Coherence Check:**

```text
Verify:
- Terminology consistency
- Narrative flow
- Absence of contradictions
Return: {Coherent, Minor Edits, Major Revision}
```

---

## Evaluation Metrics

**chrF** – Character-level n-gram overlap; handles morphology-rich languages.
**chrF++** – Character + word n-grams; preferred for Indic languages.
**sacreBLEU** – Standard BLEU implementation; sensitive to word order.

**Empirical Observations:**

* Specialist MT → Generalist refinement: +2–4 chrF++
* * LLM-as-judge: +1–2 chrF++
* * Human validation: +3–5 chrF++
* **Total improvement:** 6–11 chrF++ points over baseline.

---

## Model Inventory

| Model | Model |
|-------|-------|
| deepseek-ai/DeepSeek-V3.1 | deepseek-ai/DeepSeek-V3.1 think |
| Qwen/Qwen3-235B-A22B-Thinking-2507 | zai-org/GLM-4.5 |
| moonshotai/Kimi-K2-Instruct | Qwen/Qwen3-235B-A22B-Instruct-2507 |
| Qwen/Qwen3-Next-80B-A3B-Instruct | google/gemma-3-4b-it |
| google/gemma-3-27b-it | meta-llama/Llama-3.1-70B-Instruct |
| meta-llama/Llama-4-Maverick-17B-128E-Instruct | meta-llama/Llama-4-Scout-17B-16E-Instruct |
| facebook/nllb-200-1.3B | facebook/nllb-moe-54b |
| facebook/nllb-200-distilled-1.3B | facebook/nllb-200-3.3B |
| facebook/nllb-200-distilled-600M | IndicTrans2 |
| tencent/Hunyuan-MT-7B | Qwen/Qwen3-Next-80B-A3B-Thinking |
| openai/gpt-oss-20b | openai/gpt-oss-120b |


---

## Reproducible Commands

**IndicTrans2 Evaluation**

```bash
python3 indic_trans2_bench.py \
  --input_file_path train.jsonl \
  --output_file_path IN22_indictrans2.jsonl \
  --src_lang en \
  --batch_size 32
```

**NLLB Batch Translation**

```bash
python3 nllb_bench.py \
  --input_file_path final_merged.jsonl \
  --output_file_path IndicGenBench_flores_nllb.jsonl \
  --src_lang en \
  --model_name facebook/nllb-200-3.3B \
  --device cuda \
  --batch_size 1024 \
  --max_workers 121
```

**Generalist LLM Refinement**

```bash
python3 async_trans.py \
  --input-path train.jsonl \
  --output-file IN22_gen_refined.jsonl \
  --instruction-path instruction_prompts.yml \
  --task generalist_refinement \
  --template-fields translation_1 translation_2 source_text \
  --max-concurrency 512 \
  --backend sglang-oai-chat \
  --port 30000 \
  --extra-request-body '{"temperature":0.7,"top_p":0.6,"top_k":20,"repetition_penalty":1.05,"max_tokens":1024}' \
  --host 192.168.25.165 \
  --target-langs as bn gu hi kn mai ml mr ne or pa sa sd ta te ur
```

**LLM-as-Judge Evaluation**

```bash
python3 llm_judge.py \
  --input-path refined_translations.jsonl \
  --output-path judge_scores.jsonl \
  --judge-models Qwen3-235B DeepSeek-V3.1 gpt-oss-120b \
  --criteria semantic_fidelity grammar_fluency technical_correctness \
  --backend vllm \
  --batch-size 64
```

---

## Results

*Baseline MT:*
<details>
  <summary>📊 [ai4bharat/IN22-Gen](https://huggingface.co/datasets/ai4bharat/IN22-Gen) </summary>

  ![English to assamese](https://github.com/user-attachments/assets/0e31225c-1f7c-4c45-9623-fe606eab9ede)
  ![English to Gujarati](https://github.com/user-attachments/assets/799c71b7-43ab-4446-a734-1b2db9541d8f)
  ![English to Hindi](https://github.com/user-attachments/assets/89fd3e1d-16c1-4515-9ed4-241339f10c0f)
  ![English to Kannada](https://github.com/user-attachments/assets/0bf1ecb8-b4f3-45f6-bdfa-befbdd3ddd6c)
  ![English to Maithili](https://github.com/user-attachments/assets/fe878c6b-3fea-4cb0-9ef3-29ecff9ae831)
  ![English to Urdu](https://github.com/user-attachments/assets/9c5af4d3-15fd-43c9-a18e-352a24dee42f)
  ![English to Telugu](https://github.com/user-attachments/assets/dc879224-7459-48e8-b450-6472c353db5b)
  ![English to Tamil](https://github.com/user-attachments/assets/e8e2881d-9aa8-41e5-9174-958dc08bc5f4)
  ![English to Sanskrit](https://github.com/user-attachments/assets/011efd7f-6ea1-437b-a6e5-400781851208)
  ![English to Panjabi](https://github.com/user-attachments/assets/7e9e4492-3474-439f-85e3-c8ef680048b0)
  ![English to Oriya](https://github.com/user-attachments/assets/701582b0-0871-4221-bca0-590f2e9ab05c)
  ![English to Nepali](https://github.com/user-attachments/assets/7684b2ca-da07-4d52-ac12-10039d65eddf)
  ![English to Marathi](https://github.com/user-attachments/assets/112981f7-6d53-422b-aaf8-15ebef488152)
  ![English to Malayalam](https://github.com/user-attachments/assets/dd799a53-b23b-4c4c-b3f0-8c98aa4c97e2)
  ![English to Bengali](https://github.com/user-attachments/assets/ebc63608-5907-4f25-a160-bfaaffb7ba6b)
  
</details>

---

<details>
  <summary>📊 [google/IndicGenBench_flores_in](https://huggingface.co/datasets/google/IndicGenBench_flores_in)</summary>

  ![Flores English to Urdu](https://github.com/user-attachments/assets/ded32975-25f8-4f45-a76c-7a429ae1d69b)  
  ![Flores English to Telugu](https://github.com/user-attachments/assets/8e7aeca2-7063-4d03-a305-90ffe09eadb1)  
  ![Flores English to Tamil](https://github.com/user-attachments/assets/b3125f26-6755-41a6-9c84-b1fba09c3d45)  
  ![Flores English to Sanskrit](https://github.com/user-attachments/assets/914d3208-cb74-4a94-a69e-231c762aed58)  
  ![Flores English to Punjabi](https://github.com/user-attachments/assets/1ffd4cf2-5112-4560-8539-0fd5d67a47ed)  
  ![Flores English to Oriya](https://github.com/user-attachments/assets/ac4a2327-d824-48db-be8f-555b8f9bbbd8)  
  ![Flores English to Nepali](https://github.com/user-attachments/assets/39c2b325-7953-4965-9b2b-18c4531f17b1)  
  ![Flores English to Malayalam](https://github.com/user-attachments/assets/76cd4da6-dfc5-4c4f-81c1-5e0692a72307)  
  ![Flores English to Maithili](https://github.com/user-attachments/assets/c91bbbf8-36f0-4194-a0b9-1a26eaf4b4d4)  
  ![Flores English to Kannada](https://github.com/user-attachments/assets/25dfb25a-724b-420f-9aa3-a49d50ce6d85)  
  ![Flores English to Hindi](https://github.com/user-attachments/assets/c12b2ffc-924a-41b5-a37c-9d219085b9a1)  
  ![Flores English to Gujarati](https://github.com/user-attachments/assets/42d46e3b-4706-4e8b-82bf-e277b3cd2297)  
  ![Flores English to Bengali](https://github.com/user-attachments/assets/8d0362ce-71a6-4f53-bf09-127f5edaef23)  
  ![Flores English to Bengali (1)](https://github.com/user-attachments/assets/1669b853-9870-408b-9c45-640ee4323c04)  
  ![Flores English to Assamese](https://github.com/user-attachments/assets/272b668d-2330-4f70-b386-ff0161c7a95e)  

</details>

---

*Post Enhancements on MT:*
<details>
  <summary>📊 Enhanced IN22 — Indic Trans2 (Enhanced with LLMs)</summary>

  ![Enhanced IN22 Indic Trans2 results using Qwen_Qwen3-Next-80B-A3B-Instruct](https://github.com/user-attachments/assets/c745042d-1b5f-4b03-bbce-803f7f4c2891)  
  ![Enhanced IN22 Indic Trans2 results using Qwen_Qwen3-235B-A22B-Instruct-2507](https://github.com/user-attachments/assets/0248271c-942d-4689-a1d9-392d2ae159be)  
  ![Enhanced IN22 Indic Trans2 results using moonshotai_Kimi-K2-Instruct](https://github.com/user-attachments/assets/d83b6a7c-4e81-4019-abc4-a71d4e292d29)  
  ![Enhanced IN22 Indic Trans2 results using deepseek-ai_DeepSeek-V3-0324](https://github.com/user-attachments/assets/5b9525fa-c2d0-4b0a-bf34-13aaa5a8bafd)  

</details>

---

<details>
  <summary>📊 Enhanced Flores — Indic Trans2 (Enhanced with LLMs)</summary>

  ![Enhanced Flores Indic Trans2 results using Qwen_Qwen3-Next-80B-A3B-Instruct](https://github.com/user-attachments/assets/ade604c1-dd02-492f-ab1c-956a19ce7cd2)  
  ![Enhanced Flores Indic Trans2 results using Qwen_Qwen3-235B-A22B-Instruct-2507](https://github.com/user-attachments/assets/26b5c058-7640-4e91-a09d-7392ff0cae89)  
  ![Enhanced Flores Indic Trans2 results using moonshotai_Kimi-K2-Instruct](https://github.com/user-attachments/assets/9ee0f57d-86d2-40bf-8e30-ed92a399e0d9)  

</details>

---

## Key Insights & Best Practices

**Specialist + Generalist Synergy:**

* Specialist MT ensures structural and terminology accuracy.
* Generalist LLMs improve fluency and context understanding.

**Multi-Judge Consensus:**

* Ensemble of 2–3 judges reduces bias and aligns better with human judgments.

**Operational Recommendations:**

* Specialist MT + generalist refinement for all pipelines.
* Human-in-the-loop for low-score or domain-critical content.
* Long-context documents: hierarchical chunking + overlap + summary.

**Performance Gains:**

* Total improvement: **6–11 chrF++ points** across languages.
* High-resource languages benefit more; low-resource require higher human validation.

---
