# Translation Benchmarking: Indic Languages

## Overview

This repository provides the pipeline, evaluation, and experimental results of benchmarking **open-source machine translation models** and **LLMs** on Indic languages.
We focus on evaluating their translation capability, enhancement strategies, and performance across Indian languages.

We chose two widely used benchmarks:

* [google/IndicGenBench_flores_in](https://huggingface.co/datasets/google/IndicGenBench_flores_in)
* [ai4bharat/IN22-Gen](https://huggingface.co/datasets/ai4bharat/IN22-Gen)

The goal is to:

1. Benchmark **open-source machine translation models** and **LLMs** (small, medium, large parameter sizes).
2. Enhance machine translation outputs using **high-scoring Indic MMLU open-source LLMs**.
3. Evaluate and compare **pre-enhancement** vs **post-enhancement** results.
4. Recommend a strategy for **coherent, high-quality Indic translations**.

---

## Pipeline Steps

1. **Download the benchmark datasets**:

   * `google/IndicGenBench_flores_in`
   * `ai4bharat/IN22-Gen`

2. **Evaluate open-source MT models & LLMs**

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

3. **Enhancement Stage**

   * Pass MT outputs through an open-source LLM with strong **Indic MMLU** performance.

4. **Evaluate post-enhanced results**

   * Using metrics: **chrF**, **chrF++**, and **sacBLEU**.

---

## Model Setup & Evaluation

### 1. IndicTrans2 Evaluation

Setup IndicTrans2 and run inference:

```bash
python3 indic_trans2_bench.py \
  --input_file_path train-00000-of-00001_with_ids.jsonl \
  --output_file_path IN22_gen.jsonl \
  --src_lang en
```

---

### 2. NLLB Models Evaluation

We benchmarked multiple NLLB variants (distilled and larger). Example command:

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

---

### 3. Open-Source LLM Evaluation

We evaluated open-source LLMs of different sizes using an asynchronous translation pipeline:

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

## Benchmark Metrics

We use the following standard translation metrics:

* **chrF**
* **chrF++**
* **sacBLEU**

---

## Results

### Pre-Enhancement Results

*(Placeholder: Insert results table/plots for raw model outputs)*
<details>
  <summary>📊 Pre-Enhancement IN22</summary>

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




### Post-Enhancement Results

*(Placeholder: Insert results table/plots after enhancement with Indic MMLU-strong LLMs)*

---

## Observations & Recommendations

* **Machine translation models** perform well on high-resource languages but struggle with **low-resource Indic languages**.
* **Open-source LLMs** show potential in improving fluency and coherence, especially post-enhancement.
* The recommended strategy is to use **MT models for base translation** followed by **LLM refinement** for achieving **coherently well-translated outputs**.

---

