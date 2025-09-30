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

   * IndicTrans2
   * NLLB (distilled & large variants)
   * Open-source LLMs

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

---

### Post-Enhancement Results

*(Placeholder: Insert results table/plots after enhancement with Indic MMLU-strong LLMs)*

---

## Observations & Recommendations

* **Machine translation models** perform well on high-resource languages but struggle with **low-resource Indic languages**.
* **Open-source LLMs** show potential in improving fluency and coherence, especially post-enhancement.
* The recommended strategy is to use **MT models for base translation** followed by **LLM refinement** for achieving **coherently well-translated outputs**.

---

