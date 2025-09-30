
# Indic MMLU Dataset

## Overview
This dataset was created by translating the CAIS/MMLU test dataset into various Indic languages. Enhancements were made using an open-source large language model (LLM) to improve translation quality.

## Steps Followed
1. Take the original CAIS/MMLU test dataset (https://huggingface.co/datasets/cais/mmlu/viewer).
2. Translate the dataset using a machine translation model (IndicTrans2).
3. Enhance translations using an open-source LLM with specified instructions and concurrency.
4. Generate embeddings for both the original English and Indic translated/enhanced versions.
5. Calculate cosine similarity between the embeddings of the original and translated datasets.
6. Judge translation enhancement based on math correctness, coherence, and linguistic expert ratings.

## Pseudo Code for Indic MMLU Creation Algorithm

Here is the pseudo code outlining the high-level algorithm used to create the Indic MMLU dataset:

```
Algorithm: Create_Indic_MMLU_Dataset

Input: Original English MMLU dataset file
Output: Enhanced Indic MMLU datasets, embeddings, similarity scores, evaluation ratings

1. Procedure Translate_Dataset(input_file, tgt_lang, src_lang)
2. Procedure Enhance_Translations(translated_file, instruction_path, task, ...)
3. Procedure Generate_Embeddings(input_file, output_file, server_url, concurrency)
4. Procedure Calculate_Cosine_Similarity(input_dir, output_dir, english_file)
5. Procedure Judge_Enhancements(input_file, output_file, instruction_path, task, ...)

6. Main:
   For each target Indic language:
     - Translate dataset
     - Enhance translations
     - Generate embeddings
     - Calculate cosine similarity
     - Judge enhancements
   Aggregate and report summaries
```


## Commands Used

- **Translate:**
  ```bash
  python3 trans_mmlu.py --input_file_path test-00000-of-00001_with_ids.jsonl \
    --output_file_path mmlu_kn_in.jsonl --tgt_lang "kn" --src_lang "en"```

* **Enhance translations:**

  ```bash
  python3 async_infr.py --input-path mmlu_as_in.jsonl \
    --output-file ds_enhance_eval_hi.jsonl \
    --instruction-path instruction_prompts.yml \
    --task deepseek_enhance_instruct \
    --template-fields 'og_question' 'og_choices' 'question' 'answer' 'Hindi' \
    --max-concurrency 4096 \
    --backend sglang-oai-chat \
    --extra-request-body '{"temperature":0.7,"top_p":0.9,"top_k":50,"max_tokens":1024}' \
    --port 30000
  ```

* **Get embeddings:**

  ```bash
  python get_embeddings.py --input-file test-00000-of-00001_with_ids.jsonl \
    --output-file mmlu_qwen3_embd.jsonl \
    --server-url http://127.0.0.1:30000 \
    --concurrency 4096
  ```

* **Calculate cosine similarity:**

  ```bash
  python3 calculate_cosine_similarity.py --input_dir deepseek_enhance \
    --output_dir mmmlu/similarity/cosine/enhance/deepseek_enhance \
    --english_file mmlu_eng.jsonl
  ```

* **Judge translation:**

  ```bash
  python3 async_infr.py --input-path mmlu_as_in.jsonl \
    --output-file ds_enhance_eval_as_qwen3_instruct.jsonl \
    --instruction-path instruction_prompts.yml \
    --task rate_translated_text_linguist \
    --template-fields 'og_question' 'og_choices' 'enhanced_text' 'Assamese' \
    --max-concurrency 4096 \
    --backend sglang-oai-chat \
    --extra-request-body '{"temperature":0.7,"top_p":0.9,"top_k":50,"max_tokens":1024}' \
    --port 30000
  ```

## Summary of Teacher Ratings

Average teacher ratings for translation enhancement across languages such as Nepali, Telugu, Oriya, Punjabi, Assamese, Sanskrit, Kannada, Sindhi, Gujarati, Marathi, Tamil, Malayalam, and Maithili.

* Ratings cover **language quality, math correctness, coherence, and linguistic aspects**.

| Language  | Avg Maths Rate | Avg Coherence Rate | Avg Linguist Rate | Total Records |
|-----------|----------------|--------------------|-------------------|---------------|
| Nepali    | 9.473793       | 9.331505           | 8.402293          | 14042         |
| Telugu    | 9.220837       | 8.975929           | 7.986256          | 14042         |
| Oriya     | 9.264421       | 8.944096           | 8.220268          | 14042         |
| Punjabi   | 9.348027       | 9.049138           | 8.253454          | 14042         |
| Assamese  | 9.147201       | 9.039097           | 8.133243          | 14042         |
| Sanskrit  | 8.338485       | 8.112520           | 6.390115          | 14042         |
| Kannada   | 9.322746       | 9.111736           | 8.131819          | 14042         |
| Sindhi    | 2.070645       | 3.109956           | 2.478137          | 14042         |
| Gujrati   | 9.260647       | 9.206523           | 8.368965          | 14042         |
| Marathi   | 9.301951       | 9.342045           | 8.381926          | 14042         |
| Tamil     | 9.047856       | 9.042943           | 8.071144          | 14042         |
| Malayalam | 9.134952       | 8.998148           | 7.956488          | 14042         |
| Maithili  | 9.263282       | 9.118146           | 8.218772          | 14042         |


## Cosine Similarity Scores

Cosine similarity was calculated between the original English and enhanced/translated Indic MMLU embeddings.

* **Mean similarity** ranges from ~0.76 to 0.85 depending on the language.

| Language               | Mean Similarity | Std Similarity | Min Similarity | Max Similarity |
|-------------------------|-----------------|----------------|----------------|----------------|
| mmlu_as_in_qwen3_embed  | 0.8045          | 0.0597         | 0.4644         | 1              |
| mmlu_bn_in_qwen3_embed  | 0.8133          | 0.0504         | 0.5276         | 1              |
| mmlu_gu_in_qwen3_embed  | 0.8211          | 0.0535         | 0.4016         | 1              |
| mmlu_hi_in_qwen3_embed  | 0.8472          | 0.0489         | 0.3688         | 1              |
| mmlu_kn_in_qwen3_embed  | 0.8106          | 0.0565         | 0.4660         | 1              |
| mmlu_mai_in_qwen3_embed | 0.8226          | 0.0504         | 0.4936         | 1              |
| mmlu_ml_in_qwen3_embed  | 0.8158          | 0.0531         | 0.5109         | 1              |
| mmlu_mr_in_qwen3_embed  | 0.8129          | 0.0513         | 0.5281         | 1              |
| mmlu_ne_in_qwen3_embed  | 0.8242          | 0.0502         | 0.4802         | 1              |
| mmlu_or_in_qwen3_embed  | 0.8159          | 0.0555         | 0.4684         | 1              |
| mmlu_pa_in_qwen3_embed  | 0.8246          | 0.0516         | 0.5450         | 1              |
| mmlu_sa_in_qwen3_embed  | 0.7912          | 0.0574         | 0.4981         | 1              |
| mmlu_sdd_in_qwen3_embed | 0.7646          | 0.0735         | 0.3633         | 0.9676         |
| mmlu_ta_in_qwen3_embed  | 0.7964          | 0.0559         | 0.5242         | 1              |
| mmlu_te_in_qwen3_embed  | 0.8006          | 0.0524         | 0.5379         | 1              |


## Model Evaluation

### Model Evaluation Methodology

We evaluated ~24 open-source LLMs across 16 languages using the Indic MMLU benchmark. The evaluation workflow was as follows:

1. **Environment Setup**

   * Used the Docker image `vllm/vllm-openai:latest` for a reproducible inference environment.
   * Installed the evaluation framework:

     ```bash
     pip install lm-eval
     ```

2. **Model Evaluation**
   Each model was evaluated using the `lm-eval` framework on the Indic MMLU dataset:

   ```bash
   bash lm-eval-llm.sh <path_to_model_snapshot> <path_to_mmmlu_dataset>
   ```

   Example:

   ```bash
   bash lm-eval-llm.sh /workspace/bds/glm/hub/models--zai-org--GLM-4.5/snapshots/cbb2c7cfb52fa128a9660cb1a7a78e017899e115 /workspace/bds/mmmlu-benchmark/glm
   ```

3. **Metrics Computed**

   * **Accuracy per Language**: Each model’s performance on the test set for each Indic language.
   * **Top-Performing Models**: Identified by sorting model scores for each language.
   * **Overall Performance**: Calculated as the average score across all languages.

4. **Aggregation and Visualization**

   * Scores were aggregated to generate language-wise summaries.
   * **Radar plots** were created for each language to visually represent the relative performance of all evaluated models.

## Model Evaluation Table

The full table contains evaluation scores for ~24 models across 16 languages (including English). To simulate a landscape format split across two "pages" in markdown, it is divided as follows:

### 📊 Indic MMLU Results 

| ![Assamese](assets/assamese.png) | ![Bengali](assets/bengali.png) |
|----------------------------------|--------------------------------|
| ![English](assets/English.png)   | ![Gujarati](assets/Gujrati.png) |
| ![Hindi](assets/Hindi.png)       | ![Kannada](assets/kannada.png) |
| ![Maithili](assets/maithili.png) | ![Malayalam](assets/Malayalam.png) |
| ![Marathi](assets/marathi.png)   | ![Nepali](assets/Nepali.png)   |
| ![Oriya](assets/oriya.png)       | ![Punjabi](assets/Punjabi.png) |
| ![Sanskrit](assets/sanskrit.png) | ![Sindhi](assets/sindhi.png)   |
| ![Tamil](assets/Tamil.png)       | ![Telugu](assets/Telugu.png)   |


## Contact

For any queries or collaboration, please contact the dataset maintainer.

