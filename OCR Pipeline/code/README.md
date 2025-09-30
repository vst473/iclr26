# Asynchronous OCR Inference using Docker and Python

This repository provides a setup to run **asynchronous OCR inference** using Python inside a Docker container. The pipeline supports instruction-based prompts and high-concurrency inference via `vllm`.

## Running Model Servers with Docker

You can launch model servers using either **sglang** or **vLLM** inside Docker. Below are the generic command templates.

### Run sglang Server

```bash
sudo docker run --gpus all --shm-size <SIZE> \
  -v /path/to/cache:/root/.cache/huggingface/hub \
  --ipc=host --network=host --privileged -it \
  -e CUDA_VISIBLE_DEVICES=<GPU_IDS> \
  <sglang_image>:latest \
  bash -c "python3 -m pip install --upgrade huggingface_hub && \
           huggingface-cli login --token <HF_TOKEN> && \
           TORCH_CUDA_ARCH_LIST=<ARCH> python3 -m sglang.launch_server \
               --model-path <MODEL_PATH> \
               --device cuda \
               --tp <TENSOR_PARALLEL> \
               --dp <DATA_PARALLEL> \
               --mem-fraction-static <MEM_FRACTION> \
               --attention-backend <BACKEND> \
               --enable-torch-compile \
               --torch-compile-max-bs <MAX_BATCH> \
               --host 0.0.0.0 \
               --port <PORT> \
               --enable-metrics"
```

OR

### Run vLLM Server

```bash
sudo docker run --gpus all --shm-size <SIZE> \
  -v /path/to/cache:/root/.cache/huggingface/hub \
  --ipc=host --network=host --privileged -it \
  -e CUDA_VISIBLE_DEVICES=<GPU_IDS> \
  -e HF_TOKEN=<HF_TOKEN> \
  <vllm_image>:latest \
      --model <MODEL_PATH> \
      --tensor-parallel-size <TENSOR_PARALLEL> \
      --data-parallel-size <DATA_PARALLEL> \
      --dtype <DTYPE> \
      --max-model-len <MAX_LEN> \
      --host 0.0.0.0 \
      --port <PORT>
```

---


## Run Docker Container

The following command runs a Docker container, mounts your local directory, and installs the required Python packages:

```bash
sudo docker run --rm --ipc=host --network=host \
  -v /path/to/local/dir:/container/dir \
  -it python:latest \
  /bin/bash -c "pip install aiohttp aiofiles transformers python-magic && /bin/bash"
```

* `/path/to/local/dir` → Replace with your local directory containing the scripts and data
* `/container/dir` → Directory inside the container where your files will be accessible

This command opens an interactive bash session inside the container after installing dependencies.

---

## Run Asynchronous Inference

Once inside the Docker container, execute the OCR inference script:

```bash
python3 /container/dir/inference_scripts/async_infer.py \
  --input-path /container/dir/input_data.jsonl \
  --output-file /container/dir/output/output_data.jsonl \
  --instruction-path /container/dir/inference_scripts/instruction_prompts.yml \
  --task ocr \
  --max-concurrency 100 \
  --extra-request-body '{"temperature":0.7,"top_p":0.9,"top_k":50,"repetition_penalty":1.2,"min_p":0.01}' \
  --backend vllm-chat
```

### Arguments

| Argument               | Description                                                                        |
| ---------------------- | ---------------------------------------------------------------------------------- |
| `--input-path`         | Path to input JSONL file containing documents for OCR                              |
| `--output-file`        | Path where the processed OCR output will be saved                                  |
| `--instruction-path`   | Path to instruction prompts YAML for OCR guidance                                  |
| `--task`               | Task type, e.g., `ocr`                                                             |
| `--max-concurrency`    | Maximum number of concurrent inference requests                                    |
| `--extra-request-body` | JSON string with inference hyperparameters (`temperature`, `top_p`, `top_k`, etc.) |
| `--backend`            | Backend engine for inference (`vllm-chat`)                                         |

---

This README provides a **reproducible guide** for running high-concurrency OCR inference in a containerized environment.
