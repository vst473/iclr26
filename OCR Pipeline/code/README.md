# Asynchronous OCR Inference using Docker and Python

This repository provides a setup to run **asynchronous OCR inference** using Python inside a Docker container. The pipeline supports instruction-based prompts and high-concurrency inference via `vllm`.

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
