#!/usr/bin/env python3

import asyncio
import base64
import io
import json
import os
import resource
import sys
import time
import yaml
import magic
import re
from pathlib import Path
from argparse import ArgumentParser
from datetime import timedelta
import aiohttp
import aiofiles
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer
from collections import defaultdict

global args

# Mapping: code -> display name (same as original)
LANG_CODE_MAP = {
    'as': 'Assamese',
    'bn': 'Bengali',
    'gu': 'Gujarati',
    'hi': 'Hindi',
    'kn': 'Kannada',
    'mai': 'Maithili',
    'ml': 'Malayalam',
    'mr': 'Marathi',
    'ne': 'Nepali',
    'or': 'Oriya',
    'pa': 'Punjabi',
    'sa': 'Sanskrit',
    'sdd': 'Sindhi_Deva',
    'ta': 'Tamil',
    'te': 'Telugu',
    'ur': 'Urdu',
}

# Inverted map: lowercase display name -> code
NAME_TO_CODE = {v.lower(): k for k, v in LANG_CODE_MAP.items()}

def _create_bench_client_session():
    BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60  # 6 hours
    BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2  # 10 MB
    aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
    return aiohttp.ClientSession(timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES)

def remove_prefix(text, prefix):
    return text[len(prefix):] if text.startswith(prefix) else text

def remove_suffix(text, suffix):
    return text[:-len(suffix)] if text.endswith(suffix) else text

def get_auth_headers():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    else:
        return {}

# --- Async request functions (kept from original) ---

async def async_request_trt_llm(request_func_input, pbar=None):
    api_url = request_func_input["api_url"]
    assert api_url.endswith("generate_stream")

    async with _create_bench_client_session() as session:
        payload = {
            "accumulate_tokens": True,
            "text_input": request_func_input["prompt"],
            "stream": True,
            **request_func_input["extra_request_body"],
        }

        output = {
            "id": request_func_input.get("id"),
            "generated_text": "",
        }

        try:
            async with session.post(url=api_url, json=payload) as response:
                if response.status != 200:
                    pass
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")
                    data = json.loads(chunk)
                    output["generated_text"] += data.get("text_output", "")
        except Exception:
            pass

    if pbar:
        pbar.update(1)
    return output

async def async_request_openai_completions(request_func_input, pbar=None):
    api_url = request_func_input["api_url"]
    assert api_url.endswith("completions"), "OpenAI Completions API URL must end with 'completions'."

    prompt = request_func_input["prompt"]
    async with _create_bench_client_session() as session:
        payload = {
            "model": request_func_input["model"],
            "prompt": prompt,
            "best_of": 1,
            "stream": args.enable_stream,
            "ignore_eos": args.ignore_eos,
            **request_func_input["extra_request_body"],
        }

        headers = get_auth_headers()
        output = {
            "id": request_func_input.get("id"),
            "generated_text": "",
        }

        generated_text = ""
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    pass
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                    if chunk == "[DONE]":
                        pass
                    else:
                        data = json.loads(chunk)
                        if data.get("choices") and data["choices"][0].get("text"):
                            generated_text += data["choices"][0]["text"]
            output["generated_text"] = generated_text
        except Exception:
            pass

    if pbar:
        pbar.update(1)
    return output

async def async_request_openai_chat_completions(request_func_input, pbar=None, max_retries=3, retry_delay=2):
    """Makes a request to the OpenAI Chat Completions API with retry support."""
    api_url = request_func_input["api_url"]
    assert api_url.endswith("chat/completions"), "OpenAI Chat Completions API URL must end with 'chat/completions'."

    messages = [{"role": "user", "content": request_func_input["prompt"]}]
    headers = get_auth_headers()

    for attempt in range(1, max_retries + 1):
        async with _create_bench_client_session() as session:
            payload = {
                "model": request_func_input["model"],
                "messages": messages,
                "stream": args.enable_stream,
                "ignore_eos": args.ignore_eos,
                **request_func_input["extra_request_body"],
            }

            output = {"id": request_func_input.get("id"), "generated_text": ""}
            generated_text = ""

            try:
                async with session.post(url=api_url, json=payload, headers=headers) as response:
                    if response.status != 200:
                        continue

                    if not args.enable_stream:
                        response_json = await response.json()
                        generated_text = response_json["choices"][0]["message"]["content"]
                    else:
                        async for chunk_bytes in response.content:
                            chunk_bytes = chunk_bytes.strip()
                            if not chunk_bytes:
                                continue
                            chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                            if chunk == "[DONE]":
                                continue
                            data = json.loads(chunk)
                            delta = data.get("choices", [{}])[0].get("delta", {})
                            content = delta.get("content", "")
                            if content:
                                generated_text += content

                    output["generated_text"] = generated_text.strip()
                    if output["generated_text"]:
                        break
            except Exception:
                pass

            if not output["generated_text"] and attempt < max_retries:
                await asyncio.sleep(retry_delay)

    if pbar:
        pbar.update(1)
    return output

async def async_request_sglang_generate(request_func_input, pbar=None):
    api_url = request_func_input["api_url"]
    prompt = request_func_input["prompt"]

    async with _create_bench_client_session() as session:
        payload = {
            "text": prompt,
            "sampling_params": {
                **request_func_input["extra_request_body"],
                "ignore_eos": args.ignore_eos,
            },
            "stream": args.enable_stream,
        }

        if request_func_input.get("image_data"):
            payload["image_data"] = request_func_input["image_data"]

        headers = get_auth_headers()
        output = {
            "id": request_func_input.get("id"),
            "prompt": request_func_input.get("prompt"),
            "generated_text": "",
        }

        generated_text = ""
        try:
            async with session.post(url=api_url, json=payload, headers=headers) as response:
                if response.status != 200:
                    pass
                async for chunk_bytes in response.content:
                    chunk_bytes = chunk_bytes.strip()
                    if not chunk_bytes:
                        continue
                    chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
                    if chunk == "[DONE]":
                        pass
                    else:
                        data = json.loads(chunk)
                        if "text" in data and data["text"]:
                            generated_text = data["text"]
            output["generated_text"] = generated_text
        except Exception:
            pass

    if pbar:
        pbar.update(1)
    return output

ASYNC_REQUEST_FUNCS = {
    "sglang": async_request_sglang_generate,
    "sglang-native": async_request_sglang_generate,
    "sglang-oai": async_request_openai_completions,
    "sglang-oai-chat": async_request_openai_chat_completions,
    "vllm": async_request_openai_completions,
    "vllm-chat": async_request_openai_chat_completions,
    "lmdeploy": async_request_openai_completions,
    "lmdeploy-chat": async_request_openai_chat_completions,
    "trt": async_request_trt_llm,
}

# --- I/O helpers ---

def load_jsonl(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            records.append(obj)
    return records

def load_parquet(path):
    import pandas as pd
    df = pd.read_parquet(path)
    return df.to_dict(orient="records")

def load_task_template(yaml_path, task):
    if not os.path.isfile(yaml_path):
        raise FileNotFoundError(f"YAML file not found: {yaml_path}")

    with open(yaml_path, "r", encoding="utf-8") as f:
        templates = yaml.safe_load(f)

    if task not in templates:
        raise KeyError(f"Task '{task}' not found in YAML. Available tasks: {list(templates.keys())}")

    return templates[task]

def get_by_path(data, path):
    import re
    parts = path.split(".")
    cur = data
    for p in parts:
        if p.startswith("[?"):
            m = re.match(r"\[\?([^=]+)==(.+)\]", p)
            if not m:
                raise ValueError(f"Invalid filter syntax: {p}")
            key, val = m.group(1), m.group(2)
            val = val.strip('\"\'')
            if isinstance(cur, list):
                cur = next((x for x in cur if x.get(key) == val), None)
            else:
                cur = None
        elif p.isdigit():
            cur = cur[int(p)] if isinstance(cur, list) else None
        else:
            cur = cur.get(p) if isinstance(cur, dict) else None
        if cur is None:
            break
    return cur

def fill_instruction(template, record, template_fields, english_text, translation, target_lang, task_type="evaluate"):
    """
    Fill instruction template for enhancement task
    template_fields expected to be: [english_field, translation_field, language_field]
    """
    if task_type == "evaluate":
        # For evaluation task: English text, target language name, current translation
        values = [english_text, target_lang, translation]
    elif task_type == "enhance":
        # For enhancement task: English text, target language name, current translation
        values = [english_text, target_lang, translation]
    else:
        # Fallback to original template filling logic
        if not template_fields:
            return template
        values = [get_by_path(record, field) for field in template_fields]
        if any(v is None for v in values):
            raise ValueError(f"Missing required field(s) for template_fields: {template_fields}")

    return template.format(*values)

def normalize_target_langs(raw_list):
    if not raw_list:
        return []

    normalized = []
    for item in raw_list:
        it = item.strip()
        lower = it.lower()

        # if already a known code
        if lower in LANG_CODE_MAP:
            normalized.append(lower)
            continue

        # if matches a known display name
        if lower in NAME_TO_CODE:
            normalized.append(NAME_TO_CODE[lower])
            continue

        # try common variants: remove punctuation/spaces
        compact = re.sub(r"[^a-zA-Z]", "", lower)
        found = None
        for name, code in NAME_TO_CODE.items():
            if re.sub(r"[^a-zA-Z]", "", name) == compact:
                found = code
                break

        if found:
            normalized.append(found)
            continue

        # fallback: accept item as-is (but warn)
        normalized.append(it)

    return normalized

# --- Enhanced prompt preparation for evaluation and enhancement ---

def prepare_enhancement_prompts(args):
    """Prepare prompts for translation enhancement task"""
    records = []
    ext = os.path.splitext(args.input_path)[1].lower()

    if ext in [".json", ".jsonl"]:
        records = load_jsonl(args.input_path)
    elif ext in [".parquet"]:
        records = load_parquet(args.input_path)
    else:
        raise ValueError("Cannot detect file type and unknown extension!")

    # Load both evaluation and enhancement templates
    templates = {}
    with open(args.instruction_path, "r", encoding="utf-8") as f:
        all_templates = yaml.safe_load(f)

    if "evaluate_translation" not in all_templates:
        raise KeyError("Task 'evaluate_translation' not found in YAML templates")
    if "enhance_translation" not in all_templates:
        raise KeyError("Task 'enhance_translation' not found in YAML templates")

    templates["evaluate"] = all_templates["evaluate_translation"]
    templates["enhance"] = all_templates["enhance_translation"]

    input_requests = []
    codes = normalize_target_langs(args.target_langs) if args.target_langs else list(LANG_CODE_MAP.keys())

    for record in records:
        record_id = record.get("id")
        english_text = record.get("english_og")
        translations = record.get("translations", {})

        if not english_text or not translations:
            continue

        # Create evaluation and enhancement requests for each language
        for code in codes:
            if code not in translations:
                continue

            translation = translations[code]
            target_lang_name = LANG_CODE_MAP.get(code, code)

            # Evaluation request
            eval_prompt = fill_instruction(
                templates["evaluate"], record, args.template_fields, 
                english_text, translation, target_lang_name, "evaluate"
            )

            input_requests.append({
                "id": f"{record_id}_{code}_eval",
                "prompt": eval_prompt,
                "record": record,
                "task_type": "evaluate",
                "lang_code": code,
                "original_translation": translation
            })

            # Enhancement request (will be conditionally executed)
            enhance_prompt = fill_instruction(
                templates["enhance"], record, args.template_fields,
                english_text, translation, target_lang_name, "enhance"
            )

            input_requests.append({
                "id": f"{record_id}_{code}_enhance", 
                "prompt": enhance_prompt,
                "record": record,
                "task_type": "enhance",
                "lang_code": code,
                "original_translation": translation
            })

    print(f"Prepared {len(input_requests)} enhancement requests ({len(input_requests)//2} evaluations + {len(input_requests)//2} potential enhancements)")
    return input_requests, 0

async def get_request(input_requests, request_rate):
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request
        if request_rate == float("inf"):
            continue
        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)

# --- Enhanced inference and aggregation logic for translation enhancement ---

async def infer_enhancement(backend, api_url, base_url, model_id, input_requests, already_done, request_rate, max_concurrency, disable_tqdm, extra_request_body):
    """Enhanced inference function for translation enhancement task"""

    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            result = await request_func(request_func_input=request_func_input, pbar=pbar)
        else:
            async with semaphore:
                result = await request_func(request_func_input=request_func_input, pbar=pbar)

        # Preserve original request metadata
        result["task_type"] = request_func_input.get("task_type")
        result["lang_code"] = request_func_input.get("lang_code") 
        result["original_translation"] = request_func_input.get("original_translation")
        result["record"] = request_func_input.get("record")
        return result

    if "sglang" in backend:
        requests.post(base_url + "/flush_cache", headers=get_auth_headers())
        time.sleep(1.0)

    pbar = None if disable_tqdm else tqdm(total=len(input_requests), initial=already_done)

    # Separate evaluation and enhancement requests
    eval_requests = [req for req in input_requests if req["task_type"] == "evaluate"]
    enhance_requests = {req["id"].replace("_enhance", "_eval"): req for req in input_requests if req["task_type"] == "enhance"}

    # Process evaluation requests first
    eval_tasks = []
    for request in eval_requests:
        request_func_input = {
            "model": model_id,
            "id": request.get("id"),
            "prompt": request.get("prompt"),
            "api_url": api_url,
            "image_data": request.get("image_data"),
            "extra_request_body": extra_request_body,
            "task_type": request.get("task_type"),
            "lang_code": request.get("lang_code"),
            "original_translation": request.get("original_translation"),
            "record": request.get("record"),
        }

        eval_tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )

    # Collect evaluation results and determine which translations need enhancement
    enhancement_needed = {}
    enhancement_log = []

    for task in asyncio.as_completed(eval_tasks):
        result = await task
        gen = result.get("generated_text", "").strip().upper()

        eval_id = result.get("id")
        if not eval_id:
            continue

        # Parse evaluation result
        needs_enhancement = "ENHANCE" in gen or "IMPROVE" in gen or "BAD" in gen
        enhancement_needed[eval_id] = needs_enhancement

        # Extract record info for logging
        record_parts = eval_id.replace("_eval", "").rsplit("_", 1)
        if len(record_parts) == 2:
            orig_id, lang_code = record_parts
        else:
            continue

        record = result.get("record", {})
        english_text = record.get("english_og", "")
        original_translation = result.get("original_translation", "")

        # Log enhancement decision
        log_entry = {
            "id": f"{orig_id}_{lang_code}",
            "english_org": english_text,
            lang_code: original_translation,
            "enhanced": "yes" if needs_enhancement else "no"
        }
        enhancement_log.append(log_entry)

    # Process enhancement requests for translations that need improvement
    enhance_tasks = []
    for eval_id, needs_enhance in enhancement_needed.items():
        if needs_enhance and eval_id in enhance_requests:
            enhance_req = enhance_requests[eval_id]
            request_func_input = {
                "model": model_id,
                "id": enhance_req.get("id"),
                "prompt": enhance_req.get("prompt"),
                "api_url": api_url,
                "image_data": enhance_req.get("image_data"),
                "extra_request_body": extra_request_body,
                "task_type": enhance_req.get("task_type"),
                "lang_code": enhance_req.get("lang_code"),
                "original_translation": enhance_req.get("original_translation"),
                "record": enhance_req.get("record"),
            }

            enhance_tasks.append(
                asyncio.create_task(
                    limited_request_func(request_func_input=request_func_input, pbar=pbar)
                )
            )

    # Collect enhanced translations
    enhanced_translations = {}
    for task in asyncio.as_completed(enhance_tasks):
        result = await task
        gen = result.get("generated_text", "").strip()

        enhance_id = result.get("id")
        if not enhance_id or not gen:
            continue

        # Parse enhanced translation
        eval_id = enhance_id.replace("_enhance", "_eval")
        enhanced_translations[eval_id] = gen

    # Build final results
    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Write enhancement log
    enhancements_file = args.output_file.replace(".jsonl", "_enhancements_done.jsonl")
    with open(enhancements_file, "w", encoding="utf-8") as f:
        for entry in enhancement_log:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Build enhanced translations output
    records_dict = {rec["id"]: rec for rec in [req["record"] for req in eval_requests]}
    enhanced_records = []

    for record_id, record in records_dict.items():
        enhanced_record = {
            "id": record_id,
            "english_og": record.get("english_og"),
            "translations": {}
        }

        # For each language, use enhanced translation if available, otherwise original
        original_translations = record.get("translations", {})
        for lang_code, orig_translation in original_translations.items():
            eval_id = f"{record_id}_{lang_code}_eval"

            if eval_id in enhanced_translations:
                # Use enhanced translation
                enhanced_record["translations"][lang_code] = enhanced_translations[eval_id]
            else:
                # Keep original translation
                enhanced_record["translations"][lang_code] = orig_translation

        enhanced_records.append(enhanced_record)

    # Write enhanced translations
    enhanced_file = args.output_file.replace(".jsonl", "_enhanced_trans.jsonl")
    with open(enhanced_file, "w", encoding="utf-8") as f:
        for record in enhanced_records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    if pbar is not None:
        pbar.close()

    print(f"Enhancement log written to: {enhancements_file}")
    print(f"Enhanced translations written to: {enhanced_file}")

    # Print summary
    total_enhancements = sum(1 for entry in enhancement_log if entry["enhanced"] == "yes")
    print(f"Total translations evaluated: {len(enhancement_log)}")
    print(f"Total translations enhanced: {total_enhancements}")

# --- Remaining helpers (kept from original) ---

def check_chat_template(model_path):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        return "chat_template" in tokenizer.init_kwargs
    except Exception as e:
        print(f"Fail to load tokenizer config with error={e}")
        return False

def set_ulimit(target_soft_limit=65535):
    resource_type = resource.RLIMIT_NOFILE
    current_soft, current_hard = resource.getrlimit(resource_type)
    if current_soft < target_soft_limit:
        try:
            resource.setrlimit(resource_type, (target_soft_limit, current_hard))
        except ValueError as e:
            print(f"Fail to set RLIMIT_NOFILE: {e}")

def run_inference(args_):
    global args
    args = args_

    set_ulimit()

    extra_request_body = {}
    if args.extra_request_body:
        extra_request_body = json.loads(args.extra_request_body)

    if args.port is None:
        args.port = {
            "sglang": 30000,
            "sglang-native": 30000,
            "sglang-oai": 30000,
            "lmdeploy": 23333,
            "vllm": 8000,
            "trt": 8000,
        }.get(args.backend, 30000)

    model_url = (
        f"{args.base_url}/v1/models"
        if args.base_url
        else f"http://{args.host}:{args.port}/v1/models"
    )

    if args.backend in ["sglang", "sglang-native"]:
        api_url = (
            f"{args.base_url}/generate"
            if args.base_url
            else f"http://{args.host}:{args.port}/generate"
        )
    elif args.backend in ["sglang-oai", "vllm", "lmdeploy"]:
        api_url = (
            f"{args.base_url}/v1/completions"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/completions"
        )
    elif args.backend in ["sglang-oai-chat", "vllm-chat", "lmdeploy-chat"]:
        api_url = (
            f"{args.base_url}/v1/chat/completions"
            if args.base_url
            else f"http://{args.host}:{args.port}/v1/chat/completions"
        )
    elif args.backend == "trt":
        api_url = (
            f"{args.base_url}/v2/models/ensemble/generate_stream"
            if args.base_url
            else f"http://{args.host}:{args.port}/v2/models/ensemble/generate_stream"
        )
        if args.model is None:
            print("Please provide a model using `--model` when using `trt` backend.")
            sys.exit(1)

    base_url = (
        f"http://{args.host}:{args.port}"
        if args.base_url is None
        else args.base_url
    )

    if args.model is None:
        try:
            response = requests.get(model_url, headers=get_auth_headers())
            model_list = response.json().get("data", [])
            args.model = model_list[0]["id"] if model_list else None
        except Exception as e:
            print(f"Failed to fetch model from {model_url}. Error: {e}")
            print("Please specify the correct host and port using `--host` and `--port`.")
            sys.exit(1)

    if args.model is None:
        print("No model specified or found. Please provide a model using `--model`.")
        sys.exit(1)

    if not check_chat_template(args.model):
        print(
            "WARNING: It is recommended to use the `Chat` or `Instruct` model for benchmarking. "
            "Because when the tokenizer counts the output tokens, if there is gibberish, it might count incorrectly."
        )

    print(f"Parsed Arguments:")
    for k, v in vars(args).items():
        print(f"{k:20} {v}")
    print("")

    backend = args.backend
    model_id = args.model

    input_requests, already_done = prepare_enhancement_prompts(args)

    return asyncio.run(
        infer_enhancement(
            backend=backend,
            api_url=api_url,
            base_url=base_url,
            model_id=model_id,
            input_requests=input_requests,
            already_done=already_done,
            request_rate=args.request_rate,
            max_concurrency=args.max_concurrency,
            disable_tqdm=args.disable_tqdm,
            extra_request_body=extra_request_body,
        )
    )

if __name__ == "__main__":
    start_time = time.perf_counter()

    parser = ArgumentParser()

    # Input / Output (same as original)
    parser.add_argument("--input-path", type=str, required=True, help="Path to input data file (.jsonl, .parquet)")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file name")
    parser.add_argument("--instruction-path", type=str, required=True, help="YAML file with instruction/task templates")
    parser.add_argument("--task", type=str, default="enhancement", help="Task/template key from the YAML file (for enhancement task)")
    parser.add_argument("--template-fields", type=str, nargs="+", help="Fields from record to fill into template")

    # Target languages (same as original)
    parser.add_argument("--target-langs", type=str, nargs="+", help="List of target language codes or names (e.g. as bn gu hi or Assamese Bengali)")

    # Model & backend (same as original)
    parser.add_argument("--backend", type=str, choices=list(ASYNC_REQUEST_FUNCS.keys()), default="sglang", help="Backend inference engine")
    parser.add_argument("--model", type=str, help="Model name or path (default: use backend config)")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer name or path (default: use model config)")

    # Server / connection (same as original)
    parser.add_argument("--base-url", type=str, default=None, help="Server or API base URL")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, help="Port (default depends on backend)")

    # Generation params (same as original)
    parser.add_argument("--max-new-tokens", type=int, help="Maximum number of new tokens to generate")
    parser.add_argument("--extra-request-body", type=str, metavar='{"key":"value"}', help="Extra JSON for request payload")
    parser.add_argument("--apply-chat-template", action="store_true", help="Apply chat template to prompts")
    parser.add_argument("--ignore-eos", action="store_true", default=False, help="Stop when EOS is generated")

    # Request control (same as original)
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="Requests per second (default: inf = all at once)")
    parser.add_argument("--max-concurrency", type=int, default=100, help="Max concurrent requests (limits parallelism)")
    parser.add_argument("--enable-stream", action="store_true", help="Enable streaming mode")

    # UX (same as original)
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bar")

    args = parser.parse_args()

    run_inference(args)

    duration = time.perf_counter() - start_time
    print(f"Enhancement completed in {timedelta(seconds=int(duration))}.")
