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

# Mapping: code -> display name (as provided)
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
    return text[len(prefix) :] if text.startswith(prefix) else text


def remove_suffix(text, suffix):
    return text[: -len(suffix)] if text.endswith(suffix) else text


def detect_mime(base64_str):
    try:
        img_bytes = base64.b64decode(base64_str)
        mime = magic.from_buffer(img_bytes, mime=True)
        return mime
    except Exception:
        return "application/octet-stream"


def get_auth_headers():
    api_key = os.environ.get("OPENAI_API_KEY")
    if api_key:
        return {"Authorization": f"Bearer {api_key}"}
    else:
        return {}


# --- Async request functions (kept largely as in original script) ---
async def async_request_trt_llm(request_func_input, pbar = None):
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


async def async_request_openai_completions(request_func_input, pbar = None):
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
                    # print(generated_text.strip())
                    if output["generated_text"]:
                        break

            except Exception:
                pass

        if not output["generated_text"] and attempt < max_retries:
            await asyncio.sleep(retry_delay)

    if pbar:
        pbar.update(1)

    return output


async def async_request_sglang_generate(request_func_input, pbar = None):
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


def fill_instruction(template, record, template_fields, code):
    if not template_fields:
        return template

    values = [get_by_path(record, field) for field in template_fields]

    if any(v is None for v in values):
        raise ValueError(f"Missing required field(s) for template_fields: {template_fields}")
    values.append(LANG_CODE_MAP[code])
    print(template.format(*values))
    return template.format(*values)


def load_finished_ids(output_file):
    if not os.path.exists(output_file):
        return set()

    finished_ids = set()
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            id_ = rec.get("id")
            if id_ is not None:
                finished_ids.add(id_)

    return finished_ids


# Normalize target langs: accept codes or names (case-insensitive)
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


# --- prompt preparation: now create one request per (record, language code) ---
def prepare_prompts(args):
    records = []

    ext = os.path.splitext(args.input_path)[1].lower()
    if ext in [".json", ".jsonl"]:
        records = load_jsonl(args.input_path)
    elif ext in [".parquet"]:
        records = load_parquet(args.input_path)
    else:
        raise ValueError("Cannot detect file type and unknown extension!")

    template = load_task_template(args.instruction_path, args.task)

    finished_ids = load_finished_ids(args.output_file)

    input_requests = []
    for r in records:
        orig_id = r.get("id")
        if orig_id in finished_ids:
            continue  # already have full translations for this id

        try:
            code = ""
            # filled_prompt = fill_instruction(template, r, args.template_fields, code  )
            filled_prompt = ""
        except ValueError as e:
            raise ValueError(f"Error in record {r.get('id')}: {e}")

        # For each target language code, create an independent request
        if not args.target_langs:
            # fallback: single request as before
            input_requests.append({
                "id": orig_id,
                "prompt": filled_prompt,
                "image_data": [r.get("image", None)],
                "record": r,
            })
        else:
            codes = normalize_target_langs(args.target_langs)
            for code in codes:
                per_lang_id = f"{orig_id}_{code}"

                # per_lang_prompt = (
                #     filled_prompt
                #     + f"Translate the following segment into {LANG_CODE_MAP.get(code, code)} ({code}) and return ONLY the translated text (no JSON, no commentary)."
                # )
                per_lang_prompt = fill_instruction(template, r, args.template_fields, code)
                
                print(per_lang_prompt)
                input_requests.append({
                    "id": per_lang_id,
                    "prompt": per_lang_prompt,
                    "image_data": [r.get("image", None)],
                    "record": r,
                })

    print(f"Already done (full ids): {len(finished_ids)} | Per-language requests to do: {len(input_requests)}")

    return input_requests, len(finished_ids)


async def get_request(input_requests, request_rate):
    input_requests = iter(input_requests)
    for request in input_requests:
        yield request

        if request_rate == float("inf"):
            continue

        interval = np.random.exponential(1.0 / request_rate)
        await asyncio.sleep(interval)


# --- inference and aggregation logic ---
async def infer(backend, api_url, base_url, model_id, input_requests, already_done, request_rate, max_concurrency, disable_tqdm, extra_request_body):
    if backend in ASYNC_REQUEST_FUNCS:
        request_func = ASYNC_REQUEST_FUNCS[backend]
    else:
        raise ValueError(f"Unknown backend: {backend}")

    semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

    async def limited_request_func(request_func_input, pbar):
        if semaphore is None:
            result = await request_func(request_func_input = request_func_input, pbar = pbar)
        else:
            async with semaphore:
                result = await request_func(request_func_input = request_func_input, pbar = pbar)
        result["record"] = request_func_input.get("generated_text")
        # print(f"[DEBUG][limited_request_func] id={result.get('id')} generated_text={result.get('generated_text')!r}")

        return result

    if "sglang" in backend:
        requests.post(base_url + "/flush_cache", headers=get_auth_headers())

    time.sleep(1.0)

    pbar = None if disable_tqdm else tqdm(total=already_done + len(input_requests), initial=already_done)

    tasks = []
    async for request in get_request(input_requests, request_rate):
        request_func_input = {
            "model": model_id,
            "id": request.get("id"),
            "prompt": request.get("prompt"),
            "api_url": api_url,
            "image_data": request.get("image_data"),
            "extra_request_body": extra_request_body,
            "record": request.get("record"),
        }

        tasks.append(
            asyncio.create_task(
                limited_request_func(request_func_input=request_func_input, pbar=pbar)
            )
        )

    # Aggregation structure: id -> {"eng_Latn": ..., "translations": {code: text}}
    aggregation = defaultdict(lambda: {"eng_Latn": None, "translations": {}})
    target_set = set(normalize_target_langs(args.target_langs)) if args.target_langs else None

    output_dir = os.path.dirname(args.output_file)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # We'll append final combined records to the output file as they complete
    async with aiofiles.open(args.output_file, "a", encoding="utf-8") as f:
        for task in asyncio.as_completed(tasks):
            result = await task
            gen = result.get("generated_text", "").strip()
            if not gen:
                continue

            resp_id = result.get("id")
            if resp_id is None:
                continue

            parts = resp_id.rsplit("_", 1)
            if len(parts) == 2:
                orig_id, code = parts[0], parts[1]
            else:
                orig_id = resp_id
                code = "unknown"

            translation_text = gen.strip()
            if (translation_text.startswith('"') and translation_text.endswith('"')) or (translation_text.startswith("'") and translation_text.endswith("'")):
                translation_text = translation_text[1:-1].strip()

            record = result.get("record", {}) or {}
            if aggregation[orig_id]["eng_Latn"] is None:
                eng_text = None
                if isinstance(record, dict):
                    eng_text = record.get("eng_Latn")
                    if eng_text is None and args.template_fields:
                        first_field = args.template_fields[0]
                        eng_text = get_by_path(record, first_field) if first_field else None
                aggregation[orig_id]["eng_Latn"] = eng_text

            aggregation[orig_id]["translations"][code] = translation_text

            # if we have a known full target set and collected all codes -> flush
            if target_set is not None and set(aggregation[orig_id]["translations"].keys()) >= target_set:
                out_line = {
                    "id": orig_id,
                    "eng_Latn": aggregation[orig_id]["eng_Latn"],
                    "translations": aggregation[orig_id]["translations"],
                }
                await f.write(json.dumps(out_line, ensure_ascii=False) + "\n")
                del aggregation[orig_id]

    # flush remaining partials
    remaining = list(aggregation.items())
    if remaining:
        async with aiofiles.open(args.output_file, "a", encoding="utf-8") as f:
            for orig_id, data in remaining:
                out_line = {
                    "id": orig_id,
                    "eng_Latn": data.get("eng_Latn"),
                    "translations": data.get("translations", {}),
                }
                await f.write(json.dumps(out_line, ensure_ascii=False) + "\n")

    if pbar is not None:
        pbar.close()


# --- remaining helpers and CLI glue (kept similar to original) ---

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
            "WARNING It is recommended to use the `Chat` or `Instruct` model for benchmarking."
            "Because when the tokenizer counts the output tokens, if there is gibberish, it might count incorrectly."
        )

    print(f"Parsed Arguments:")
    for k, v in vars(args).items():
        print(f"{k:20} {v}")
    print("")

    backend = args.backend
    model_id = args.model
    input_requests, already_done = prepare_prompts(args)

    return asyncio.run(
        infer(
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

    # Input / Output
    parser.add_argument("--input-path", type=str, required=True, help="Path to input data file (.jsonl, .parquet)")
    parser.add_argument("--output-file", type=str, required=True, help="Output JSONL file name")
    parser.add_argument("--instruction-path", type=str, required=True, help="YAML file with instruction/task templates")
    parser.add_argument("--task", type=str, required=True, help="Task/template key from the YAML file")
    parser.add_argument("--template-fields", type=str, nargs="+", help="Fields from record to fill into template")

    # New: target languages (one independent request per language code)
    parser.add_argument("--target-langs", type=str, nargs="+", help="List of target language codes or names (e.g. as bn gu hi or Assamese Bengali)")

    # Model & backend
    parser.add_argument("--backend", type=str, choices=list(ASYNC_REQUEST_FUNCS.keys()), default="sglang", help="Backend inference engine")
    parser.add_argument("--model", type=str, help="Model name or path (default: use backend config)")
    parser.add_argument("--tokenizer", type=str, help="Tokenizer name or path (default: use model config)")

    # Server / connection
    parser.add_argument("--base-url", type=str, default=None, help="Server or API base URL")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
    parser.add_argument("--port", type=int, help="Port (default depends on backend)")

    # Generation params
    parser.add_argument("--max-new-tokens", type=int, help="Maximum number of new tokens to generate")
    parser.add_argument("--extra-request-body", type=str, metavar='{"key":"value"}', help="Extra JSON for request payload")
    parser.add_argument("--apply-chat-template", action="store_true", help="Apply chat template to prompts")
    parser.add_argument("--ignore-eos", action="store_true", default=False, help="Stop when EOS is generated")

    # Request control
    parser.add_argument("--request-rate", type=float, default=float("inf"), help="Requests per second (default: inf = all at once)")
    parser.add_argument("--max-concurrency", type=int, default=100, help="Max concurrent requests (limits parallelism)")
    parser.add_argument("--enable-stream", action="store_true", help="Enable streaming mode")

    # UX
    parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bar")

    args = parser.parse_args()
    run_inference(args)
    duration = time.perf_counter() - start_time
    print(f"Inference completed in {timedelta(seconds=int(duration))}.")
