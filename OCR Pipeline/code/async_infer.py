import asyncio
import json
import os
import random
import resource
import sys
import time
import traceback
import yaml
from pathlib import Path
import filetype
from argparse import ArgumentParser
from datetime import timedelta
import aiohttp
import aiofiles
import numpy as np
import requests
from tqdm.asyncio import tqdm
from transformers import AutoTokenizer

global args


def _create_bench_client_session():

	BENCH_AIOHTTP_TIMEOUT_SECONDS = 6 * 60 * 60
	BENCH_AIOHTTP_READ_BUFSIZE_BYTES = 10 * 1024**2

	aiohttp_timeout = aiohttp.ClientTimeout(total=BENCH_AIOHTTP_TIMEOUT_SECONDS)
	return aiohttp.ClientSession(timeout=aiohttp_timeout, read_bufsize=BENCH_AIOHTTP_READ_BUFSIZE_BYTES)


def remove_prefix(text, prefix):
	return text[len(prefix) :] if text.startswith(prefix) else text


def remove_suffix(text, suffix):
	return text[: -len(suffix)] if text.endswith(suffix) else text


def get_auth_headers():
	api_key = os.environ.get("OPENAI_API_KEY")
	if api_key:
		return {"Authorization": f"Bearer {api_key}"}
	else:
		return {}


async def async_request_trt_llm(request_func_input, pbar = None):
	api_url = request_func_input["api_url"]
	assert api_url.endswith("generate_stream")

	async with _create_bench_client_session() as session:
		payload = {
			"accumulate_tokens": True,
			"text_input": request_func_input["prompt"],
			"temperature": 0.7,
			"top_p": 1.0,
			"max_tokens": request_func_input["max_new_tokens"],
			"stream": True,
			"min_length": request_func_input["max_new_tokens"],
			"end_id": 1048576,
			**request_func_input["extra_request_body"],
		}
		if args.disable_ignore_eos:
			del payload["min_length"]
			del payload["end_id"]

		output = {
			"id": request_func_input.get("id"),
			"generated_text": "",
		}

		try:
			async with session.post(url=api_url, json=payload) as response:
				if response.status != 200:
					raise RuntimeError(f"API request failed with status {response.status}: {response.reason}")
					# pass
				
				async for chunk_bytes in response.content:
					chunk_bytes = chunk_bytes.strip()
					if not chunk_bytes:
						continue

					chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data:")

					data = json.loads(chunk)
					output["generated_text"] += data["text_output"]
		except Exception as e:
			raise RuntimeError(f"Error while generating output for id {output['id']}: {e}") from e
			# pass

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
			"temperature": 0.7,
			"best_of": 1,
			"max_tokens": request_func_input["max_new_tokens"],
			"stream": not args.disable_stream,
			"ignore_eos": not args.disable_ignore_eos,
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
					raise RuntimeError(f"API request failed with status {response.status}: {response.reason}")
					# pass

				async for chunk_bytes in response.content:
					chunk_bytes = chunk_bytes.strip()
					if not chunk_bytes:
						continue

					chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
					if chunk == "[DONE]":
						pass
					else:
						data = json.loads(chunk)

						
						if data["choices"][0]["text"]:
							generated_text += data["choices"][0]["text"]

				output["generated_text"] = generated_text
				output["success"] = True
				output["output_len"] = output_len
		except Exception as e:
			raise RuntimeError(f"Error while generating output for id {mmlu_mr_in.jsonloutput['id']}: {e}") from e
			# pass

	if pbar:
		pbar.update(1)
	return output


async def async_request_openai_chat_completions(request_func_input, pbar = None):
	"""Makes a request to the OpenAI Chat Completions API.

	Handles both streaming and non-streaming responses, including support
	for image data in messages.
	"""
	api_url = request_func_input["api_url"]
	assert api_url.endswith("chat/completions"), "OpenAI Chat Completions API URL must end with 'chat/completions'."

	if request_func_input["image_data"]:
		content_items = [
			{
				"type": "image_url",
				"image_url": {"url": img_url},
			}
			for img_url in request_func_input["image_data"]
		]
		content_items.append({"type": "text", "text": request_func_input["prompt"]})
		messages = [
			{
				"role": "user",
				"content": content_items,
			},
		]
	else:
		messages = [{"role": "user", "content": request_func_input["prompt"]}]

	async with _create_bench_client_session() as session:
		payload = {
			"model": request_func_input["model"],
			"messages": messages,
			"temperature": 0.7,
			"max_tokens": request_func_input["max_new_tokens"],
			"stream": not args.disable_stream,
			**request_func_input["extra_request_body"],
		}
		headers = get_auth_headers()

		output = {
			"id": request_func_input.get("id"),
			"generated_text": "",
			"subject":request_func_input.get("subject"),
			"answer":request_func_input.get("answer"),
			"og_question":request_func_input.get("og_question"),
			"og_choices":request_func_input.get("og_choices"),
			"enhanced_text":request_func_input.get("enhanced_text")
		}

		generated_text = ""
		try:
			async with session.post(url=api_url, json=payload, headers=headers) as response:
				if response.status != 200:
					raise RuntimeError(f"API request failed with status {response.status}: {response.reason}")
					# pass

				if args.disable_stream:
					# Non-streaming response
					response_json = await response.json()
					output["generated_text"] = response_json["choices"][0]["message"]["content"]
				else:
					# Streaming response
					async for chunk_bytes in response.content:
						chunk_bytes = chunk_bytes.strip()
						if not chunk_bytes:
							continue

						chunk = remove_prefix(chunk_bytes.decode("utf-8"), "data: ")
						if chunk == "[DONE]":
							pass
						else:
							data = json.loads(chunk)

							# Check if this chunk contains content
							delta = data.get("choices", [{}])[0].get("delta", {})
							content = delta.get("content", "")

							if content:
								generated_text += content

					output["generated_text"] = generated_text
		except Exception as e:
			raise RuntimeError(f"Error while generating output for id {output['id']}: {e}") from e
			# pass

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
				"temperature": 0.7,
				"top_p": 0.9,
				"repetition_penalty": 1.1,
				"max_new_tokens": request_func_input["max_new_tokens"],
				"ignore_eos": False,
			},
			"stream": not args.disable_stream,
			**request_func_input["extra_request_body"],
		}

		# Add image data if available (list of image urls/base64)
		if request_func_input["image_data"]:
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
					raise RuntimeError(f"API request failed with status {response.status}: {response.reason}")
					# pass

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
		except Exception as e:
			raise RuntimeError(f"Error while generating output for id {output['id']}: {e}") from e
			# pass

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


def load_jsonl(path):
	"""
	Load a JSONL file and return a list of dictionaries with the same keys/values as in the file.
	"""
	records = []
	with open(path, "r", encoding="utf-8") as f:
		for line in f:
			line = line.strip()
			if not line:
				continue
			try:
				obj = json.loads(line)
			except Exception:
				continue	# skip malformed line silently; optionally log here
			records.append(obj)
	return records


def load_parquet(path):
	"""
	Load a Parquet file and return a list of dictionaries.
	"""
	df = pd.read_parquet(path)
	return df.to_dict(orient="records")


def load_task_template(yaml_path, task):
	"""
	Load a YAML file containing multiple task templates and return the one specified.
	"""
	if not os.path.isfile(yaml_path):
		raise FileNotFoundError(f"YAML file not found: {yaml_path}")

	with open(yaml_path, "r", encoding="utf-8") as f:
		templates = yaml.safe_load(f)

	if task not in templates:
		raise KeyError(f"Task '{task}' not found in YAML. Available tasks: {list(templates.keys())}")

	return templates[task]


def get_by_path(data, path):
	"""
	Get a value from nested dict/list using dot-separated path.
	Supports basic list index and simple filters.
	Example:
	  "messages.0.content"
	  "messages.[?role==user].content"
	"""
	import re

	parts = path.split(".")
	cur = data
	for p in parts:
		# Handle list filter like [ ?role==user ]
		if p.startswith("[?"):
			# Extract key and value (only supports == for simplicity)
			m = re.match(r"\[\?([^=]+)==(.+)\]", p)
			if not m:
				raise ValueError(f"Invalid filter syntax: {p}")
			key, val = m.group(1), m.group(2)
			val = val.strip('"\'')  # remove quotes
			if isinstance(cur, list):
				cur = next((x for x in cur if x.get(key) == val), None)
			else:
				cur = None
		# Handle list index
		elif p.isdigit():
			cur = cur[int(p)] if isinstance(cur, list) else None
		else:
			cur = cur.get(p) if isinstance(cur, dict) else None

		if cur is None:
			break
	return cur


def fill_instruction(template, record, template_fields):
	"""
	Fill a template string with specified fields from a record.
	If a field is not found in record, use the field value directly.
	"""
	values = []
	for field in template_fields:
		try:
			val = get_by_path(record, field)
			if val is None:  # if path exists but value is None
				val = field
		except Exception:
			# If get_by_path fails (invalid path), use field as literal
			val = field
		values.append(val)

	# print(template.format(*values))
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
		if r.get("id") in finished_ids:
			continue

		try:
			filled_prompt = fill_instruction(template, r, args.template_fields)
		except ValueError as e:
			raise ValueError(f"Error in record {r.get("id")}: {e}")
		# print(r.get("translated_text"))
		# raise "Stop"
		input_requests.append({
			"id": r.get("id", None),
			"prompt": filled_prompt,
			"subject":r.get("subject"),
			"answer":r.get("answer"),
			"og_question":r.get("og_question"),
			"og_choices":r.get("og_choices"),
			"enhanced_text":r.get("translated_text")
		})
	
	print(f"Already done: {len(finished_ids)} | To do now: {len(input_requests)}")

	return input_requests, len(finished_ids)


async def get_request(input_requests, request_rate):
	input_requests = iter(input_requests)
	for request in input_requests:
		yield request

		if request_rate == float("inf"):
			# If the request rate is infinity, then we don't need to wait.
			continue

		# Sample the request interval from the exponential distribution.
		interval = np.random.exponential(1.0 / request_rate)
		# The next request will be sent after the interval.
		await asyncio.sleep(interval)


async def infer(backend, api_url, base_url, model_id, input_requests, already_done, request_rate, max_concurrency, disable_tqdm, extra_request_body):
	if backend in ASYNC_REQUEST_FUNCS:
		request_func = ASYNC_REQUEST_FUNCS[backend]
	else:
		raise ValueError(f"Unknown backend: {backend}")

	semaphore = asyncio.Semaphore(max_concurrency) if max_concurrency else None

	async def limited_request_func(request_func_input, pbar):
		if semaphore is None:
			return await request_func(request_func_input = request_func_input, pbar = pbar)
		async with semaphore:
			return await request_func(request_func_input = request_func_input, pbar = pbar)

	# # Warmup 
	# requests.get(base_url + "/health_generate", headers=get_auth_headers())

	# Flush cache
	if "sglang" in backend:
		requests.post(base_url + "/flush_cache", headers=get_auth_headers())

	time.sleep(1.0)

	pbar = None if disable_tqdm else tqdm(total=already_done + len(input_requests), initial=already_done)

	# Run all requests
	tasks = []
	async for request in get_request(input_requests, request_rate):
		request_func_input = {
			"model": model_id,
			"id": request.get("id"), 
			"prompt": request.get("prompt"),
			"api_url": api_url,
			"max_new_tokens": args.max_new_tokens,
			"image_data": request.get("image_data"),
			"extra_request_body": extra_request_body,
			"subject":request.get("subject"),
			"answer":request.get("answer"),
			"og_question":request.get("og_question"),
			"og_choices":request.get("og_choices"),
			"enhanced_text":request.get("enhanced_text")
		}

		tasks.append(
			asyncio.create_task(
				limited_request_func(request_func_input=request_func_input, pbar=pbar)
			)
		)

	output_dir = os.path.dirname(args.output_file)
	if output_dir:
		os.makedirs(output_dir, exist_ok=True)

	async with aiofiles.open(args.output_file, "a", encoding="utf-8") as f:
		for task in asyncio.as_completed(tasks):
			result = await task
			await f.write(json.dumps(result, ensure_ascii=False) + "\n")

	if pbar is not None:
		pbar.close()


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

	# Set global environments
	set_ulimit()

	extra_request_body = {}
	if args.extra_request_body:
		extra_request_body = json.loads(args.extra_request_body)

	# Set url
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

	# Get model name
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

	# TODO: apply chat template while preparing prompts
	if not check_chat_template(args.model):
		print(
			"\nWARNING It is recommended to use the `Chat` or `Instruct` model for benchmarking.\n"
			"Because when the tokenizer counts the output tokens, if there is gibberish, it might count incorrectly.\n"
		)

	print(f"\nParsed Arguments:")
	for k, v in vars(args).items():
		print(f"{k:20} {v}")
	print("\n")

	# Read Inputs
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
	parser.add_argument("--template-fields", type=str, nargs="+", required=True, help="Fields from record to fill into template")

	# Model & backend
	parser.add_argument("--backend", type=str, choices=list(ASYNC_REQUEST_FUNCS.keys()), default="sglang", help="Backend inference engine")
	parser.add_argument("--model", type=str, help="Model name or path (default: use backend config)")
	parser.add_argument("--tokenizer", type=str, help="Tokenizer name or path (default: use model config)")

	# Server / connection
	parser.add_argument("--base-url", type=str, default=None, help="Server or API base URL")
	parser.add_argument("--host", type=str, default="0.0.0.0", help="Host address")
	parser.add_argument("--port", type=int, help="Port (default depends on backend)")

	# Generation params
	parser.add_argument("--max-new-tokens", type=int, default=1024, help="Maximum number of new tokens to generate")
	parser.add_argument("--extra-request-body", type=str, metavar='{"key":"value"}', help="Extra JSON for request payload (e.g., sampling params)")
	parser.add_argument("--apply-chat-template", action="store_true", help="Apply chat template to prompts")
	parser.add_argument("--disable-ignore-eos", action="store_true", help="Disable ignoring EOS")

	# Request control
	parser.add_argument("--request-rate", type=float, default=float("inf"), help="Requests per second (default: inf = all at once)")
	parser.add_argument("--max-concurrency", type=int, default=100, help="Max concurrent requests (limits parallelism)")
	parser.add_argument("--disable-stream", action="store_true", help="Disable streaming mode")

	# UX
	parser.add_argument("--disable-tqdm", action="store_true", help="Disable tqdm progress bar")


	args = parser.parse_args()
	run_inference(args)
	duration = time.perf_counter() - start_time
	print(f"\nInference completed in {timedelta(seconds=int(duration))}.")
