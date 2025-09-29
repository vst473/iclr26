#!/usr/bin/env python3
# Requires: pip install aiohttp aiofiles tqdm
import argparse
import asyncio
import aiohttp
import aiofiles
import json
import os
import random
import traceback
from tqdm import tqdm
from typing import List, Optional

async def _backoff_sleep(attempt: int, base: float = 0.5, cap: float = 10.0):
    sleep = min(cap, base * (2 ** attempt))
    jitter = random.random() * sleep * 0.5
    await asyncio.sleep(sleep * 0.5 + jitter)

async def fetch_embeddings(*, session: aiohttp.ClientSession, server_url: str, inputs: List[str],
                           model_name: Optional[str] = None, max_retries: int = 4, timeout: int = 60):
    """
    Keyword-only function. Call like:
      await fetch_embeddings(session=session, server_url=..., inputs=..., model_name=..., ...)
    Returns list of embeddings (list of lists of floats).
    """
    url = server_url.rstrip("/") + "/v1/embeddings"
    payload = {"input": inputs}
    if model_name:
        payload["model"] = model_name

    for attempt in range(max_retries):
        try:
            async with session.post(url, json=payload, timeout=timeout) as resp:
                text = await resp.text()
                if resp.status == 200:
                    data = json.loads(text)
                    embeddings = [item["embedding"] for item in data.get("data", [])]
                    return embeddings
                # retry on transient server errors
                if resp.status >= 500 or resp.status == 429:
                    await _backoff_sleep(attempt)
                    continue
                # other 4xx -> likely bad payload, raise
                raise RuntimeError(f"Bad response {resp.status}: {text}")
        except (aiohttp.ClientConnectionError, aiohttp.ClientPayloadError, asyncio.TimeoutError) as e:
            if attempt < max_retries - 1:
                await _backoff_sleep(attempt)
                continue
            raise
    raise RuntimeError("Failed to get embeddings after retries")

async def process_record(record: dict,
                         session: aiohttp.ClientSession,
                         server_url: str,
                         model_name: Optional[str],
                         sem: asyncio.Semaphore,
                         outfile,
                         pbar: tqdm,
                         max_retries: int,
                         timeout: int,
                         instruct_prefix: str,
                         write_lock: asyncio.Lock):
    async with sem:
        id = record.get("id", "")
        question = record.get("question", "")
        choices = record.get("choices", []) or []
        if not isinstance(choices, list):
            choices = list(choices)

        # Build inputs for embedding
        inputs = f"Question: {question}\nChoices: {choices}"
        # print(inputs)
        embeddings = None
        try:
            embeddings = await fetch_embeddings(
                session=session,
                server_url=server_url,
                inputs=inputs,
                model_name=model_name,
                max_retries=max_retries,
                timeout=timeout
            ) if inputs else []
        except Exception as e:
            err_msg = f"ERROR_FETCHING_EMBEDDINGS: {e}"
            raise err_msg
            
            pbar.update(1)
            return  
        if not embeddings or not all(embeddings):
            pbar.update(1)
            return

        async with write_lock:
            out = {"id": id, "inputs": inputs, "embedding": embeddings}
            await outfile.write(json.dumps(out, ensure_ascii=False) + "\n")
            await outfile.flush()

        pbar.update(1)

async def process_file(input_file: str,
                       output_file: str,
                       server_url: str,
                       model_name: Optional[str],
                       concurrency: int,
                       max_retries: int,
                       timeout: int,
                       instruct_prefix: str):
    total = 0
    with open(input_file, "r", encoding="utf-8") as f:
        for _ in f:
            total += 1

    connector = aiohttp.TCPConnector(limit=concurrency * 2)
    timeout_cfg = aiohttp.ClientTimeout(total=None)
    sem = asyncio.Semaphore(concurrency)
    write_lock = asyncio.Lock()

    async with aiohttp.ClientSession(connector=connector, timeout=timeout_cfg) as session, \
               aiofiles.open(output_file, "a", encoding="utf-8") as outfile:

        tasks = []
        pbar = tqdm(total=total, desc="Embedding", unit="rec")

        with open(input_file, "r", encoding="utf-8") as inf:
            for line in inf:
                line = line.strip()
                if not line:
                    pbar.update(1)
                    continue
                try:
                    record = json.loads(line)
                except Exception:
                    pbar.update(1)
                    continue

                tasks.append(asyncio.create_task(
                    process_record(record=record,
                                   session=session,
                                   server_url=server_url,
                                   model_name=model_name,
                                   sem=sem,
                                   outfile=outfile,
                                   pbar=pbar,
                                   max_retries=max_retries,
                                   timeout=timeout,
                                   instruct_prefix=instruct_prefix,
                                   write_lock=write_lock)
                ))

                if len(tasks) >= max(256, concurrency * 8):
                    await asyncio.gather(*tasks)
                    tasks = []

            if tasks:
                await asyncio.gather(*tasks)

        pbar.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-file", required=True, help="Input JSONL file with fields: question, choices")
    parser.add_argument("--output-file", required=True, help="Output JSONL (one line per choice)")
    parser.add_argument("--server-url", default="http://127.0.0.1:30000", help="vLLM server base URL")
    parser.add_argument("--model", default=None, help="Optional model name to include in payload (omit to use server default)")
    parser.add_argument("--concurrency", type=int, default=8, help="Number of concurrent HTTP requests")
    parser.add_argument("--max-retries", type=int, default=4, help="Retries per request")
    parser.add_argument("--timeout", type=int, default=60, help="Request timeout (seconds)")
    parser.add_argument("--instruct-prefix", default="Instruct: Given a web search query, retrieve relevant passages that answer the query",
                        help="Instruction prefix used to form inputs")
    args = parser.parse_args()

    try:
        asyncio.run(process_file(
            input_file=args.input_file,
            output_file=args.output_file,
            server_url=args.server_url,
            model_name=args.model,
            concurrency=args.concurrency,
            max_retries=args.max_retries,
            timeout=args.timeout,
            instruct_prefix=args.instruct_prefix,
        ))
    except Exception:
        print("Fatal error in main:")
        traceback.print_exc()

if __name__ == "__main__":
    main()
