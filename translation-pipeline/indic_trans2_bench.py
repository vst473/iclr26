import sqlite3
import tritonclient.http as http_client
from tritonclient.utils import *
import numpy as np
import json
import re
import concurrent.futures
import logging
import threading
import nltk
import signal
import sys
import argparse
import os
from datetime import datetime
import time
from tqdm import tqdm
import queue

def get_current_time():
    """Returns the current server time in a formatted string."""
    return datetime.now().strftime("%d/%m/%y at %I:%M %p")

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('translation_script.log'),
        logging.StreamHandler()
    ]
)

#  start time variable 
start_time1 = get_current_time()
start_time = time.time()
logging.info(f"Translation started at: {start_time}")

nltk.download('punkt')
nltk.download('punkt_tab')


ENABLE_SSL = False
ENDPOINT_URL = 'localhost:8000'
HTTP_HEADERS = {"Authorization": "Bearer __PASTE_KEY_HERE__"}

already_done = set()
global_pbar = None

# All 16 target languages from IN22-Gen
TARGET_LANGS = [
    "as", "bn", "gu",
    "hi", "kn", "mai", "ml", "mr", "ne", "or", "pa", "sa",
    "sdd", "ta", "te", "ur"
]

# Function to extract text from JSONL file
def extract_text_from_jsonl(file_path, output_file_path, chunk_size=100000):
    text_chunk = []
    total_lines = sum(1 for line in open(file_path, 'r', encoding='utf-8') if "error" not in line)
    logging.info(f"Total lines in input file: {total_lines}")
    
    completed = set()
    if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
        print("started with checking completed ones")
        with open(output_file_path, "r", encoding="utf-8", errors='replace') as file:
            for line in file:
                try: 
                    single_line_json = json.loads(line.strip())
                    _id1 = single_line_json["id"]
                    # if len(single_line_json.get('translations', {})) > 0:
                    #     total_lines -= 1
                    completed.add(_id1)
                except:
                    print(line)
                    continue
        print("completed ID's loaded")

    global global_pbar
    global_pbar = tqdm(total=total_lines, desc="Overall progress")
    
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            try:
                json_obj = json.loads(line)
                if json_obj['id'] not in completed and len(json_obj['eng_Latn']) > 0:
                    text_chunk.append((json_obj['id'], json_obj['eng_Latn']))
            except json.JSONDecodeError as e:
                logging.error(f"Skipping invalid JSON line: {e}")
                continue

            if len(text_chunk) >= chunk_size:
                yield text_chunk
                text_chunk = []

        if text_chunk:
            yield text_chunk
    print("")


def get_string_tensor(string_values, tensor_name):
    string_obj = np.array(string_values, dtype="object")
    input_obj = http_client.InferInput(tensor_name, string_obj.shape, np_to_triton_dtype(string_obj.dtype))
    input_obj.set_data_from_numpy(string_obj)
    return input_obj


def get_translation_input_for_triton(texts, src_lang, tgt_lang):
    return [
        get_string_tensor([[text] for text in texts], "INPUT_TEXT"),
        get_string_tensor([[src_lang]] * len(texts), "INPUT_LANGUAGE_ID"),
        get_string_tensor([[tgt_lang]] * len(texts), "OUTPUT_LANGUAGE_ID"),
    ]


def clean_and_chunk_text(text, max_words=200):
    text = re.sub(r'[^\x00-\x7F]+', '', text)
    sentences = nltk.sent_tokenize(text)
    chunks, current_chunk = [], []

    for sentence in sentences:
        words = sentence.strip().split()
        if len(words) == 0:
            continue
        if len(current_chunk) + len(words) > max_words:
            chunks.append(' '.join(current_chunk))
            current_chunk = []
        current_chunk.extend(words)

    if current_chunk:
        chunks.append(' '.join(current_chunk))

    return chunks


def translate_and_reassemble_text(client, text, src_lang, tgt_lang):
    chunks = clean_and_chunk_text(text)
    b_size = 512
    translated_chunks = []
    for i in range(0, len(chunks), b_size):
        chunk_batch = chunks[i:i + b_size]
        inputs = get_translation_input_for_triton(chunk_batch, src_lang, tgt_lang)
        output0 = http_client.InferRequestedOutput("OUTPUT_TEXT")
        try:
            response = client.infer("nmt", inputs=inputs, outputs=[output0], headers=HTTP_HEADERS, timeout=1000000000)
            batch_translations = response.as_numpy('OUTPUT_TEXT').tolist()
            translated_chunks.extend([chunk[0].decode("utf-8") for chunk in batch_translations])
        except Exception as e:
            logging.error(e)
    return ' '.join(translated_chunks)


def writer_thread_func(output_queue: queue.Queue, file_path: str):
    with open(file_path, "a", encoding="utf-8") as f:
        count = 0
        while True:
            item = output_queue.get()
            if item is None:
                output_queue.task_done()
                break
            
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            
            if count % 1000 == 0:
                print(f"Writer thread: {count} items written so far.")
            output_queue.task_done()
        print(f"Writer thread: finished writing {count} items.")


def process_and_append_batch(batch_index, texts, src_lang, output_queue, args):
    if ENABLE_SSL:
        import gevent.ssl
        triton_http_client = http_client.InferenceServerClient(
            url=ENDPOINT_URL, verbose=False, ssl=True, ssl_context_factory=gevent.ssl._create_default_https_context
        )
    else:
        triton_http_client = http_client.InferenceServerClient(url=ENDPOINT_URL, verbose=False)

    for idx, (text_id, text) in enumerate(texts, start=1):
        try:
            if text_id in already_done:
                global_pbar.update(1)
                continue

            translations = {}
            for tgt_lang in TARGET_LANGS:
                translated_text = translate_and_reassemble_text(triton_http_client, text, src_lang, tgt_lang)
                translations[tgt_lang] = translated_text

            output_queue.put({"id": text_id, "english_og": text,"translations": translations})
            global_pbar.update(1)

        except Exception as e:
            logging.error(f"Batch {batch_index}, Text {idx}, ID {text_id}: Error - {e}")
            basepathh = os.path.dirname(args.output_file_path)
            os.makedirs(f"{basepathh}/log/", exist_ok=True)
            with open(f"{basepathh}/log/translation_failed.log",'a', encoding="utf-8") as f1:
                f1.write(f"{text_id}\n")


def process_and_save(args2, text_data, batch_size=100, src_lang="en", output_file_path="translated_file.jsonl"):
    batches = [text_data[i:i + batch_size] for i in range(0, len(text_data), batch_size)]
    total_batches = len(batches)

    output_queue = queue.Queue()
    writer_thread = threading.Thread(target=writer_thread_func, args=(output_queue, output_file_path))
    writer_thread.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(80, total_batches)) as executor:
        futures = {}
        for idx, batch in enumerate(batches, start=1):
            futures[executor.submit(process_and_append_batch, idx, batch, src_lang, output_queue, args2)] = idx
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing batch: {e}")

    output_queue.put(None)
    writer_thread.join()


def process_and_save_in_chunks(args, file_path, batch_size=100, src_lang="en", output_file_path="translated_file.jsonl"):
    for i, chunk in enumerate(extract_text_from_jsonl(file_path, output_file_path)):
        process_and_save(args, chunk, batch_size, src_lang, output_file_path)
        logging.info(f"=============Chunk-{i} completed===============")


def main():
    global start_time, start_time1, already_done, global_pbar
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file_path", type=str, help="Input file path", required=True)
    parser.add_argument("--output_file_path", type=str, help="Output_file_path", required=True)
    parser.add_argument("--src_lang", type=str, help="Source Language", required=False, default="en")
    
    args = parser.parse_args()

    base_dir = os.path.dirname(args.input_file_path)
    
    logging.basicConfig(
        filename=f"{base_dir}/translation_process.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print(f"Output will be saved to: {args.output_file_path}")

    process_and_save_in_chunks(args,
        file_path=args.input_file_path,
        batch_size=1,   
        src_lang=args.src_lang,
        output_file_path=args.output_file_path
    )
    
    if global_pbar:
        global_pbar.close()
    
    end_time = time.time()
    total_time = end_time - start_time
    days = total_time // (3600 * 24)
    hours = (total_time % (3600 * 24)) // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60

    logging.info(f"Total translation time: {days} days, {hours} hours, {minutes} minutes, {seconds:.2f} seconds")
    logging.info(f"Translation started at: {start_time1}")
    logging.info(f"Translation completed at: {get_current_time()}")


if __name__ == "__main__":
    main()
