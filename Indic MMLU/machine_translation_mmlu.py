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
# Global progress bar for tracking overall progress
global_pbar = None

# Function to extract text from JSONL file
def extract_text_from_jsonl(file_path, output_file_path, chunk_size=100000):
    """
    Yields chunks of text from a large JSONL file to process incrementally.
    """
    text_chunk = []
    # Count total lines
    total_lines = sum(1 for line in open(file_path, 'r', encoding='utf-8') if "error" not in line)
    logging.info(f"Total lines in input file: {total_lines}")
    
    # Initialize the global progress bar
    


    completed = set()

    if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
        print("started with checking completed ones")
        with open(output_file_path, "r", encoding="utf-8", errors='replace') as file:
            for line in file:
                try: 
                    single_line_json = json.loads(line.strip())
                    _id1 = single_line_json["id"]
                    if len(single_line_json['question']) > 1:
                        total_lines -= 1
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
                if json_obj['id'] not in completed and len(json_obj['question']) > 0:
                    text_chunk.append((json_obj['id'], json_obj['question'], json_obj['choices']))
                    
            except json.JSONDecodeError as e:
                logging.error(f"Skipping invalid JSON line: {e}")
                continue

            # Yield the chunk once the size reaches chunk_size
            if len(text_chunk) >= chunk_size:
                yield text_chunk
                text_chunk = []  # Reset for the next chunk

        # Yield remaining lines (if any)
        if text_chunk:
            yield text_chunk
    print("")


# Function to append translated text to JSONL
def append_text_to_jsonl(text_list, text_id, output_file_path):
    with open(output_file_path, 'a', encoding='utf-8') as file:
        for text in text_list:
            json_obj = {'nemo_id': text_id, 'text': text}
            file.write(json.dumps(json_obj, ensure_ascii=False) + '\n')

# Function to create string tensor for input
def get_string_tensor(string_values, tensor_name):
    string_obj = np.array(string_values, dtype="object")
    input_obj = http_client.InferInput(tensor_name, string_obj.shape, np_to_triton_dtype(string_obj.dtype))
    input_obj.set_data_from_numpy(string_obj)
    return input_obj

# Function to prepare input for translation
def get_translation_input_for_triton(texts, src_lang, tgt_lang):
    return [
        get_string_tensor([[text] for text in texts], "INPUT_TEXT"),
        get_string_tensor([[src_lang]] * len(texts), "INPUT_LANGUAGE_ID"),
        get_string_tensor([[tgt_lang]] * len(texts), "OUTPUT_LANGUAGE_ID"),
    ]

# Function to clean and chunk text
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

# Function to perform translation and reassemble
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
            # print("ERRRRRRRRRRRRRRORRRRRRRRRR")
            logging.error(e)
            # raise
    return ' '.join(translated_chunks)

def writer_thread_func(output_queue: queue.Queue, file_path: str):
    """Continuously reads JSON objects from the queue and writes them to a file in JSONL format."""
    
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

def process_and_append_batch(batch_index, texts, src_lang, tgt_lang, output_file_path, output_queue, args):
    if ENABLE_SSL:
        import gevent.ssl
        triton_http_client = http_client.InferenceServerClient(
            url=ENDPOINT_URL, verbose=False, ssl=True, ssl_context_factory=gevent.ssl._create_default_https_context
        )
    else:
        triton_http_client = http_client.InferenceServerClient(url=ENDPOINT_URL, verbose=False)
    
    for idx, (text_id, question, choices) in enumerate(texts, start=1):
        try:
            if text_id in already_done:
                global_pbar.update(1)
                continue
            translated_ques = translate_and_reassemble_text(triton_http_client, question, src_lang, tgt_lang)
            # translated_choices = translate_and_reassemble_text(triton_http_client, question, src_lang, tgt_lang)
            # append_text_to_jsonl([translated_text], text_id, output_file_path)
            translated_choices = []
            for choice in choices:
                translated_choices.append(translate_and_reassemble_text(triton_http_client, choice, src_lang, tgt_lang))

            output_queue.put({"id": text_id, "question": translated_ques, "choices":translated_choices})
            
            # logging.info(f"Batch {batch_index}, Text {idx}: Completed.")
            global_pbar.update(1)
        except Exception as e:
            logging.error(f"Batch {batch_index, text_id}, Text {idx}: Error - {e}")
            filenamee = os.path.basename(output_file_path).split('.')[0]   # returns 'elite_personas.part8'
            basepathh = os.path.dirname(output_file_path)
            os.makedirs(f"{basepathh}/log/", exist_ok=True)
            with open(f"{basepathh}/log/translation_to_tgt_lang_failed_{args.tgt_lang}",'a', encoding="utf-8") as f1:
                    f1.write(f"{text_id}\n")
            # global_pbar.update(1)




def process_and_save(args2, text_data, batch_size=100, src_lang="en", tgt_lang="hi", output_file_path="translated_file.jsonl"):
    batches = [text_data[i:i + batch_size] for i in range(0, len(text_data), batch_size)]
    total_batches = len(batches)

    output_queue = queue.Queue()
    writer_thread = threading.Thread(target=writer_thread_func, args=(output_queue, output_file_path))
    writer_thread.start()

    with concurrent.futures.ThreadPoolExecutor(max_workers=min(80, total_batches)) as executor:
        futures = {}
        for idx, batch in enumerate(batches, start=1):
            futures[executor.submit(process_and_append_batch, idx, batch, src_lang, tgt_lang, 
                                   output_file_path, output_queue, args2)] = idx
        
        for future in concurrent.futures.as_completed(futures):
            try:
                future.result()
            except Exception as e:
                logging.error(f"Error processing batch: {e}")
    output_queue.put(None)
    writer_thread.join()

def process_and_save_in_chunks(args, file_path, batch_size=100, src_lang="en", tgt_lang="hi", output_file_path="translated_file.jsonl"):
    for i, chunk in enumerate(extract_text_from_jsonl(file_path, output_file_path)):
        process_and_save(args, chunk, batch_size, src_lang, tgt_lang, output_file_path)
        logging.info(f"=============Chunk-{i} completed===============")

def main():
    global start_time, start_time1, already_done, global_pbar
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--input_file_path", type=str, help="Input file path", required=True)
    parser.add_argument("--output_file_path", type=str, help="Output_file_path", required=True)
    parser.add_argument("--src_lang", type=str, help="Source Language", required=False, default="en")
    parser.add_argument("--tgt_lang", type=str, help="Target language", required=False, default="mr")
    parser.add_argument("--fail_ids_file", type=str, help="fail file path", required=False, default="mr")
    
    args = parser.parse_args()

    base_dir = os.path.dirname(args.input_file_path)
    
    logging.basicConfig(
        filename=f"{base_dir}/translation_process.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s"
    )

    print(f"Output will be saved to: {args.output_file_path}")
    
    # Check if the output file exists to populate already_done
    # if os.path.exists(args.output_file_path):
    #     print("Loading already processed IDs...")
    #     count = 0
    #     with open(args.output_file_path, 'r', encoding='utf-8', errors="replace") as file:
    #         for line in file:
    #             try:
    #                 json_obj = json.loads(line)
    #                 if 'nemo_id' in json_obj:
    #                     already_done.add(json_obj['nemo_id'])
    #                     count += 1
    #             except json.JSONDecodeError as e:
    #                 logging.error(f"Skipping invalid JSON line: {e}")
    #     print(f"Found {len(already_done)} already processed items")
    #     logging.info(f"Already Done-- {len(already_done)}")

    process_and_save_in_chunks(args,
        file_path=args.input_file_path,
        #batch_size=500,
        batch_size=1,   
        src_lang=args.src_lang,
        tgt_lang=args.tgt_lang,
        output_file_path=args.output_file_path
    )
    
    # Close the progress bar when done
    if global_pbar:
        global_pbar.close()
    
    end_time = time.time()  # Ensure this is a float
    total_time = end_time - start_time

    days = total_time // (3600 * 24)
    hours = (total_time % (3600 * 24)) // 3600
    minutes = (total_time % 3600) // 60
    seconds = total_time % 60

    logging.info(f"Total translation time: {days} days, {hours} hours, {minutes} minutes, {seconds:.2f} seconds")

    logging.info(f"Translation started at: {start_time1}")
    logging.info(f"Translation completed at: {get_current_time()}")  # Keep this as string

if __name__ == "__main__":
    main()


