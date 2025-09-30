#!/usr/bin/env python3

import argparse
import os
import json
import re
import logging
import threading
import time
from datetime import datetime
import queue
import concurrent.futures
from tqdm import tqdm
import nltk
import numpy as np
from collections import defaultdict

# For model
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import gc

# --- Utils & logging --------------------------------------------------------
def get_current_time():
    return datetime.now().strftime("%d/%m/%y at %I:%M %p")

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('nllb_translation.log'),
        logging.StreamHandler()
    ]
)

start_time1 = get_current_time()
start_time = time.time()
logging.info(f"Translation started at: {start_time1}")

# Download NLTK data once
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

# --- Language lists & mappings ---------------------------------------------
TARGET_LANGS = [
    "as", "bn", "gu", "hi", "kn", "mai", "ml", "mr", 
    "ne", "or", "pa", "sa", "sdd", "ta", "te", "ur"
]

LANG_CODE_MAP = {
    "as": "asm_Beng", "bn": "ben_Beng", "gu": "guj_Gujr", "hi": "hin_Deva",
    "kn": "kan_Knda", "mai": "mai_Deva", "ml": "mal_Mlym", "mr": "mar_Deva",
    "ne": "npi_Deva", "or": "ory_Orya", "pa": "pan_Guru", "sa": "san_Deva",
    "sdd": "snd_Arab", "ta": "tam_Taml", "te": "tel_Telu", "ur": "urd_Arab",
}

# Global model instance (shared across threads with locks)
model_lock = threading.Lock()
global_tokenizer = None
global_model = None
forced_bos_cache = {}

# --- Optimized model management --------------------------------------------
def initialize_global_model(model_name, device):
    """Initialize model once globally"""
    global global_tokenizer, global_model, forced_bos_cache
    
    with model_lock:
        if global_model is None:
            logging.info(f"Loading model {model_name} on device {device}...")
            
            # Load tokenizer
            global_tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            
            # Load model with optimizations
            if device.startswith("cuda") and torch.cuda.is_available():
                global_model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,  # Use half precision
                    device_map="auto"
                ).eval()  # Set to evaluation mode
                
                # Enable attention slicing for memory efficiency
                if hasattr(global_model, 'enable_attention_slicing'):
                    global_model.enable_attention_slicing()
            else:
                global_model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device).eval()
            
            # Pre-compute forced BOS tokens
            for short_lang, nllb_lang in LANG_CODE_MAP.items():
                forced_bos_cache[short_lang] = get_forced_bos_token_id(global_tokenizer, nllb_lang)
            
            logging.info("Model loaded successfully!")

def get_forced_bos_token_id(tokenizer, tgt_lang_code):
    """Get forced BOS token ID with caching"""
    try:
        if hasattr(tokenizer, "lang_code_to_id"):
            return tokenizer.lang_code_to_id[tgt_lang_code]
    except Exception:
        pass
    
    token_id = tokenizer.convert_tokens_to_ids(tgt_lang_code)
    if token_id is None or token_id == tokenizer.unk_token_id:
        token_id = tokenizer.convert_tokens_to_ids(f"▁{tgt_lang_code}")
    return token_id

# --- Optimized text processing ---------------------------------------------
def fast_sentence_split(text):
    """Faster sentence splitting using regex instead of NLTK for simple cases"""
    # For most cases, this regex works faster than NLTK
    sentences = re.split(r'(?<=[.!?])\s+', text.strip())
    return [s.strip() for s in sentences if s.strip()]

def optimized_chunk_text(text, max_words=150):  # Reduced from 200 for better batching
    """Optimized text chunking"""
    if not text or not text.strip():
        return []
    
    # Try fast splitting first
    sentences = fast_sentence_split(text)
    if not sentences:  # Fallback to NLTK if needed
        sentences = nltk.sent_tokenize(text)
    
    chunks = []
    current_chunk = []
    current_word_count = 0
    
    for sentence in sentences:
        words = sentence.split()  # Faster than strip().split()
        word_count = len(words)
        
        if word_count == 0:
            continue
            
        if current_word_count + word_count > max_words and current_chunk:
            chunks.append(' '.join(current_chunk))
            current_chunk = words
            current_word_count = word_count
        else:
            current_chunk.extend(words)
            current_word_count += word_count
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
    
    return chunks

# --- Optimized batch translation -------------------------------------------
def translate_batch_all_languages(texts, device, max_length=400):  # Reduced max_length
    """Translate a batch of texts to all target languages at once"""
    if not texts:
        return []
    
    global global_tokenizer, global_model
    
    results = []
    
    # Process each target language
    for tgt_short in TARGET_LANGS:
        tgt_nllb = LANG_CODE_MAP[tgt_short]
        forced_bos = forced_bos_cache.get(tgt_short)
        
        try:
            with model_lock:  # Ensure thread safety
                # Tokenize batch
                inputs = global_tokenizer(
                    texts, 
                    return_tensors="pt", 
                    padding=True, 
                    truncation=True, 
                    max_length=1024  # Reduced for speed
                )
                
                # Move to device
                inputs = {k: v.to(device) for k, v in inputs.items()}
                
                # Generation parameters optimized for speed
                gen_kwargs = {
                    "max_length": max_length,
                    "num_beams": 4,  # Reduced from 4 for speed
                    "early_stopping": True,
                    "do_sample": False,
                    "pad_token_id": global_tokenizer.pad_token_id,
                }
                
                if forced_bos is not None and forced_bos != global_tokenizer.unk_token_id:
                    gen_kwargs["forced_bos_token_id"] = int(forced_bos)
                
                # Generate translations
                with torch.no_grad():  # Disable gradient computation
                    outputs = global_model.generate(**inputs, **gen_kwargs)
                
                # Decode results
                translations = global_tokenizer.batch_decode(outputs, skip_special_tokens=True)
                
                # Store results
                for i, translation in enumerate(translations):
                    if i >= len(results):
                        results.append({})
                    results[i][tgt_short] = translation
                
        except Exception as e:
            logging.error(f"Error translating to {tgt_short}: {e}")
            # Fill with empty strings for failed translations
            for i in range(len(texts)):
                if i >= len(results):
                    results.append({})
                results[i][tgt_short] = ""
    
    return results

# --- Optimized file processing ---------------------------------------------
def extract_text_optimized(file_path, output_file_path, chunk_size=50000):  # Reduced chunk size
    """Optimized text extraction with better memory usage"""
    
    # Count total lines efficiently
    logging.info("Counting total lines...")
    with open(file_path, 'rb') as f:
        total_lines = sum(1 for _ in f) 
    logging.info(f"Total lines in input file: {total_lines}")
    
    # Load completed IDs
    completed = set()
    if os.path.exists(output_file_path) and os.path.getsize(output_file_path) > 0:
        logging.info("Loading completed IDs...")
        with open(output_file_path, "r", encoding="utf-8", errors='replace') as file:
            for line in file:
                try:
                    obj = json.loads(line.strip())
                    completed.add(obj["id"])
                except:
                    continue
        logging.info(f"Loaded {len(completed)} completed IDs")
    
    # Process file in chunks
    text_chunk = []
    processed = 0
    
    with open(file_path, 'r', encoding='utf-8', buffering=8192*4) as file:  # Larger buffer
        for line in file:
            try:
                obj = json.loads(line)
                if (obj['id'] not in completed and 
                    len(obj.get('eng_Latn', "").strip()) > 0):
                    text_chunk.append((obj['id'], obj['eng_Latn']))
                
                processed += 1
                
                if len(text_chunk) >= chunk_size:
                    yield text_chunk, total_lines
                    text_chunk = []
                    
            except json.JSONDecodeError:
                continue
    
    if text_chunk:
        yield text_chunk, total_lines

# --- Optimized processing pipeline -----------------------------------------
def process_batch_optimized(batch_data, device, output_queue):
    """Process a batch of texts with optimized translation"""
    batch_texts = []
    batch_ids = []
    batch_originals = []
    
    # Prepare batch data
    for text_id, original_text in batch_data:
        # Chunk the text
        chunks = optimized_chunk_text(original_text)
        if chunks:
            batch_texts.extend(chunks)
            batch_ids.extend([text_id] * len(chunks))
            batch_originals.extend([original_text] * len(chunks))
    
    if not batch_texts:
        return
    
    # Translate all chunks for all languages
    translation_results = translate_batch_all_languages(batch_texts, device)
    
    # Reassemble results by ID
    id_translations = defaultdict(lambda: defaultdict(list))
    id_originals = {}
    
    for i, (text_id, original_text) in enumerate(zip(batch_ids, batch_originals)):
        id_originals[text_id] = original_text
        if i < len(translation_results):
            for lang, translation in translation_results[i].items():
                id_translations[text_id][lang].append(translation)
    
    # Create final output
    for text_id in id_originals:
        final_translations = {}
        for lang in TARGET_LANGS:
            chunks = id_translations[text_id][lang]
            final_translations[lang] = ' '.join(chunks) if chunks else ""
        
        output_queue.put({
            "id": text_id,
            "english_og": id_originals[text_id],
            "translations": final_translations
        })

# --- Writer thread ---------------------------------------------------------
def writer_thread_func(output_queue: queue.Queue, file_path: str):
    """Optimized writer with buffering"""
    buffer = []
    buffer_size = 100  # Write in batches
    count = 0
    
    with open(file_path, "a", encoding="utf-8", buffering=8192*4) as f:
        while True:
            item = output_queue.get()
            if item is None:
                # Write remaining buffer
                if buffer:
                    f.writelines(buffer)
                    f.flush()
                output_queue.task_done()
                break
                
            buffer.append(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            
            # Write buffer when full
            if len(buffer) >= buffer_size:
                f.writelines(buffer)
                f.flush()
                buffer = []
                
            if count % 500 == 0:
                logging.info(f"Processed {count} items")
                
            output_queue.task_done()
    
    logging.info(f"Writer finished: {count} items total")

# --- Main processing -------------------------------------------------------
def process_file_optimized(args):
    """Main optimized processing function"""
    device = args.device
    
    # Initialize model once
    initialize_global_model(args.model_name, device)
    
    # Setup writer thread
    output_queue = queue.Queue(maxsize=1000)  # Limit queue size
    writer_thread = threading.Thread(
        target=writer_thread_func, 
        args=(output_queue, args.output_file_path), 
        daemon=True
    )
    writer_thread.start()
    
    # Process file
    total_processed = 0
    pbar = None
    
    for chunk_data, total_lines in extract_text_optimized(args.input_file_path, args.output_file_path):
        if pbar is None:
            pbar = tqdm(total=total_lines, desc="Processing")
        
        # Process in smaller batches for better parallelization
        batch_size = min(args.batch_size, 20)  # Limit batch size
        batches = [chunk_data[i:i + batch_size] for i in range(0, len(chunk_data), batch_size)]
        
        # Use ThreadPoolExecutor with limited workers
        max_workers = min(args.max_workers, 2) if device.startswith("cuda") else args.max_workers
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_batch_optimized, batch, device, output_queue)
                for batch in batches
            ]
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    if pbar:
                        pbar.update(batch_size)
                except Exception as e:
                    logging.error(f"Batch processing error: {e}")
        
        total_processed += len(chunk_data)
        logging.info(f"Completed chunk. Total processed: {total_processed}")
        
        # Force garbage collection
        gc.collect()
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    
    # Cleanup
    output_queue.put(None)
    writer_thread.join()
    
    if pbar:
        pbar.close()

# --- CLI -------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Optimized NLLB Translation")
    parser.add_argument("--input_file_path", type=str, required=True, help="Input JSONL file")
    parser.add_argument("--output_file_path", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--src_lang", type=str, default="eng_Latn", help="Source language")
    parser.add_argument("--model_name", type=str, default="facebook/nllb-200-distilled-600M", help="Model name")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device")
    parser.add_argument("--batch_size", type=int, default=16, help="Batch size for processing")
    parser.add_argument("--max_workers", type=int, default=2, help="Max worker threads")
    
    args = parser.parse_args()
    
    # Optimize for GPU if available
    if args.device.startswith("cuda"):
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    logging.info(f"Starting optimized translation with model: {args.model_name}")
    logging.info(f"Device: {args.device}, Batch size: {args.batch_size}, Workers: {args.max_workers}")
    
    process_file_optimized(args)
    
    # Final statistics
    end_time = time.time()
    total_time = end_time - start_time
    hours, remainder = divmod(total_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    
    logging.info(f"Translation completed in {int(hours)}h {int(minutes)}m {seconds:.1f}s")
    logging.info(f"Started: {start_time1}, Finished: {get_current_time()}")

if __name__ == "__main__":
    main()
