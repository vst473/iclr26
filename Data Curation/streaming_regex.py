import os
import re
import sys
import time
import queue
import json
import threading
import argparse
import concurrent.futures
from typing import Dict, List, Pattern, Set, Tuple

# Define a worker class for handling pattern matching in parallel
class PatternMatcher:
    def __init__(self, patterns: Dict[str, Pattern], batch_size: int = 1000):
        self.patterns = patterns
        self.batch_size = batch_size
        self.result_queues: Dict[str, queue.Queue] = {}
        
    def initialize_queues(self):
        """Initialize result queues for each language"""
        for lang in self.patterns.keys():
            self.result_queues[lang] = queue.Queue()
        return self.result_queues
        
    def process_batch(self, batch: List[str]) -> Dict[str, List[str]]:
        """Process a batch of lines and return matches by language"""
        results: Dict[str, List[str]] = {lang: [] for lang in self.patterns.keys()}
        
        for line in batch:
            for lang, pattern in self.patterns.items():
                if pattern.fullmatch(line):
                    results[lang].append(line)
                    # Uncomment this if you want exclusive matching (one language per line)
                    # break
        
        return results
    
    def matcher_worker(self, input_queue: queue.Queue):
        """Worker function that processes batches from input queue"""
        batch = []
        while True:
            try:
                line = input_queue.get(timeout=1)
                if line is None:
                    break
                
                batch.append(line)
                if len(batch) >= self.batch_size:
                    self._process_and_distribute(batch)
                    batch = []
                    
            except queue.Empty:
                if batch:  # Process remaining items in batch
                    self._process_and_distribute(batch)
                    batch = []
            finally:
                if line is not None:
                    input_queue.task_done()
        
        # Process any remaining items
        if batch:
            self._process_and_distribute(batch)
        
        # Signal completion to all result queues
        for q in self.result_queues.values():
            q.put(None)
            
        input_queue.task_done()
    
    def _process_and_distribute(self, batch: List[str]):
        """Process a batch and put results in appropriate queues"""
        results = self.process_batch(batch)
        for lang, items in results.items():
            for item in items:
                self.result_queues[lang].put(item)

def writer_thread_func(output_queue: queue.Queue, output_path: str):
    """Continuously reads items from the queue and writes them to a file in JSONL format"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    count = 0
    
    with open(output_path, "a", encoding="utf-8") as f:
        while True:
            item = output_queue.get()
            if item is None:
                output_queue.task_done()
                break
            
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
            count += 1
            output_queue.task_done()
    
    # print(f"Writer thread for {output_path}: finished writing {count} items.")

def main():
    parser = argparse.ArgumentParser(description='Process text and separate by language with parallel processing')
    parser.add_argument('--output-dir', type=str, required=True,
                        help='Directory to store output files')
    parser.add_argument('--output-file', type=str, required=True,
                        help='Name of output file in each language directory')
    parser.add_argument('--workers', type=int, default=max(1, os.cpu_count() - 1),
                        help='Number of worker threads for pattern matching')
    parser.add_argument('--batch-size', type=int, default=1000,
                        help='Number of lines to process in a batch')
    args = parser.parse_args()
    
    # Define patterns for all supported languages
    patterns = {
        "hin": re.compile(r"^[\u0900-\u097F\s]+$"),  # Hindi/Devanagari
        "ben": re.compile(r"^[\u0980-\u09FF\s]+$"),  # Bengali
        "asm": re.compile(r"^[\u0980-\u09FF\s]+$"),  # Assamese (same as Bengali)
        "guj": re.compile(r"^[\u0A80-\u0AFF\s]+$"),  # Gujarati
        "kan": re.compile(r"^[\u0C80-\u0CFF\s]+$"),  # Kannada
        "mal": re.compile(r"^[\u0D00-\u0D7F\s]+$"),  # Malayalam
        "ori": re.compile(r"^[\u0B00-\u0B7F\s]+$"),  # Oriya/Odia
        "pan": re.compile(r"^[\u0A00-\u0A7F\s]+$"),  # Punjabi (Gurmukhi)
        "tam": re.compile(r"^[\u0B80-\u0BFF\s]+$"),  # Tamil
        "tel": re.compile(r"^[\u0C00-\u0C7F\s]+$"),  # Telugu
        "urd": re.compile(r"^[\u0600-\u06FF\s]+$"),  # Urdu
        "bod": re.compile(r"^[\u0F00-\u0FFF\s]+$"),  # Tibetan
    }
    
    # Initialize input queue and pattern matcher
    input_queue = queue.Queue(maxsize=args.batch_size * 10)  # Buffer some batches
    matcher = PatternMatcher(patterns, batch_size=args.batch_size)
    result_queues = matcher.initialize_queues()
    
    # Create and start matcher worker threads
    matcher_threads = []
    for _ in range(args.workers):
        thread = threading.Thread(target=matcher.matcher_worker, args=(input_queue,))
        thread.daemon = True
        thread.start()
        matcher_threads.append(thread)
    
    # Create and start writer threads for each language
    writer_threads = {}
    for lang in patterns.keys():
        output_path = os.path.join(args.output_dir, lang, args.output_file)
        thread = threading.Thread(
            target=writer_thread_func,
            args=(result_queues[lang], output_path)
        )
        thread.daemon = True
        thread.start()
        writer_threads[lang] = thread
    
    # Process input from stdin
    try:
        for line in sys.stdin.buffer:
            try:
                decoded_line = line.decode('utf-8', errors='ignore').strip()
                if decoded_line:
                    input_queue.put(decoded_line)
            except Exception as e:
                sys.stderr.write(f"Error processing line: {e}\n")
                sys.stderr.flush()
    except KeyboardInterrupt:
        sys.stderr.write("Interrupted by user. Finishing processing...\n")
    except Exception as e:
        sys.stderr.write(f"Error reading from stdin: {e}\n")
        sys.stderr.flush()
    finally:
        # Signal matcher threads to finish
        for _ in range(args.workers):
            input_queue.put(None)
        
        # Wait for all matcher threads to finish
        for thread in matcher_threads:
            thread.join()
        
        # Wait for all writer threads to finish
        for thread in writer_threads.values():
            thread.join()

if __name__ == "__main__":
    main()
