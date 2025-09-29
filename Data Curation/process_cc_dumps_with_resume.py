import argparse
import logging
import os
import resource
import signal
import subprocess
import glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Optional

from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Update the COMMAND_TEMPLATE to use wget instead of curl
COMMAND_TEMPLATE = """
bash -c 'set -o pipefail; \
wget -q -O - {url} | \
pigz -d | \
python3 streaming_regex.py \
--output-dir {output_folder} \
--output-file {output_file}'
"""

# List of language subfolders
LANGUAGE_SUBFOLDERS = [
    'tel', 'asm', 'pan', 'ben', 'tam', 'mal', 
    'guj', 'ori', 'kan', 'urd', 'hin', 'bod'
]

def set_resource_limits():
    """Set resource limits for the process to prevent memory issues"""
    # Set soft limit for number of file descriptors
    resource.setrlimit(
        resource.RLIMIT_NOFILE, 
        (4096, resource.getrlimit(resource.RLIMIT_NOFILE)[1])
    )
    
    # Optional: Set memory limit per process (8GB)
    # resource.setrlimit(resource.RLIMIT_AS, (8 * 1024 * 1024 * 1024, -1))

def check_file_exists(output_folder: str, output_filename: str) -> bool:
    """
    Check if the file exists in any of the language subfolders
    
    Args:
        output_folder: Base output folder
        output_filename: Filename to check
        
    Returns:
        bool: True if file exists in any subfolder, False otherwise
    """
    # First check in the main output folder
    if os.path.exists(os.path.join(output_folder, output_filename)):
        return True
        
    # Then check in all language subfolders
    for lang in LANGUAGE_SUBFOLDERS:
        lang_path = os.path.join(output_folder, lang)
        if os.path.exists(os.path.join(lang_path, output_filename)):
            return True
            
    # Check if the file exists in any subfolder using glob
    # This handles cases where files might be in unexpected subfolders
    glob_pattern = os.path.join(output_folder, "**", output_filename)
    matching_files = glob.glob(glob_pattern, recursive=True)
    
    return len(matching_files) > 0

def process_url(url: str, output_folder: str, timeout: int = 3600) -> Dict:
    """
    Process a single URL with proper resource management and error handling.
    Skip processing if output file already exists in any language subfolder.
    
    Args:
        url: The URL to process
        output_folder: Where to save the output
        timeout: Maximum seconds to allow for processing
        
    Returns:
        Dictionary containing status and details
    """
    # Create a cleaner filename from the URL
    output_filename = f"regex_{url.split('segments')[-1].replace('.ec2.internal.warc.wet.gz', '').replace('/', '_').replace('.','_').replace('-','_')}.txt"
    
    result = {
        'url': url,
        'success': False,
        'error': None,
        'output_file': output_filename
    }

    # Check if the file already exists in any of the language subfolders
    if check_file_exists(output_folder, output_filename):
        result['success'] = True
        result['error'] = "File already exists in one of the language subfolders, skipping"
        return result

    # Ensure process is None initially for safe cleanup
    process = None
    
    try:
        # Set resource limits for this process
        set_resource_limits()
        
        # Ensure output directory exists
        os.makedirs(output_folder, exist_ok=True)
        
        # Prepare command with proper argument formatting
        command = COMMAND_TEMPLATE.format(
            url=url,
            output_folder=output_folder,
            output_file=output_filename
        )
        
        # Start process with timeout control
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            preexec_fn=os.setsid,  # Create new process group
            bufsize=1,  # Line buffered
            universal_newlines=True  # Text mode
        )
        
        # Capture output with timeout
        try:
            stdout, stderr = process.communicate(timeout=timeout)
            if process.returncode == 0:
                result['success'] = True
            else:
                result['error'] = f"Process failed with return code {process.returncode}: {stderr}"
        except subprocess.TimeoutExpired:
            # Kill the process group if timeout occurs
            os.killpg(os.getpgid(process.pid), signal.SIGKILL)
            result['error'] = f"Process timed out after {timeout} seconds"
            
    except Exception as e:
        result['error'] = str(e)
    finally:
        # Ensure process group is terminated if still running
        if process is not None:
            try:
                os.killpg(os.getpgid(process.pid), signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass
    
    return result

def fetch_url_list(input_url: str, output_folder: str) -> Optional[List[str]]:
    """Download and parse the URL list in a streaming manner using wget."""
    logging.info("Fetching URL list...")
    version = input_url.split('/')[5]
    wet_urls = []
    
    try:
        # Stream download and process the URL list without saving to disk, using wget
        process = subprocess.Popen(
            f"wget -q -O - {input_url} | gunzip",
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logging.error(f"Failed to download URL list: {stderr}")
            return None
            
        # Process each line and add prefix
        for line in stdout.splitlines():
            if line.strip():
                wet_urls.append(f"https://data.commoncrawl.org/{line.strip()}")
                
        logging.info(f"Found {len(wet_urls)} URLs to process")
        return wet_urls
        
    except Exception as e:
        logging.error(f"Error downloading URL list: {e}")
        return None

def parallel_processing(input_url: str, output_folder: str, num_workers: int, batch_size: int = 20):
    """
    Main processing function with improved resource management and progress tracking.
    
    Args:
        input_url: URL to the list of files to process
        output_folder: Where to save output files
        num_workers: Number of parallel workers
        batch_size: Number of URLs to process in each batch
    """
    logging.info('Starting parallel processing...')
    os.makedirs(output_folder, exist_ok=True)

    # Create language subfolders if they don't exist
    for lang in LANGUAGE_SUBFOLDERS:
        lang_path = os.path.join(output_folder, lang)
        os.makedirs(lang_path, exist_ok=True)
        logging.info(f"Ensured language folder exists: {lang_path}")

    # Get URL list in a streaming manner
    wet_urls = fetch_url_list(input_url, output_folder)
    if not wet_urls:
        return

    total_urls = len(wet_urls)
    logging.info(f"Processing {total_urls} URLs with {num_workers} workers...")

    # Track results
    successful = []
    skipped = []
    failed = []
    
    # Create progress bar
    pbar = tqdm(total=total_urls, desc="Processing files", unit="file")
    
    # Process in batches to manage memory
    batch_size = min(batch_size, total_urls)
    
    with ProcessPoolExecutor(max_workers=num_workers) as executor:
        for i in range(0, total_urls, batch_size):
            batch = wet_urls[i:i + batch_size]
            futures = [executor.submit(process_url, url, output_folder) for url in batch]
            
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result['success']:
                        if result['error'] and "already exists" in result['error']:
                            skipped.append(result['url'])
                            logging.info(f"Skipped {result['url']}: {result['error']}")
                        else:
                            successful.append(result['url'])
                    else:
                        failed.append((result['url'], result['error']))
                        logging.error(f"Failed to process {result['url']}: {result['error']}")
                except Exception as e:
                    logging.error(f"Unexpected error in worker: {e}")
                finally:
                    pbar.update(1)
                    pbar.set_postfix({
                        'succeeded': len(successful),
                        'skipped': len(skipped),
                        'failed': len(failed)
                    })
            
            # Optional: Force garbage collection to prevent memory issues
            import gc
            gc.collect()
            
    pbar.close()

    # Final report
    logging.info("\nProcessing completed:")
    logging.info(f"- Total URLs: {total_urls}")
    logging.info(f"- Successfully processed: {len(successful)}")
    logging.info(f"- Skipped (already exist): {len(skipped)}")
    logging.info(f"- Failed: {len(failed)}")
    
    # Write failed URLs to file for potential retry
    if failed:
        failed_file = os.path.join(output_folder, "failed_urls.txt")
        with open(failed_file, 'w') as f:
            for url, error in failed:
                f.write(f"{url}\t{error}\n")
        logging.info(f"\nFailed URLs have been saved to: {failed_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract pure Hindi lines by streaming zip files for unzipping then to regex for Hindi identification."
    )
    parser.add_argument("url", type=str, help="URL for wet.paths.gz file")
    parser.add_argument("output_folder", type=str, help="Path to the output folder")
    parser.add_argument("--workers", type=int, default=8, help="Number of worker processes")
    parser.add_argument("--batch-size", type=int, default=20, help="Batch size for processing")
    parser.add_argument("--timeout", type=int, default=3600, help="Timeout per URL in seconds")

    args = parser.parse_args()
    parallel_processing(args.url, args.output_folder, num_workers=args.workers, batch_size=args.batch_size)

    # python3 process_final.py https://data.commoncrawl.org/wet.paths.gz /nfs/shyam.pawar/cc_download/CC-MAIN-2025-08/
