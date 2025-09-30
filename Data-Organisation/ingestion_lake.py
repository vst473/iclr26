import os
import boto3
import logging
import json
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from multiprocessing import cpu_count
from botocore.exceptions import NoCredentialsError, ClientError
from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import time

@dataclass
class UploadConfig:
    endpoint: str
    bucket_name: str
    access_key: str
    secret_key: str
    region: str = "us-east-1"
    use_ssl: bool = False
    verify_ssl: bool = False
    max_workers: int = None
    chunk_size: int = 100

class JuiceFSUploader:
    def __init__(self, config: UploadConfig):
        """Initialize JuiceFS uploader with configuration"""
        self.config = config
        if config.max_workers is None:
            self.config.max_workers = min(cpu_count() * 2, 32)  # Optimal for I/O bound tasks
        
    def _create_s3_client(self):
        """Create and configure S3 client for JuiceFS"""
        return boto3.client(
            's3',
            endpoint_url=self.config.endpoint,
            aws_access_key_id=self.config.access_key,
            aws_secret_access_key=self.config.secret_key,
            region_name=self.config.region,
            use_ssl=self.config.use_ssl,
            verify=self.config.verify_ssl
        )
    
    def upload_file(self, local_path: str, s3_key: str) -> bool:
        """Upload a single file to JuiceFS storage"""
        s3_client = self._create_s3_client()  # Create client per thread
        try:
            s3_client.upload_file(local_path, self.config.bucket_name, s3_key)
            logging.info(f"Successfully uploaded: {s3_key}")
            return True
        except FileNotFoundError:
            logging.error(f"File not found: {local_path}")
        except NoCredentialsError:
            logging.error("Credentials not available")
        except Exception as e:
            logging.error(f"Error uploading {local_path}: {str(e)}")
        return False
    
    def generate_presigned_url(self, s3_key: str, expiration: int = 157680000) -> Optional[str]:
        """Generate a presigned URL for the given object key (default: 5 years)"""
        s3_client = self._create_s3_client()
        try:
            url = s3_client.generate_presigned_url(
                'get_object',
                Params={
                    'Bucket': self.config.bucket_name,
                    'Key': s3_key
                },
                ExpiresIn=expiration
            )
            return url
        except ClientError as e:
            logging.error(f"Error generating presigned URL: {e}")
            return None

    def upload_and_get_url(self, local_path: str, s3_key: str, 
                          expiration: int = 157680000) -> Optional[str]:
        """Upload file and return a presigned URL"""
        if self.upload_file(local_path, s3_key):
            return self.generate_presigned_url(s3_key, expiration)
        return None

def process_single_identifier(args):
    """Process a single identifier - designed for multiprocessing"""
    identifier_data, base_dir, metadata, config = args
    
    # Create uploader instance per process
    uploader = JuiceFSUploader(config)
    
    identifier = identifier_data.get('identifier')
    if not identifier:
        return identifier, {"status": "error", "message": "Identifier not found"}
    
    dir_path = os.path.join(base_dir, identifier)
    
    if not os.path.exists(dir_path):
        return identifier, {"status": "error", "message": "Directory not found"}
    
    files = os.listdir(dir_path)
    pdf_zip_files = [f for f in files if f.endswith(('.pdf', '.zip'))]
    
    if not pdf_zip_files:
        return identifier, {"status": "error", "message": "No PDF/ZIP files found"}
    
    # Use ThreadPoolExecutor for I/O bound file uploads within each process
    file_results = []
    with ThreadPoolExecutor(max_workers=4) as executor:
        future_to_file = {}
        
        for file_name in pdf_zip_files:
            local_path = os.path.join(dir_path, file_name)
            s3_key = f"{metadata['data_type']}/{metadata['source']}/{metadata['language']}/{identifier}/{file_name}"
            
            future = executor.submit(uploader.upload_and_get_url, local_path, s3_key)
            future_to_file[future] = (file_name, s3_key)
        
        for future in as_completed(future_to_file):
            file_name, s3_key = future_to_file[future]
            try:
                url = future.result()
                if url:
                    file_results.append({
                        "file_name": file_name,
                        "presigned_url": url,
                        "s3_key": s3_key
                    })
            except Exception as e:
                logging.error(f"Error processing {file_name}: {e}")
    
    result = {
        "status": "success" if file_results else "error",
        "files": file_results
    }
    result.update(identifier_data)
    
    return identifier, result

class OptimizedJuiceFSUploader(JuiceFSUploader):
    def batch_upload_files(self, base_dir: str, identifiers: List[str], 
                          metadata: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        """
        Optimized batch upload using multiprocessing
        """
        results = {}
        
        # Prepare arguments for multiprocessing
        process_args = [
            (identifier_data, base_dir, metadata, self.config)
            for identifier_data in identifiers
        ]
        
        # Process in chunks to manage memory
        chunk_size = self.config.chunk_size
        total_chunks = len(process_args) // chunk_size + (1 if len(process_args) % chunk_size else 0)
        
        logging.info(f"Processing {len(identifiers)} identifiers in {total_chunks} chunks of {chunk_size}")
        
        for chunk_idx in range(total_chunks):
            start_idx = chunk_idx * chunk_size
            end_idx = min(start_idx + chunk_size, len(process_args))
            chunk_args = process_args[start_idx:end_idx]
            
            logging.info(f"Processing chunk {chunk_idx + 1}/{total_chunks} ({len(chunk_args)} items)")
            start_time = time.time()
            
            # Use ProcessPoolExecutor for CPU parallelization
            with ProcessPoolExecutor(max_workers=self.config.max_workers) as executor:
                future_to_args = {
                    executor.submit(process_single_identifier, args): args 
                    for args in chunk_args
                }
                
                for future in as_completed(future_to_args):
                    try:
                        identifier, result = future.result()
                        results[identifier] = result
                    except Exception as e:
                        args = future_to_args[future]
                        identifier = args[0].get('identifier', 'unknown')
                        logging.error(f"Error processing identifier {identifier}: {e}")
                        results[identifier] = {"status": "error", "message": str(e)}
            
            chunk_time = time.time() - start_time
            logging.info(f"Chunk {chunk_idx + 1} completed in {chunk_time:.2f} seconds")
        
        return results

# Example usage
if __name__ == "__main__":
    
    languages = [ #'hindi', "kannada", "telugu", "urdu",
                 "sanskrit", "malyalam"]  # Add more languages as needed
    for lang in languages:
    # Configure logging
        logging.basicConfig(level=logging.INFO,
                            format='%(asctime)s - %(levelname)s - %(message)s')
        
        # Configuration with optimization settings
        config = UploadConfig(
            endpoint='http://localhost:9000',
            bucket_name='myjfs',
            access_key='access_key',
            secret_key='password',
            max_workers=min(cpu_count() * 2, 16),  # Adjust based on your system
            chunk_size=100  # Process 100 identifiers at a time
        )
        
        # Metadata for upload
        metadata = {
            "language": f"{lang}",
            "source": "archive",
            "data_type": "books"
        }
        
        # Load data
        path = f"data_{lang}"
        data = []
        
        for filename in os.listdir(path):
            if filename.endswith('.jsonl'):
                with open(os.path.join(path, filename), 'r') as file:
                    try:
                        json_obj = [json.loads(line.strip()) for line in file if line.strip()]
                        data.extend(json_obj)
                    except json.JSONDecodeError:
                        print(f"Could not parse {filename} as JSON")
        
        
        print(f"Loaded {len(data)} records from metadata files.")
        # Create optimized uploader
        uploader = OptimizedJuiceFSUploader(config)
        
        # Extract identifiers
        data = [d['fields'] for d in data]
        
        # Base directory for files
        base_dir = f"/projects/data/downloads/nauman/archive_scraper/downloads_{lang}"
        if not os.path.exists(base_dir):
            logging.error(f"Base directory does not exist: {base_dir}")
            continue
        # Perform batch upload with timing
        start_time = time.time()
        results = uploader.batch_upload_files(base_dir, data, metadata)
        total_time = time.time() - start_time
        
        logging.info(f"Total upload time: {total_time:.2f} seconds")
        
        with open(f'final_results_{lang}.json', 'w') as f:
            json.dump(results, f, indent=4)
