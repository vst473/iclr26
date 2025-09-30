import asyncio
import aiohttp
import aiofiles
import json
import random
import io
import sys
import os
import multiprocessing
import logging
import hashlib
import pickle
from urllib.parse import urlparse
from warcio.archiveiterator import ArchiveIterator
import justext
import trafilatura
import fasttext
from langdetect import detect
from langdetect.lang_detect_exception import LangDetectException
import pandas as pd
import datetime
import re

# Create output directory if it doesn't exist
os.makedirs('./warc_text_output', exist_ok=True)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('./warc_text_output/processing.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
# Set PARQUET_PATH to a local file path or a URL( parquet file from Common Crawl index)
PARQUET_PATH = "part-00288-88b30a59-3c73-48ba-a167-077611bfd245.c000.gz.parquet"
if not PARQUET_PATH:
    logger.error("Usage: python script.py <parquet_file_path_or_url>")
    sys.exit(1)

MAX_CONCURRENT = 100
logger.info(f"Using MAX_CONCURRENT = {MAX_CONCURRENT}")

MY_AGENT = "simple-warc-extractor/1.0 (Text extraction; research@example.com)"
OUTPUT_FILE = "./warc_text_output/extracted_text.jsonl"
OUTPUT_FILE_SCORED = "./warc_text_output/extracted_text_scored.jsonl"
ERROR_LOG_FILE = "./warc_text_output/extraction_errors.log"
PROCESSED_URLS_FILE = "./warc_text_output/processed_urls.pkl"
MAX_RETRIES = 3

# Global variables for fasttext model and processed URLs tracking
fasttext_model = None
processed_urls = set()

# Simple quality filters
MIN_TEXT_LENGTH = 20  # Reduced from 50
MAX_TEXT_LENGTH = 500000
MIN_WORD_COUNT = 5    # Reduced from 10

def load_fasttext_model():
    """Download and load fasttext language detection model."""
    global fasttext_model
    if fasttext_model is None:
        try:
            # Download model if it doesn't exist
            model_path = "./warc_text_output/lid.176.bin"
            if not os.path.exists(model_path):
                logger.info("Downloading fasttext language detection model...")
                import urllib.request
                urllib.request.urlretrieve(
                    "https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin",
                    model_path
                )
            fasttext_model = fasttext.load_model(model_path)
            logger.info("FastText model loaded successfully")
        except Exception as e:
            logger.warning(f"Failed to load fasttext model: {e}, falling back to langdetect")
            fasttext_model = None
    return fasttext_model

def detect_language_fast(text):
    """Fast language detection using multiple fallback methods."""
    try:
        # Clean text for language detection
        clean_text = re.sub(r'[^\w\s]', ' ', text[:1000]).strip()
        if not clean_text:
            return "unknown"
        
        # Method 1: FastText (if available)
        if fasttext_model:
            try:
                predictions = fasttext_model.predict(clean_text, k=1)
                lang_code = predictions[0][0].replace('__label__', '')
                confidence = predictions[1][0]
                if confidence > 0.3:
                    return lang_code
            except:
                pass
        
        # Method 2: langdetect
        try:
            detected = detect(clean_text)
            if detected:
                return detected
        except:
            pass
        
        # Method 3: Simple English word frequency check
        english_words = {
            'the', 'and', 'a', 'to', 'of', 'in', 'is', 'for', 'with', 'this', 'that', 'have', 'are', 'was', 'be',
            'on', 'as', 'by', 'they', 'we', 'an', 'will', 'can', 'or', 'from', 'has', 'had', 'but', 'not', 'you',
            'all', 'were', 'been', 'their', 'said', 'each', 'which', 'do', 'how', 'if', 'about', 'up', 'out', 'time',
            'there', 'use', 'her', 'would', 'make', 'like', 'into', 'him', 'two', 'more', 'very', 'what', 'know'
        }
        
        words = re.findall(r'\b[a-zA-Z]+\b', clean_text.lower())
        if words and len(words) >= 10:  # Need at least 10 words for reliable detection
            english_count = sum(1 for word in words[:50] if word in english_words)
            english_ratio = english_count / min(len(words), 50)
            
            # If more than 30% are common English words, likely English
            if english_ratio > 0.3:
                return "en"
        
        return "unknown"
    except Exception:
        return "unknown"

def calculate_english_score(text):
    """Calculate English quality score (0-100) based on various factors."""
    if not text or len(text.strip()) < 10:
        return 0.0
    
    score = 0.0
    
    # Factor 1: English word ratio (0-30 points)
    common_english_words = {
        'the', 'and', 'a', 'to', 'of', 'in', 'i', 'you', 'it', 'have', 'for', 'not', 'with', 'he', 'as', 'on',
        'do', 'his', 'by', 'but', 'they', 'this', 'from', 'or', 'she', 'an', 'be', 'we', 'can', 'out', 'other',
        'were', 'all', 'your', 'when', 'up', 'use', 'word', 'how', 'said', 'each', 'which', 'their', 'time',
        'will', 'about', 'if', 'there', 'many', 'some', 'has', 'would', 'like', 'into', 'them', 'see', 'him'
    }
    
    words = re.findall(r'\b[a-zA-Z]+\b', text.lower())
    if words:
        english_word_count = sum(1 for word in words if word in common_english_words)
        english_ratio = english_word_count / len(words)
        score += min(30, english_ratio * 30)
    
    # Factor 2: Sentence structure (0-25 points)
    sentences = re.split(r'[.!?]+', text)
    valid_sentences = [s.strip() for s in sentences if len(s.strip()) > 10]
    if valid_sentences:
        good_sentences = sum(1 for s in valid_sentences if 5 <= len(s.split()) <= 50)
        if len(valid_sentences) > 0:
            score += (good_sentences / len(valid_sentences)) * 25
    
    # Factor 3: Proper capitalization (0-15 points)
    if valid_sentences:
        capitalized_starts = sum(1 for s in valid_sentences if s and s[0].isupper())
        score += (capitalized_starts / len(valid_sentences)) * 15
    
    # Factor 4: Punctuation usage (0-15 points)
    punct_count = len(re.findall(r'[.!?,;:]', text))
    word_count = len(words) if words else 1
    punct_ratio = punct_count / word_count
    if 0.05 <= punct_ratio <= 0.2:
        score += 15
    elif punct_ratio > 0:
        score += 8
    
    # Factor 5: Avoid spam patterns (0-15 points)
    spam_patterns = [r'click here', r'buy now', r'free.*trial', r'limited.*time', r'call.*now']
    spam_count = sum(len(re.findall(pattern, text, re.IGNORECASE)) for pattern in spam_patterns)
    if spam_count == 0:
        score += 15
    elif spam_count <= 2:
        score += 10
    elif spam_count <= 5:
        score += 5
    
    return round(min(100, score), 2)

def load_processed_urls():
    """Load previously processed URLs from disk."""
    global processed_urls
    try:
        if os.path.exists(PROCESSED_URLS_FILE):
            with open(PROCESSED_URLS_FILE, 'rb') as f:
                processed_urls = pickle.load(f)
            logger.info(f"Loaded {len(processed_urls)} previously processed URLs")
        else:
            processed_urls = set()
    except Exception as e:
        logger.error(f"Error loading processed URLs: {e}")
        processed_urls = set()

def save_processed_urls():
    """Save processed URLs to disk."""
    try:
        with open(PROCESSED_URLS_FILE, 'wb') as f:
            pickle.dump(processed_urls, f)
        logger.info(f"Saved {len(processed_urls)} processed URLs")
    except Exception as e:
        logger.error(f"Error saving processed URLs: {e}")

def is_already_processed(url):
    """Check if URL has already been processed."""
    return url in processed_urls

def mark_as_processed(url):
    """Mark URL as processed."""
    processed_urls.add(url)

def is_good_quality_text(text: str, url: str = "") -> bool:
    """Simple quality check for extracted text."""
    if not text or not text.strip():
        return False
    
    # Basic length checks
    if len(text) < MIN_TEXT_LENGTH or len(text) > MAX_TEXT_LENGTH:
        return False
    
    words = text.split()
    if len(words) < MIN_WORD_COUNT:
        return False
    
    # Check for error patterns
    error_patterns = [
        "Notice: Function", "Fatal error:", "Warning:", "Parse error:",
        "Call to undefined", "Permission denied", "404 Not Found", 
        "403 Forbidden", "500 Internal Server Error", "Page not found",
        "Access denied", "Under construction", "Coming soon",
        "JavaScript is disabled", "Enable JavaScript", "Cookie policy"
    ]
    
    text_lower = text.lower()
    if any(pattern.lower() in text_lower for pattern in error_patterns):
        return False
    
    # Check alphanumeric ratio
    alphanumeric_chars = sum(1 for c in text if c.isalnum())
    if alphanumeric_chars / len(text) < 0.7:
        return False
    
    # Check for excessive repeated content
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if len(lines) > 10:  # Only check if we have enough lines
        if len(set(lines)) / len(lines) < 0.3:  # Too many duplicate lines
            return False
    
    # Check for spam patterns
    spam_indicators = ["click here", "buy now", "special offer", "limited time", 
                      "call now", "visit our website", "subscribe now"]
    spam_count = sum(1 for indicator in spam_indicators if indicator in text_lower)
    if spam_count > 3:
        return False
    
    return True

async def download_parquet(session, url):
    """Download parquet file from URL and return as bytes."""
    logger.info(f"Downloading parquet file from {url}")
    try:
        async with session.get(url, headers={"user-agent": MY_AGENT}) as response:
            if response.status == 200:
                data = await response.read()
                logger.info(f"Download completed ({len(data)} bytes)")
                return data
            else:
                logger.error(f"Failed to download file: HTTP {response.status}")
                return None
    except Exception as e:
        logger.error(f"Exception during download: {e}")
        return None

async def read_parquet(session):
    """Read parquet file from local path or URL and return records as list of dicts."""
    try:
        if PARQUET_PATH.startswith("http://") or PARQUET_PATH.startswith("https://"):
            data = await download_parquet(session, PARQUET_PATH)
            if data is None:
                logger.error("Failed to download parquet file")
                return []
            buffer = io.BytesIO(data)
            df = pd.read_parquet(buffer)
        else:
            logger.info(f"Reading parquet file from local path: {PARQUET_PATH}")
            df = pd.read_parquet(PARQUET_PATH)
        
        # Remove duplicates and limit for testing
        # df = df.drop_duplicates(subset=['url'])
        logger.info(f"Loaded {len(df)} unique records from parquet")
        
        # Apply filtering on dataframe first
        initial_count = len(df)
        
        # Filter by required fields
        required_fields = ["warc_record_offset", "warc_record_length", "warc_filename"]
        for field in required_fields:
            df = df[df[field].notna()]

        print(df.columns)
        df = df[(df.content_languages.str.contains('eng', case=False, na=False)) ]
        
        logger.info(f"Loaded {len(df)} unique records from parquet")
        logger.info(f"After required fields filter: {len(df)} records (removed {initial_count - len(df)})")
        

        # Filter by mime type if available
        if 'content_mime_detected' in df.columns:
            mime_count = len(df)
            df = df[(df['content_mime_detected'].isna()) | 
                   (df['content_mime_detected'].str.contains('text/html', case=False, na=True))]
            logger.info(f"After mime type filter: {len(df)} records (removed {mime_count - len(df)})")
        
        # Limit to first 10 for testing (remove this line for full processing)
        # if len(df) > 10:
        #     df = df.iloc[:5000]
        #     logger.info("Limited to first 10 records for testing")
        
        return df.to_dict(orient="records")
    except Exception as e:
        logger.error(f"Error reading parquet file: {e}")
        return []

async def download_warc_content(session, record, semaphore):
    """Download WARC content separately from processing."""
    url = record.get('url', 'unknown')
    
    # Skip if already processed
    if is_already_processed(url):
        logger.debug(f"Skipping already processed URL: {url}")
        return None
    
    # Get metadata for later use
    languages = record.get('content_languages', '')
    mime = record.get('content_mime_detected', '')
    
    offset = int(record["warc_record_offset"])
    length = int(record["warc_record_length"])
    filename = record["warc_filename"]
    s3_url = f"https://data.commoncrawl.org/{filename}"
    byte_range = f"bytes={offset}-{offset + length - 1}"
    
    headers = {
        "user-agent": MY_AGENT,
        "Range": byte_range
    }
    
    async with semaphore:
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                async with session.get(s3_url, headers=headers) as response:
                    if response.status == 206:
                        data = await response.content.read()
                        stream = ArchiveIterator(io.BytesIO(data))
                        
                        for warc_record in stream:
                            if warc_record.rec_type == "response":
                                html_bytes = warc_record.content_stream().read()
                                html = html_bytes.decode("utf-8", errors="ignore")
                                
                                # Extract text using Justext first
                                paragraphs = justext.justext(html, justext.get_stoplist("English"))
                                text = "\n\n".join(p.text for p in paragraphs 
                                                 if not p.is_boilerplate and p.text.strip())
                                
                                # If justext gives us nothing, try trafilatura
                                if not text.strip():
                                    text = trafilatura.extract(html) or ""
                                    if text.strip():
                                        logger.info(f"Used trafilatura for {url} - got {len(text)} chars")
                                
                                # Always return a result, even if text is empty
                                text = text.strip()
                                
                                # Fast language detection
                                detected_lang = "unknown"
                                if text.strip():
                                    detected_lang = detect_language_fast(text)
                                    logger.debug(f"Detected language for {url}: {detected_lang}")
                                else:
                                    logger.info(f"Empty text from {url}")
                                
                                word_count = len(text.split()) if text else 0
                                char_count = len(text)
                                
                                if text:
                                    logger.info(f"Successfully extracted text from {url} "
                                              f"({word_count} words, {char_count} chars)")
                                else:
                                    logger.info(f"Blank text from {url}")
                                
                                return {
                                    "url": url,
                                    "warc_filename": filename,
                                    "text": text,
                                    "word_count": word_count,
                                    "char_count": char_count,
                                    "content_languages": languages,
                                    "content_mime_detected": mime,
                                    "detected_language": detected_lang
                                }
                    
                    elif response.status in [403, 429, 503]:
                        wait_time = 2 ** attempt * random.uniform(10, 20)
                        logger.warning(f"Rate limited or access denied ({response.status}) for {url}, "
                                     f"retry {attempt}/{MAX_RETRIES}, waiting {wait_time:.2f}s")
                        await asyncio.sleep(wait_time)
                    else:
                        logger.info(f"Failed to fetch {url}: HTTP {response.status}")
                        return None
            
            except Exception as e:
                wait_time =( 2 ** attempt) + random.uniform(10, 20)
                logger.warning(f"Exception for {url}: {e}, "
                             f"retry {attempt}/{MAX_RETRIES}, waiting {wait_time:.2f}s")
                await asyncio.sleep(wait_time)
    
    # Log final failure
    logger.error(f"Failed to extract from {url} after {MAX_RETRIES} retries")
    try:
        async with aiofiles.open(ERROR_LOG_FILE, "a") as err_file:
            await err_file.write(f"Failed: {url}\n")
    except:
        pass
    return None

def process_extracted_content(content_data):
    """Process extracted content and create both direct and scored versions."""
    if not content_data or not content_data.get('text'):
        return None, None
    
    url = content_data['url']
    text = content_data['text']
    
    # Mark as processed
    mark_as_processed(url)
    
    # Create direct version (fast)
    direct_result = {
        "url": url,
        "warc_filename": content_data['warc_filename'],
        "text": text,
        "word_count": content_data['word_count'],
        "char_count": content_data['char_count'],
        "content_languages": content_data['content_languages'],
        "content_mime_detected": content_data['content_mime_detected'],
        "detected_language": content_data['detected_language']
    }
    
    # Create scored version (slower)
    scored_result = direct_result.copy()
    if text:
        # Always calculate English score if there's text
        # The scoring function will determine the quality internally
        scored_result['english_score'] = calculate_english_score(text)
    else:
        scored_result['english_score'] = 0.0
    
    return direct_result, scored_result

async def process_record(session, record, file_lock, semaphore):
    """Process one record and save both direct and scored results."""
    try:
        # Download and extract content
        content_data = await download_warc_content(session, record, semaphore)
        if content_data is not None:
            # Process content to create both versions
            direct_result, scored_result = process_extracted_content(content_data)
            
            if direct_result:
                # Save both versions
                async with file_lock:
                    # Save direct version (fast)
                    async with aiofiles.open(OUTPUT_FILE, "a", encoding="utf-8") as f_out:
                        await f_out.write(json.dumps(direct_result, ensure_ascii=False) + "\n")
                    
                    # Save scored version (with English score)
                    async with aiofiles.open(OUTPUT_FILE_SCORED, "a", encoding="utf-8") as f_scored:
                        await f_scored.write(json.dumps(scored_result, ensure_ascii=False) + "\n")
                
                if direct_result['text']:
                    logger.debug(f"Saved text from {direct_result['url']}")
                else:
                    logger.debug(f"Saved blank record from {direct_result['url']}")
                return True
        return False
    except Exception as e:
        logger.error(f"Error processing record {record.get('url', '')}: {e}")
        return False

async def main():
    """Main function to process records from parquet file."""
    start_time = datetime.datetime.now()
    logger.info(f"Started at {start_time}")
    
    # Initialize fasttext model and load processed URLs
    logger.info("Loading fasttext model and processed URLs...")
    load_fasttext_model()
    load_processed_urls()
    
    file_lock = asyncio.Lock()
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    success_count = 0
    total_processed = 0
    
    try:
        async with aiohttp.ClientSession() as session:
            # Load parquet file
            records = await read_parquet(session)
            if not records:
                logger.error("No records loaded from parquet file. Exiting.")
                return
            
            logger.info(f"Processing {len(records)} records with max concurrency: {MAX_CONCURRENT}")
            
            # Process all records
            tasks = []
            for record in records:
                task = process_record(session, record, file_lock, semaphore)
                tasks.append(task)
                total_processed += 1
            
            # Execute all tasks
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Count successes
            success_count = sum(1 for result in results 
                              if result is not None and not isinstance(result, Exception))
            
        end_time = datetime.datetime.now()
        elapsed = end_time - start_time
        
        # Save processed URLs
        save_processed_urls()
        
        logger.info(f"Finished at {end_time}")
        logger.info(f"Total elapsed time: {elapsed}")
        logger.info(f"Successfully processed {success_count} out of {total_processed} records")
        logger.info(f"Success rate: {success_count/total_processed*100:.1f}%")
        logger.info(f"Direct results saved to: {OUTPUT_FILE}")
        logger.info(f"Scored results saved to: {OUTPUT_FILE_SCORED}")
        logger.info(f"Errors logged to: {ERROR_LOG_FILE}")
        logger.info(f"Total processed URLs tracked: {len(processed_urls)}")
        
        # Print some stats
        if success_count > 0:
            logger.info(f"Check {OUTPUT_FILE} for direct extracted text data")
            logger.info(f"Check {OUTPUT_FILE_SCORED} for scored text data")
        
    except Exception as e:
        logger.error(f"Fatal error: {e}")
        # Save processed URLs even on error
        save_processed_urls()
        raise

if __name__ == "__main__":
    asyncio.run(main())