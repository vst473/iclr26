# Archive.org Scraper - Automated PDF Download Pipeline

A comprehensive Python toolkit for scraping and downloading PDF documents from Archive.org. This pipeline provides both manual step-by-step execution and automated batch processing for extracting metadata, building download URLs, and downloading PDF files from Archive.org collections.

## 🎯 **WHAT THIS TOOL DOES**:
- **Extracts metadata** from Archive.org collections using their API
- **Builds download URLs** for PDF documents from the metadata
- **Downloads PDF files** with concurrent processing and resume capability
- **Supports multiple collections** (texts, books by language, etc.)
- **Handles large-scale operations** with batch processing and error recovery


## ⚡ **QUICK WORKFLOW SUMMARY**:

### Option 1: Manual Steps
1. **Configure** variables in `1_extract_metadata.py`, `2_build_url.py`, `3_download_pdf.py`
2. **Run** `python 1_extract_metadata.py` (fetches metadata, saves JSONL files)
3. **Run** `python 2_build_url.py` (builds URL list from metadata)  
4. **Run** `python 3_download_pdf.py` (downloads PDFs from URLs)

### Option 2: Automated Pipeline
1. **Configure** variables in the three main scripts
2. **Run** `python setup_check.py` (validate configuration)
3. **Run** `python run_pipeline.py` (runs complete pipeline)

**Note**: `get_metadata.py` is **NEVER run directly** - it only contains helper functions!

## 📁 File Structure

```
archive_scraper_cleaned/
├── README.md                    # Complete documentation
├── PROJECT.md                   # Quick project overview
├── CONFIG_TEMPLATE.md           # Configuration examples
├── requirements.txt             # Python dependencies
├── .gitignore                   # Git ignore patterns
├── setup_check.py              # Validate setup and configuration
├── run_pipeline.py             # Complete pipeline runner
├── get_metadata.py             # FUNCTIONS ONLY - API helpers (never run directly)
├── 1_extract_metadata.py       # STEP 1 - Fetches and saves metadata 
├── 2_build_url.py              # STEP 2 - Build download URLs from metadata
└── 3_download_pdf.py           # STEP 3 - Download PDF files from URLs
```

## 🔄 Pipeline Workflow

### Helper: Get Metadata (`get_metadata.py`)
- **Purpose**: **FUNCTION LIBRARY ONLY** - Contains helper functions for API calls
- **Note**: ⚠️ **This file is NEVER run directly!** It only provides functions.
- **Functions**:
  - `fetch_data_with_params()` - Fetch data from Archive.org API
  - `fetch_aggregation()` - Get aggregation data (years, subjects)
  - `get_bucket_aggregation()` - Process aggregation buckets
- **Used by**: Step 1 (1_extract_metadata.py)

### Step 1: Extract Metadata (`1_extract_metadata.py`)
- **Purpose**: **MAIN SCRIPT** - Fetches and processes metadata from Archive.org
- **Function**: Uses functions from get_metadata.py to fetch and save metadata
- **Input**: Configuration (FOLDER, COLLECTION, QUERY, LANGUAGES)
- **Output**: Structured metadata files (JSONL format)
- **Key Features**:
  - Handles pagination
  - Subject filtering
  - Organized output structure
  - Error handling

### Step 2: Build URLs (`2_build_url.py`)
- **Purpose**: Generate download URLs from processed metadata
- **Input**: Metadata JSONL files from Step 1
- **Output**: Text files containing Archive.org URLs
- **Key Features**:
  - Extracts identifiers from metadata
  - Builds proper Archive.org URLs
  - Removes duplicates
  - Batch processing

### Step 3: Download PDFs (`3_download_pdf.py`)
- **Purpose**: Download PDF files from the generated URLs
- **Input**: URL lists from Step 2
- **Output**: Downloaded PDF files
- **Key Features**:
  - Asynchronous downloading
  - Resume interrupted downloads
  - Concurrent downloads (configurable)
  - Proxy support
  - Progress tracking
  - Error logging

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Validate Setup (Optional)
```bash
python setup_check.py
# Checks dependencies and configuration
```

### 3. Run Complete Pipeline (Optional)
```bash
python run_pipeline.py --config your_project_name
# Runs all three steps automatically with progress tracking
```

## 📝 Configuration Examples

### Example 1: Scraping Mathematics Books
```python
# 1_extract_metadata.py
FOLDER = "metadata_mathematics"
COLLECTION = "texts"  
QUERY = "mathematics"
LANGUAGES = ["English"]

# 2_build_url.py  
METADATA_FOLDER = "metadata_mathematics"
OUTPUT_FILENAME = "archive_urls_mathematics.txt"

# 3_download_pdf.py
LANGUAGE = "mathematics"
URL_FILE = "url/archive_urls_mathematics.txt"
```

### Example 2: Scraping Hindi Books
```python
# 1_extract_metadata.py
FOLDER = "metadata_hindi_books"
COLLECTION = "booksbylanguage_hindi"
QUERY = ""  # Empty for entire collection
LANGUAGES = ["Hindi"]

# 2_build_url.py
METADATA_FOLDER = "metadata_hindi_books" 
OUTPUT_FILENAME = "archive_urls_hindi.txt"

# 3_download_pdf.py
LANGUAGE = "hindi"
URL_FILE = "url/archive_urls_hindi.txt"
```

### 2. Configure Variables
**IMPORTANT**: The scripts come with example configuration values. Customize them for your specific use case!

1. **Step 1**: Edit `FOLDER`, `COLLECTION`, `QUERY`, `LANGUAGES` in `1_extract_metadata.py`
2. **Step 2**: Edit `METADATA_FOLDER`, `OUTPUT_FILENAME` in `2_build_url.py` 
3. **Step 3**: Edit `LANGUAGE`, `URL_FILE` in `3_download_pdf.py`
4. **Helper**: ⚠️ **DO NOT CONFIGURE** - `get_metadata.py` contains functions only

💡 **Current example config**: Scrapes "science" documents from Archive.org

### 3. Run the Pipeline

#### Step 1: Extract Metadata (MAIN SCRIPT)
```bash
python 1_extract_metadata.py
# This script:
# 1. Imports functions from get_metadata.py
# 2. Fetches metadata from Archive.org API
# 3. Processes and saves data to JSONL files
# Output: Creates metadata JSONL files in the specified FOLDER
```

#### Step 2: Build URLs  
```bash
python 2_build_url.py
# Output: Creates URL text file from metadata files
```

#### Step 3: Download PDFs
```bash
python 3_download_pdf.py
# Output: Downloads PDF files using the URL file
```

#### Helper File: ⚠️ **DO NOT RUN** (functions only)
```bash
# get_metadata.py contains ONLY helper functions
# It is imported and used by 1_extract_metadata.py
# NEVER run this file directly!
```

## ⚙️ Configuration

**IMPORTANT**: Before running any script, you MUST update the configuration variables at the top of each file!

### Step 1: Extract Metadata Configuration (`1_extract_metadata.py`)
```python
FOLDER = "metadata_maths"                    # UPDATE: Output folder for metadata
COLLECTION = "texts"                         # UPDATE: Collection type
QUERY = "mathematics"                        # UPDATE: Search query/subject
LANGUAGES = ["English"]                      # UPDATE: Languages to filter
```

**Collection Types**:
- `"texts"` - General text documents (search-based)
- `"booksbylanguage_hindi"` - Hindi books collection  
- `"booksbylanguage_english"` - English books collection
- `"booksbylanguage_bengali"` - Bengali books collection
- `"booksbylanguage_kannada"` - Kannada books collection

### Step 2: Build URLs Configuration (`2_build_url.py`)
```python
METADATA_FOLDER = "metadata_maths"           # UPDATE: Must match FOLDER from step 1
OUTPUT_FOLDER = "url"                        # UPDATE: Folder to save URL files
OUTPUT_FILENAME = "archive_urls_maths.txt"  # UPDATE: Name of output URL file
```

### Step 3: Download PDFs Configuration (`3_download_pdf.py`)
```python
LANGUAGE = "maths"                           # UPDATE: Subject identifier
URL_FILE = "url/archive_urls_maths.txt"     # UPDATE: Must match output from step 2

CONFIG = {
    "file_format": "PDF",                    # UPDATE: PDF, EPUB, TXT, etc.
    "concurrent_downloads": 30,              # UPDATE: Reduce for slow internet
    "concurrent_items": 60,                  # UPDATE: Items to process simultaneously
    "batch_size": 300,                       # Batch processing size
    "resume_downloads": True,                # Resume interrupted downloads
}
```

## 📊 Output Structure

```
downloads_<language>/           # Downloaded files
├── pdf_files/                 # PDF documents
├── metadata/                  # Item metadata
└── failed_downloads.log       # Failed download log

metadata_<subject>/            # Extracted metadata
├── chunk1/                    # Metadata chunks
│   ├── metadata_archive_*.jsonl
│   └── ...
└── ...

url/                          # Generated URLs
├── archive_urls_<language>.txt
└── ...
```

## 🔧 Customization

### Adding New Collections
1. Modify the collection parameter in `1_get_metadata.py`
2. Update subject filters in `2_extract_metadata.py`
3. Adjust URL patterns if needed in `3_build_url.py`

### Changing File Formats
The downloader supports multiple formats:
- PDF (default)
- EPUB
- TXT
- Other Archive.org formats

Edit the `file_format` in the CONFIG section of `4_download_pdf.py`.

### Proxy Configuration
For large-scale scraping, configure proxies:
1. Create `proxies.txt` with proxy URLs
2. Create `user_agents.txt` with user agent strings
3. Enable proxy usage in the download script

## 📝 Logging

All scripts generate detailed logs:
- Console output for real-time monitoring
- Log files for debugging and tracking
- Progress bars for download tracking

## ⚠️ Important Notes

1. **Rate Limiting**: Be respectful of Archive.org's servers
2. **Legal Compliance**: Ensure you have rights to download content
3. **Storage Space**: PDF downloads can consume significant disk space
4. **Network**: Stable internet connection recommended for large downloads

## 🐛 Troubleshooting

### Common Issues:
1. **Connection Errors**: Check internet connection and proxy settings
2. **Permission Errors**: Ensure write permissions for output directories
3. **Memory Issues**: Reduce concurrent downloads for large files
4. **API Limits**: Implement delays between requests if needed

### Debug Mode:
Enable debug logging by changing the logging level in each script:
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📄 License

This tool is for educational and research purposes. Respect Archive.org's terms of service and content licensing.

---

**Created**: September 30, 2025  
**Version**: 1.0  
**Maintained by**: Archive Scraper Team
