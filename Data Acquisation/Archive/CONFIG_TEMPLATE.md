# Archive Scraper Configuration Template
# Copy these settings to the respective files before running

# =============================================================================
# STEP 1: 1_extract_metadata.py Configuration
# =============================================================================
FOLDER = "metadata_your_subject"           # Output folder for metadata files
COLLECTION = "texts"                       # Collection type (see options below)  
QUERY = "your_search_term"                 # Search query/subject
LANGUAGES = ["English"]                    # Languages to filter

# Collection Options:
# "texts" - General text documents (requires QUERY)
# "booksbylanguage_hindi" - Hindi books collection
# "booksbylanguage_english" - English books collection  
# "booksbylanguage_bengali" - Bengali books collection
# "booksbylanguage_kannada" - Kannada books collection

# =============================================================================
# STEP 2: 2_build_url.py Configuration  
# =============================================================================
METADATA_FOLDER = "metadata_your_subject"  # Must match FOLDER from step 1
OUTPUT_FOLDER = "url"                      # Folder to save URL files
OUTPUT_FILENAME = "archive_urls_your_subject.txt"  # Name of output URL file

# =============================================================================
# STEP 3: 3_download_pdf.py Configuration
# =============================================================================
LANGUAGE = "your_subject"                  # Subject identifier for folder names
URL_FILE = "url/archive_urls_your_subject.txt"  # Must match output from step 2

# Download Settings:
CONFIG = {
    "file_format": "PDF",                  # File format: PDF, EPUB, TXT, etc.
    "concurrent_downloads": 30,            # Reduce for slower internet (try 10-15)
    "concurrent_items": 60,                # Items to process simultaneously
    "batch_size": 300,                     # Batch processing size
    "resume_downloads": True,              # Resume interrupted downloads
}

# =============================================================================
# EXAMPLE CONFIGURATIONS
# =============================================================================

# Example 1: Mathematics Books
# 1_extract_metadata.py:
# FOLDER = "metadata_mathematics"
# COLLECTION = "texts"
# QUERY = "mathematics" 
# LANGUAGES = ["English"]
#
# 2_build_url.py:
# METADATA_FOLDER = "metadata_mathematics"
# OUTPUT_FILENAME = "archive_urls_mathematics.txt"
#
# 3_download_pdf.py:
# LANGUAGE = "mathematics"
# URL_FILE = "url/archive_urls_mathematics.txt"

# Example 2: Hindi Literature
# 1_extract_metadata.py:
# FOLDER = "metadata_hindi_literature"
# COLLECTION = "booksbylanguage_hindi"
# QUERY = "literature"
# LANGUAGES = ["Hindi"]
#
# 2_build_url.py:
# METADATA_FOLDER = "metadata_hindi_literature"  
# OUTPUT_FILENAME = "archive_urls_hindi_literature.txt"
#
# 3_download_pdf.py:
# LANGUAGE = "hindi_literature"
# URL_FILE = "url/archive_urls_hindi_literature.txt"

# Example 3: Computer Science Papers
# 1_extract_metadata.py:
# FOLDER = "metadata_computer_science"
# COLLECTION = "texts"
# QUERY = "computer science"
# LANGUAGES = ["English"]
#
# 2_build_url.py:
# METADATA_FOLDER = "metadata_computer_science"
# OUTPUT_FILENAME = "archive_urls_computer_science.txt"
#
# 3_download_pdf.py:
# LANGUAGE = "computer_science"
# URL_FILE = "url/archive_urls_computer_science.txt"
