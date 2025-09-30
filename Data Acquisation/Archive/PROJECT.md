# Archive.org Scraper

A clean, organized Python pipeline for scraping documents from Archive.org collections.

## 🚀 Quick Start

1. **Install dependencies**: `pip install -r requirements.txt`
2. **Configure variables** in each script (see CONFIG_TEMPLATE.md)
3. **Run the pipeline**:
   ```bash
   python 1_extract_metadata.py  # Fetch metadata
   python 2_build_url.py         # Build URL list
   python 3_download_pdf.py      # Download files
   ```

## 📁 Files

- `get_metadata.py` - Helper functions (never run directly)
- `1_extract_metadata.py` - Main extraction script  
- `2_build_url.py` - URL builder
- `3_download_pdf.py` - File downloader
- `CONFIG_TEMPLATE.md` - Configuration examples

## 📖 Documentation

See `README.md` for complete documentation and examples.

## ⚙️ Configuration

Update these variables before running:

**Step 1**: `FOLDER`, `COLLECTION`, `QUERY`, `LANGUAGES`  
**Step 2**: `METADATA_FOLDER`, `OUTPUT_FILENAME`  
**Step 3**: `LANGUAGE`, `URL_FILE`

## 🎯 Pipeline Flow

Extract Metadata → Build URLs → Download Files

---
**Version**: 1.0 | **License**: Educational Use
