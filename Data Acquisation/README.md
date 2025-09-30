# Data Acquisition Tools

Collection of tools and scripts for acquiring data from various sources for research and analysis.

## 📁 Directory Structure

### 🗃️ [Archive/](Archive/)
Archive.org document scraping pipeline
- Extract metadata from Archive.org collections
- Build download URLs and download PDF files
- Complete automated pipeline with validation

### 🕷️ [Crawler/](Crawler/) 
Specialized web scrapers
- GeeksforGeeks programming tutorials
- Common Crawl WARC text extraction
- NDLI educational content scraper
- AIME mathematics problems scraper

### 📚 [Wikimedia/](Wikimedia/)
Wikimedia project dump processor
- Wikipedia, Wikibooks, Wiktionary dumps
- Content extraction and text processing
- Multi-language support

## 🚀 Quick Overview

| Folder | Purpose | Key Files | Usage |
|--------|---------|-----------|-------|
| **Archive** | Archive.org PDFs | `run_pipeline.py` | `python run_pipeline.py` |
| **Crawler** | Web scraping | `*.py` scrapers | Individual scrapers |
| **Wikimedia** | Wiki dumps | `wikimedia_downloader.py` | Dump processing |

## 📖 Documentation

Each folder contains its own README with detailed usage instructions:
- `Archive/README.md` - Complete Archive.org pipeline documentation
- `Crawler/README_MINIMAL.md` - Web scrapers overview
- `Wikimedia/Readme.md` - Wikimedia tools documentation

---
**Project**: ICLR26 Data Acquisition  
**Purpose**: Multi-source data collection and processing tools
