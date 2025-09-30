# Data Acquisition

Large-scale data collection pipeline for multilingual LLM training. Automated tools for acquiring documents, books, and educational content from multiple sources.

## Dataset Statistics

| **Language** | **# PDFs** | **# Pages** | **Word Count** |
|--------------|-----------:|------------:|---------------:|
| Hindi        | 396.12 K   | 7.53 M      | 4.15 B         |
| Marathi      | 124.22 K   | 3.02 M      | 1.26 B         |
| Malayalam    | 65.03 K    | 2.18 M      | 1.06 B         |
| Telugu       | 77.86 K    | 5.93 M      | 1.53 B         |
| Tamil        | 43.59 K    | 5.28 M      | 1.44 B         |
| Kannada      | 41.71 K    | 4.08 M      | 1.01 B         |
| Sanskrit     | 44.49 K    | 10.09 M     | 2.68 B         |
| Bengali      | 41.25 K    | 10.95 M     | 3.10 B         |
| Urdu         | 126.03 K   | 32.15 M     | 10.03 B        |
| English      | 45.10 K    | 2.57 M      | 0.89 B         |
| **Total**    | **1.00 M** | **84.00 M** | **27.15 B**    |

## Components

### Archive/ - Archive.org Pipeline
- **Purpose**: Automated PDF download from Archive.org
- **Pipeline**: 3-stage process (metadata → URLs → download)
- **Key Files**: `run_pipeline.py`, `1_extract_metadata.py`, `2_build_url.py`, `3_download_pdf.py`
- **Features**: Concurrent processing, resume capability, error handling

### Crawler/ - Web Scrapers
- **GeeksforGeeks**: Programming tutorials and articles
- **Common Crawl**: Large-scale web text extraction
- **NDLI**: Educational content from National Digital Library of India
- **AIME**: Mathematics competition problems
- **Additional**: Gadhkosh (Hindi literature), IRADAI (regional content), SEBI (financial docs)

### Wikimedia/ - Wikipedia Processing
- **Purpose**: Extract text from Wikipedia dumps
- **Languages**: Multi-language support, especially Indic languages
- **Processing**: Automated dump download and text extraction

## Quick Start

### Archive.org Pipeline
```bash
cd Archive/
python setup_check.py    # Validate setup
python run_pipeline.py   # Run complete pipeline
```

### Web Scrapers
```bash
cd Crawler/
python gfg_scraper.py                    # GeeksforGeeks
python common_crawl_url_warc_to_text.py  # Common Crawl
scrapy crawl NDLI                        # NDLI content
```

### Wikimedia Processing
```bash
cd Wikimedia/
python wikimedia_downloader.py  # Download and process dumps
```

## Key Features

- **Scalable**: Handles millions of documents with parallel processing
- **Multi-language**: Strong support for Indic languages
- **Quality Control**: Built-in validation and error handling
- **Resume Capability**: Checkpoint-based recovery for large operations
- **Automated**: End-to-end pipelines with minimal intervention

## Data Sources

- **Archive.org**: ~1M books and academic documents
- **NDLI**: 28,500 curriculum-aligned educational materials
- **Common Crawl**: Web text with language detection and quality filtering
- **Hugging Face**: Access to >1700 open datasets
- **Educational Platforms**: Specialized content from various educational sources

## Dependencies

```bash
# Core requirements
pip install requests aiohttp beautifulsoup4 scrapy
pip install fasttext pandas tqdm asyncio

# Archive pipeline
pip install -r Archive/requirements.txt

# Language detection
wget https://dl.fbaipublicfiles.com/fasttext/supervised-models/lid.176.bin
```

## Configuration

Each module contains configurable parameters:
- **Concurrency**: Number of parallel workers
- **Languages**: Target language filtering
- **Output**: Directory structure and formats
- **Rate Limiting**: Ethical scraping delays

## Integration

This module feeds into:
- **Data Curation**: Quality filtering and language processing
- **Data Organisation**: Metadata cataloging and storage management
- **Translation Pipeline**: Source content for translation tasks

---

**Status**: Production-ready | **Processed**: 1M+ documents | **Languages**: 10+ supported
