# Data Acquisition

Large-scale multilingual data collection pipeline for LLM training, emphasizing authentic curriculum-aligned content from trusted academic and cultural sources.

## 📊 Dataset Overview

**Total Scale**: 1.03M+ documents across multiple sources and languages

### Key Sources
- **Archive.org**: ~1M books (multilingual literature, academic texts, cultural documents)
- **NDLI**: 28,500 curriculum-aligned educational documents  
- **Common Crawl**: Web-scale multilingual text extraction
- **Specialized Scrapers**: Educational platforms, technical documentation

## � Language Statistics (Archive.org)

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

## 🗂️ Key Data Sources

- **Web Crawl + Open Datasets**
  - Common Crawl, multilingual websites, forums, academic repositories
  - >1700 datasets hosted on [Hugging Face](https://huggingface.co)
- **Book Collections**
  - ~1M books from [Archive.org](https://archive.org)
  - 28,500 curriculum-aligned documents from the [National Digital Library of India (NDLI)](https://ndl.iitkgp.ac.in)

## 🔬 Acquisition Methodology

1. **Source Integration**
   - Inspired by *Pile*, *RedPajama*, and *C4* methodologies
   - Unified crawl + dataset ingestion pipeline with provenance tracking

2. **Schema-First Cataloging**
   - Every acquired item is labeled along **orthogonal axes**:
     - Language classification
     - Grade level (for school curricula)
     - Content provider / institution
     - Subject domain (for higher education)
   - Multi-labeled items (e.g., bilingual, multi-grade) are preserved with **multiple metadata tags**

3. **Deduplication & Integrity**
   - Deduplication performed at the **item-ID level**, preventing loss of cross-listed data
   - Metadata normalization ensures **cross-source compatibility**

4. **Distribution Reporting**
   - Tabular distributions provided at multiple levels:
     - NDLI **school-level content** (languages, classes, providers)
     - NDLI **higher education** (providers, levels, subjects)
   - Enables **coverage quantification** and **balanced sampling**

5. **Scalable Processing**
   - Designed for **millions of documents**, including OCR + post-correction workflows for Indic scripts
   - Robust **checkpointing + shard-based retries** for week-scale ingestion tasks

## � Reproducibility

Our acquisition pipeline is designed for full reproducibility with:
- **Detailed Configuration**: All parameters documented and version-controlled
- **Provenance Tracking**: Source attribution and processing history
- **Checkpointing**: Resume capability for interrupted large-scale operations
- **Schema Validation**: Consistent metadata structure across all sources

### Steps to Reproduce
Detailed reproduction instructions are provided in each module's documentation and scripts.

## �📁 Directory Structure

### Acquisition Folder Layout
```
Data Acquisation/
 ├── Archive/                    # Archive.org pipeline
 │    ├── run_pipeline.py       # Complete automated pipeline
 │    ├── 1_extract_metadata.py # API-based metadata extraction
 │    ├── 2_build_url.py        # Download URL generation
 │    ├── 3_download_pdf.py     # Concurrent PDF downloading
 │    ├── setup_check.py        # Configuration validation
 │    ├── get_metadata.py       # API helper functions
 │    ├── CONFIG_TEMPLATE.md    # Configuration examples
 │    └── README.md             # Detailed pipeline documentation
 ├── Crawler/                   # Specialized web scrapers
 │    ├── gfg_scraper.py        # GeeksforGeeks content
 │    ├── common_crawl_url_warc_to_text.py # Common Crawl processing
 │    ├── ndli_highschool.py    # NDLI educational content
 │    ├── aimim.py              # AIME mathematics problems
 │    ├── gadhkosh.py           # Hindi literature
 │    ├── iradai.py             # Regional research content
 │    ├── sebi/                 # Financial regulations
 │    └── README.md             # Scrapers documentation
 ├── Wikimedia/                 # Wikipedia ecosystem
 │    ├── wikimedia_downloader.py # Automated dump downloading
 │    ├── indic_dump.py         # Indic language processing
 │    ├── scrape_dumplist.py    # Dump inventory
 │    └── Readme.md             # Wikimedia documentation
 └── README.md                  # This comprehensive guide
```

### 🗃️ [Archive/](Archive/) - Archive.org Document Pipeline
**Complete automated system for Archive.org document acquisition**

**Key Features:**
- 3-stage pipeline: Metadata extraction → URL building → PDF download
- Automated batch processing with `run_pipeline.py`
- Resume capability for interrupted downloads
- Concurrent processing with configurable workers
- Comprehensive error handling and logging

**Core Files:**
- `run_pipeline.py` - Complete automated pipeline runner
- `1_extract_metadata.py` - API-based metadata extraction
- `2_build_url.py` - Download URL generation
- `3_download_pdf.py` - Concurrent PDF downloading
- `setup_check.py` - Configuration validation
- `get_metadata.py` - API helper functions library

**Capabilities:**
- Process multiple Archive.org collections simultaneously
- Language-specific filtering (particularly strong for Indic languages)
- Subject-based categorization and filtering
- Pagination handling for large collections
- Duplicate detection and removal

### 🕷️ [Crawler/](Crawler/) - Specialized Web Scrapers
**Advanced web scraping tools for educational and technical content**

**Scrapers Included:**
- **GeeksforGeeks** (`gfg_scraper.py`) - Programming tutorials and computer science content
- **Common Crawl** (`common_crawl_url_warc_to_text.py`) - Large-scale web text extraction with language detection
- **NDLI** (`ndli_highschool.py`) - National Digital Library of India educational content
- **AIME** (`aimim.py`) - Mathematical competition problems and solutions
- **Gadhkosh** (`gadhkosh.py`) - Hindi literature and cultural content scraper
- **IRADAI** (`iradai.py`) - Regional language content and research materials
- **SEBI** (`sebi/scrp.py`) - Financial regulations and compliance documents

**Advanced Features:**
- Async processing for high throughput
- FastText-based language detection
- Content cleaning and normalization
- Scrapy framework integration for complex sites
- Rate limiting and ethical scraping practices

### 📚 [Wikimedia/](Wikimedia/) - Wikipedia Ecosystem Processor
**Comprehensive Wikimedia project data extraction and processing**

**Key Components:**
- `wikimedia_downloader.py` - Automated dump downloading
- `indic_dump.py` - Indic language Wikipedia processing
- `scrape_dumplist.py` - Dump inventory and metadata collection
- `dumps.txt` - Curated list of target dumps

**Capabilities:**
- Multi-project support (Wikipedia, Wikibooks, Wiktionary, Wikisource)
- Language-specific processing pipelines
- Automated dump discovery and downloading
- Text extraction with markup cleaning
- Cross-language link processing

## 🚀 Quick Start Guide

### Archive.org Pipeline
```bash
# Complete automated pipeline
cd "Data Acquisation/Archive"
python setup_check.py          # Validate configuration
python run_pipeline.py         # Run complete pipeline

# Manual step-by-step
python 1_extract_metadata.py   # Extract metadata
python 2_build_url.py          # Build download URLs  
python 3_download_pdf.py       # Download PDFs
```

### Web Scrapers
```bash
cd "Data Acquisation/Crawler"
python gfg_scraper.py                    # GeeksforGeeks content
python common_crawl_url_warc_to_text.py  # Common Crawl processing
scrapy crawl NDLI                        # NDLI educational content
scrapy crawl aime                        # AIME mathematics problems
scrapy crawl gadhkosh                    # Hindi literature content
scrapy crawl iradai                      # Regional research content
scrapy crawl sebi_spider                 # SEBI financial documents
```

### Wikimedia Processing
```bash
cd "Data Acquisation/Wikimedia"
python scrape_dumplist.py      # Get available dumps
python wikimedia_downloader.py # Download and process dumps
python indic_dump.py           # Process Indic language dumps
```

## � Configuration & Setup

### Archive.org Configuration
Each script contains configurable parameters:
- **Collections**: Target Archive.org collections
- **Languages**: Language filtering (strong Indic support)
- **Subjects**: Subject-based filtering
- **Concurrency**: Number of parallel workers
- **Output**: Directory structure and naming

### Crawler Configuration
- **Rate Limiting**: Respectful crawling speeds
- **User Agents**: Rotating user agent strings  
- **Language Detection**: FastText model integration
- **Output Formats**: JSONL, CSV, plain text options

### Wikimedia Configuration
- **Dump Selection**: Automated latest dump discovery
- **Language Filtering**: Multi-language processing
- **Content Filtering**: Article quality and length filters
- **Processing Options**: Memory-efficient streaming processing

## 🎯 Use Cases for LLM Training

### Large-Scale Document Collection
- **Archive.org**: Historical texts, books, academic papers
- **Educational Content**: NDLI high school materials, programming tutorials
- **Mathematical Content**: Competition problems for reasoning datasets
- **Multilingual Corpora**: Wikipedia articles across languages

### Quality Datasets
- **Curated Collections**: Subject-specific document sets
- **Educational Materials**: Structured learning content
- **Technical Documentation**: Programming and computer science resources
- **Cultural Content**: Indic language literature and texts

### Data Pipeline Integration
- **Metadata Extraction**: Rich document metadata for filtering and organization
- **Language Detection**: Automatic language classification and routing
- **Quality Filtering**: Built-in content quality assessment
- **Format Standardization**: Consistent output formats for downstream processing

## 📊 Performance Features

### Scalability
- **Concurrent Processing**: Multi-threaded/multi-process architecture
- **Memory Management**: Streaming processing for large files
- **Resume Capability**: Checkpoint-based recovery from interruptions
- **Batch Processing**: Efficient handling of large document collections

### Quality Assurance
- **Input Validation**: Configuration and parameter validation
- **Error Handling**: Comprehensive error recovery and logging
- **Duplicate Detection**: Cross-source duplicate identification
- **Content Validation**: Format and encoding verification

### Monitoring & Logging
- **Progress Tracking**: Real-time progress indicators with ETA
- **Detailed Logging**: Comprehensive operation logs
- **Performance Metrics**: Processing speed and success rate tracking
- **Error Reporting**: Structured error reporting and analysis

## � Integration with Other Modules

### Data Curation Pipeline
- Feeds processed documents to the curation filters
- Provides metadata for quality assessment
- Supports language-specific processing workflows

### Data Organisation
- Integrates with JuiceFS for distributed storage
- Provides metadata for DataHub cataloging
- Supports hierarchical data organization

### Translation Pipeline
- Supplies source documents for translation
- Provides multilingual content for benchmarking
- Supports cross-language document alignment

## 📚 Documentation

**Detailed documentation available in each subfolder:**
- `Archive/README.md` - Complete Archive.org pipeline guide (303 lines)
- `Archive/PROJECT.md` - Quick project overview and status
- `Archive/CONFIG_TEMPLATE.md` - Configuration examples and templates
- `Crawler/README.md` - Web scrapers overview and usage
- `Wikimedia/Readme.md` - Wikimedia processing documentation

## 🛠️ Dependencies

### Core Requirements
```bash
pip install requests aiohttp beautifulsoup4 lxml
pip install scrapy fasttext pandas tqdm
pip install asyncio concurrent.futures
```

### Archive.org Pipeline
- `requests` - HTTP client for API calls
- `aiohttp` - Async HTTP for concurrent downloads
- `tqdm` - Progress bars and monitoring

### Web Scrapers
- `scrapy` - Web scraping framework
- `beautifulsoup4` - HTML parsing
- `fasttext` - Language detection
- `selenium` - Dynamic content scraping (when needed)

### Wikimedia Processing
- `mwparserfromhell` - MediaWiki markup parsing
- `xml.etree.ElementTree` - XML dump processing
- `bz2`, `gzip` - Compression handling

## 🚀 Recent Updates

### Latest Additions (September 30, 2025)
- **Complete Archive.org Pipeline**: Full 3-stage automated pipeline
- **Pipeline Runner**: `run_pipeline.py` for end-to-end automation
- **Setup Validation**: `setup_check.py` for configuration verification
- **Enhanced Documentation**: Comprehensive README files with examples
- **Error Recovery**: Resume capability for interrupted operations
- **Performance Optimization**: Concurrent processing improvements

### Enhanced Features
- **Multi-language Support**: Improved Indic language handling
- **Quality Filters**: Better content quality assessment
- **Metadata Enrichment**: Enhanced document metadata extraction
- **Pipeline Integration**: Better integration with downstream modules

## 🔮 Future Enhancements

### Planned Features
- **Real-time Processing**: Live content monitoring and acquisition
- **ML-based Quality Assessment**: Automated content quality scoring
- **Advanced Deduplication**: Semantic similarity-based duplicate detection
- **API Integration**: RESTful API for external integration
- **Dashboard**: Web-based monitoring and control interface

### Scalability Improvements
- **Distributed Processing**: Multi-node processing support
- **Cloud Integration**: AWS/GCP integration for large-scale operations
- **Streaming Architecture**: Real-time data processing pipelines
- **Automated Scheduling**: Cron-based automated data acquisition

---

## 📋 Summary

This comprehensive data acquisition infrastructure has successfully collected and processed:
- **1.00M documents** across 10 languages
- **84.00M pages** of multilingual content  
- **27.15B words** for LLM training
- **Curriculum-aligned** educational content from NDLI
- **High-quality** academic and cultural texts

The pipeline ensures **reproducibility**, **scalability**, and **quality** through automated processes, comprehensive metadata management, and robust error handling.

---

**Project**: ICLR26 Data Acquisition Infrastructure  
**Purpose**: Large-scale, multi-source data collection for multilingual LLM training and research  
**Status**: Production-ready with proven scalability (1M+ documents processed)  
**Last Updated**: September 30, 2025

*Note: This README combines research methodology documentation with technical implementation details for comprehensive coverage of the data acquisition pipeline.*
