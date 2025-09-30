# Web Crawlers & Scrapers

Collection of specialized web scrapers for various educational and content platforms.

## 📁 Contents

- `gfg_scraper.py` - GeeksforGeeks articles and tutorials scraper
- `common_crawl_url_warc_to_text.py` - Common Crawl WARC file text extractor
- `ndli_highschool.py` - National Digital Library of India (NDLI) scraper for high school content
- `aimim.py` - AIME (American Invitational Mathematics Examination) problems scraper

## 🎯 Scrapers Overview

### GeeksforGeeks Scraper
- **Target**: Programming tutorials and articles
- **Format**: Clean text extraction with metadata
- **Language**: Python with BeautifulSoup

### Common Crawl Extractor  
- **Target**: Web text from Common Crawl WARC files
- **Features**: Language detection, text cleaning, async processing
- **Languages**: Multi-language support with FastText

### NDLI Scraper
- **Target**: Educational content from National Digital Library
- **Focus**: High school level materials
- **Framework**: Scrapy spider

### AIME Scraper
- **Target**: Mathematics competition problems
- **Source**: Art of Problem Solving wiki
- **Framework**: Scrapy spider

## 🚀 Usage

Each scraper can be run independently:

```bash
python gfg_scraper.py                    # GeeksforGeeks
python common_crawl_url_warc_to_text.py  # Common Crawl
scrapy crawl NDLI                        # NDLI (requires Scrapy)
scrapy crawl aime                        # AIME (requires Scrapy)
```

---
**Purpose**: Specialized web scrapers for educational and technical content
