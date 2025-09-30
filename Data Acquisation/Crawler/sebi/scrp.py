import scrapy
from scrapy.crawler import CrawlerProcess
import logging
import helper as h 
import os
import json
import time
import asyncio
from urllib.parse import urlparse
import pathlib

class SEBISpider(scrapy.Spider):
    name = 'sebi_spider'
    
    # Custom settings per spider (these override global settings)
    custom_settings = {
        'CONCURRENT_REQUESTS': 32,
        'CONCURRENT_REQUESTS_PER_DOMAIN': 32,
        'REACTOR_THREADPOOL_MAXSIZE': 20,
        'DOWNLOAD_TIMEOUT': 180,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 408, 429],
        'DOWNLOAD_DELAY': 0.25,  # Avoid overwhelming the server with too many simultaneous requests
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'COOKIES_ENABLED': False,  # Disable cookies for better performance
        'HTTPCACHE_ENABLED': True,  # Enable HTTP cache
        'HTTPCACHE_EXPIRATION_SECS': 86400,  # 24 hours
        'HTTPCACHE_DIR': 'httpcache',
        'HTTPCACHE_IGNORE_HTTP_CODES': [503, 504, 500, 400, 401, 403, 404, 408]
    }
    
    def __init__(self, base_url=None, url_file="sebi_menu_url.json", *args, **kwargs):
        # Setup logging first
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        
        super(SEBISpider, self).__init__(*args, **kwargs)
        
        # Load or create menu URLs
        if os.path.exists(url_file):
            links = h.load_menu_from_txt(url_file)
        else:
            self.base_url = base_url or "https://www.sebi.gov.in/js/menu.js"
            links = h.find_home_action_links(self.base_url)
            
            base_url = "https://www.sebi.gov.in"
            links = [base_url + link for link in links if 'HomeAction' in link]
            h.save_menu_to_txt(links, url_file)
            self.logger.info(f"Menu data saved to {url_file}")
        
        # Track files being downloaded to avoid duplicates
        self.downloading_files = set()
        
        # Create required directories
        pathlib.Path('downloads').mkdir(exist_ok=True)
        pathlib.Path('metadata').mkdir(exist_ok=True)
        
        # Use a subset of links for testing
        # self.start_urls = links[:5]  # Uncomment for testing with fewer links
        self.start_urls = links
        self.allowed_domains = ["sebi.gov.in"]

    def parse(self, response):
        """Parse main listing page"""
        xpath_selector = '//tbody//tr'
        
        # Extract directory path from breadcrumb
        t = response.xpath('//*[@class="cfc-breadcrumb"]')
        directory = [x.replace('\n', '').strip() for x in t.xpath('string()').get().strip().split('»') if x.replace('\n', '') is not None]
        directory_path = os.path.join(*directory) if directory else "default_directory"
        
        # Create directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        self.logger.info(f"Directory path: {directory_path}")
        
        # Process items in the current page
        items_to_process = []
        for item in response.xpath(xpath_selector):
            url = item.xpath('.//a/@href').get()
            if url:
                items_to_process.append({
                    'url': url,
                    'date': item.xpath('td')[0].xpath('string()').get().strip(),
                    'title': item.xpath('td')[1].xpath('string()').get().strip(),
                    'directory_path': directory_path
                })
        
        # Process items in batches to improve concurrency
        for i, item in enumerate(items_to_process):
            # Request document page
            yield scrapy.Request(
                url=item['url'],
                callback=self.parse_document,
                meta={
                    'date': item['date'],
                    'title': item['title'],
                    'directory_path': item['directory_path']
                },
                priority=2  # Higher priority for document pages
            )
        
        # Now handle pagination
        url_params = {}
        if '?' in response.url:
            for param in response.url.split('?')[1].split('&'):
                if '=' in param:
                    key, value = param.split('=', 1)
                    url_params[key] = value
        
        # Extract pagination details
        pages_details = response.xpath('//*[@id="ajax_cat"]/div[2]/div[1]/p/text()').get()
        if not pages_details:
            self.logger.info("No pagination details found.")
            return
        
        # Parse pagination info
        try:
            pages_details = pages_details.split(' ')
            if len(pages_details) >= 5:
                total_pages = int(pages_details[4]) // int(pages_details[2]) + 1
                next_value = response.xpath('//input[@name="nextValue"]/@value').get()
                
                if next_value and total_pages > int(next_value):
                    next_value = int(next_value)
                    
                    # Queue multiple pagination requests at once (up to 5 pages)
                    max_pages = min(total_pages, next_value + 5)
                    
                    self.logger.info(f"Queueing pages {next_value+1} to {max_pages} of {total_pages}")
                    
                    # Create a batch of requests for better concurrency
                    for page_num in range(next_value + 1, max_pages + 1):
                        ajax_url = "https://www.sebi.gov.in/sebiweb/ajax/home/getnewslistinfo.jsp"
                        payload = f'nextValue={page_num-1}&next=n&sid={url_params.get("sid", "1")}&ssid={url_params.get("ssid", "1")}&smid={url_params.get("smid", "0")}'
                        
                        yield scrapy.Request(
                            url=ajax_url,
                            method='POST',
                            headers={
                                'Content-Type': 'application/x-www-form-urlencoded',
                                'Origin': 'https://www.sebi.gov.in',
                                'Referer': response.url
                            },
                            body=payload,
                            callback=self.parse_ajax_response,
                            meta={'directory_path': directory_path, 'url_params': url_params},
                            priority=1,  # Lower priority than document pages
                            dont_filter=True  # Don't filter duplicate requests for AJAX
                        )
        except (IndexError, ValueError) as e:
            self.logger.error(f"Error parsing pagination: {e}")

    def parse_ajax_response(self, response):
        """Parse AJAX response containing next page of results"""
        xpath_selector = '//tr[@role="row"]'
        directory_path = response.meta['directory_path']
        
        # Process items in the AJAX response
        for item in response.xpath(xpath_selector):
            url = item.xpath('.//a/@href').get()
            if url:
                date_text = item.xpath('td[1]/text()').get()
                title_text = item.xpath('td[2]/text()').get()
                
                date = date_text.strip() if date_text else 'not_f'
                title = title_text.strip() if title_text else None
                
                if title is None:
                    title = url.split('/')[-1].split('.')[0]
                
                if url and date:
                    yield scrapy.Request(
                        url=url,
                        callback=self.parse_document,
                        meta={
                            'date': date,
                            'title': title,
                            'directory_path': directory_path
                        },
                        priority=2  # Higher priority for document pages
                    )
                else:
                    self.logger.warning(f"Skipping item with missing data: URL={url}, DATE={date}, TITLE={title}")
        
        # Continue pagination if more pages exist
        url_params = response.meta['url_params']
        total_pages = response.xpath('//input[@name="totalpage"]/@value').get()
        next_value = response.xpath('//input[@name="nextValue"]/@value').get()
        
        if total_pages and next_value:
            try:
                total_pages = int(total_pages)
                next_value = int(next_value)
                
                # Only continue if there are more pages
                if next_value < total_pages:
                    # Queue the next batch of pages (up to 5 more)
                    max_pages = min(total_pages, next_value + 5)
                    
                    self.logger.info(f"Next batch of pages: {next_value+1} to {max_pages} of {total_pages}")
                    
                    for page_num in range(next_value + 1, max_pages + 1):
                        ajax_url = "https://www.sebi.gov.in/sebiweb/ajax/home/getnewslistinfo.jsp"
                        payload = f'nextValue={page_num-1}&next=n&sid={url_params.get("sid", "1")}&ssid={url_params.get("ssid", "1")}&smid={url_params.get("smid", "0")}'
                        
                        yield scrapy.Request(
                            url=ajax_url,
                            method='POST',
                            headers={
                                'Content-Type': 'application/x-www-form-urlencoded',
                                'Origin': 'https://www.sebi.gov.in',
                                'Referer': response.url
                            },
                            body=payload,
                            callback=self.parse_ajax_response,
                            meta={'directory_path': directory_path, 'url_params': url_params},
                            priority=1,
                            dont_filter=True
                        )
            except (ValueError, TypeError) as e:
                self.logger.error(f"Error parsing pagination values: {e}")
                
    def parse_document(self, response):
        """Parse document page and extract file or content"""
        # Clean up metadata
        date = response.meta.get('date', '').replace('\n', '').strip()
        title = response.meta.get('title', '').replace('\n', '').strip()
        directory_path = response.meta.get('directory_path', 'downloads')
        
        # Make sure directory exists
        os.makedirs(directory_path, exist_ok=True)
        
        # Create a unique ID for this file to avoid duplicates
        file_id = f"{response.url}_{date}_{title}"
        
        # Skip if already downloading this file
        if file_id in self.downloading_files:
            self.logger.info(f"Already downloading: {response.url}")
            return
        
        # Mark as being downloaded
        self.downloading_files.add(file_id)
        
        # Check if URL is a direct file download
        url = response.url.lower()
        known_extensions = ['.pdf', '.xlsx', '.xls', '.docx', '.doc', '.pptx', '.ppt', '.csv', '.zip', '.rar', '.txt']
        
        for extension in known_extensions:
            if url.endswith(extension):
                self.logger.info(f"Direct file download: {url}")
                
                # Create filename based on date and title
                filename = self._create_valid_filename(f"{date}_{title}{extension}")
                full_path = os.path.join(directory_path, filename)
                
                # Use direct streaming to file instead of loading into memory
                with open(full_path, 'wb') as f:
                    f.write(response.body)
                
                # Save metadata
                self._save_metadata(date, title, url, full_path, directory_path, extension[1:])
                
                # Remove from downloading set
                self.downloading_files.discard(file_id)
                
                self.logger.info(f"Saved file: {full_path}")
                return
        
        # Handle embedded PDF
        pdf_iframe = response.xpath('//*[@id="member-wrapper"]/section[2]/div[1]/section/div[2]/div/iframe/@src').get()
        
        if pdf_iframe:
            pdf_web_url = response.urljoin(pdf_iframe)
            
            # Extract the actual PDF URL
            pdf_url = pdf_web_url.replace("web/?file=/", '')
            
            if 'file=' in pdf_url:
                pdf_url = pdf_url.split('file=')[1]
            
            self.logger.info(f"Found PDF URL: {pdf_url}")
            
            # Download the PDF file
            yield scrapy.Request(
                url=pdf_url,
                callback=self.save_pdf,
                meta={
                    'date': date,
                    'title': title,
                    'directory_path': directory_path,
                    'pdf_url': pdf_url,
                    'file_id': file_id
                },
                priority=3  # Highest priority for file downloads
            )
            return
        
        # Handle text content if no file is found
        text_content = response.xpath('//*[@id="member-wrapper"]/section[2]/div[1]').xpath('string()').get()
        if text_content:
            self.logger.info(f"Found text content: {len(text_content)} characters")
            
            # Create filename based on date and title
            filename = self._create_valid_filename(f"{date}_{title}.txt")
            full_path = os.path.join(directory_path, filename)
            
            # Save text content
            with open(full_path, 'w', encoding='utf-8') as f:
                f.write(text_content)
            
            # Save metadata
            self._save_metadata(date, title, response.url, full_path, directory_path, 'txt')
            
            # Remove from downloading set
            self.downloading_files.discard(file_id)
            
            self.logger.info(f"Saved text: {full_path}")
        else:
            self.logger.warning(f"No content found at {response.url}")
            # Remove from downloading set
            self.downloading_files.discard(file_id)
    
    def save_pdf(self, response):
        """Save PDF file"""
        date = response.meta['date']
        title = response.meta['title']
        directory_path = response.meta['directory_path']
        file_id = response.meta['file_id']
        
        # Create filename based on date and title
        filename = self._create_valid_filename(f"{date}_{title}.pdf")
        full_path = os.path.join(directory_path, filename)
        
        # Save the PDF file
        with open(full_path, 'wb') as f:
            f.write(response.body)
        
        # Save metadata
        self._save_metadata(date, title, response.meta['pdf_url'], full_path, directory_path, 'pdf')
        
        # Remove from downloading set
        self.downloading_files.discard(file_id)
        
        self.logger.info(f"Saved PDF: {full_path}")
    
    def _create_valid_filename(self, filename):
        """Create a valid filename without invalid characters"""
        # Replace invalid characters
        invalid_chars = '<>:"/\\|?*'
        for char in invalid_chars:
            filename = filename.replace(char, '_')
        
        # Limit length to avoid path too long issues
        if len(filename) > 150:
            name, ext = os.path.splitext(filename)
            filename = name[:145] + ext
            
        return filename
    
    def _save_metadata(self, date, title, url, file_path, directory_path, file_type):
        """Save metadata to central JSON file"""
        metadata = {
            'date': date,
            'title': title,
            'url': url,
            'file_path': file_path,
            'directory_path': directory_path,
            'type': file_type,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
        }
        
        # Create metadata directory
        metadata_dir = 'metadata'
        os.makedirs(metadata_dir, exist_ok=True)
        metadata_file = os.path.join(metadata_dir, 'sebi_documents_metadata.json')
        
        # Use a lock file to prevent concurrent writes
        lock_file = metadata_file + '.lock'
        
        # Try to acquire lock
        try:
            # Create lock file if it doesn't exist
            if not os.path.exists(lock_file):
                with open(lock_file, 'w') as f:
                    f.write('1')
                
                # Load existing metadata
                all_metadata = []
                if os.path.exists(metadata_file):
                    try:
                        with open(metadata_file, 'r', encoding='utf-8') as f:
                            all_metadata = json.load(f)
                    except json.JSONDecodeError:
                        self.logger.error(f"Error decoding metadata file, creating new one")
                
                # Add new metadata
                all_metadata.append(metadata)
                
                # Save updated metadata
                with open(metadata_file, 'w', encoding='utf-8') as f:
                    json.dump(all_metadata, f, ensure_ascii=False, indent=2)
                
                # Release lock
                os.remove(lock_file)
            else:
                # Another process is writing, wait and try again later
                self.logger.info(f"Metadata lock file exists, will retry later for {title}")
        except Exception as e:
            self.logger.error(f"Error saving metadata: {e}")
            # Make sure we clean up the lock file
            if os.path.exists(lock_file):
                try:
                    os.remove(lock_file)
                except:
                    pass


def main():
    """Main entry point for the crawler"""
    # Configure crawler process
    process = CrawlerProcess({
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'LOG_LEVEL': 'INFO',
        'ROBOTSTXT_OBEY': False,  # We'll respect robots.txt but handle it differently
        'DOWNLOAD_DELAY': 0.25,  # Small delay to avoid overwhelming the server
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'CONCURRENT_REQUESTS': 32,  # Reasonable concurrency level
        'CONCURRENT_REQUESTS_PER_DOMAIN': 32,
        'REACTOR_THREADPOOL_MAXSIZE': 20,  # Increase reactor thread pool size
        'COOKIES_ENABLED': False,  # Disable cookies for better performance
        'DEFAULT_REQUEST_HEADERS': {
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
            'Referer': 'https://www.sebi.gov.in/'
        }
    })

    # Start crawling
    process.crawl(SEBISpider)
    process.start()

if __name__ == "__main__":
    main()