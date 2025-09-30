import scrapy
from scrapy.crawler import CrawlerProcess
import logging
# import helper as h 
import os
import json
import time
import asyncio
from urllib.parse import urlparse
import pathlib

class IRADAISpider(scrapy.Spider):
    name = 'iradai'
    
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
        
        super(IRADAISpider, self).__init__(*args, **kwargs)
        
        # # Load or create menu URLs
        # if os.path.exists(url_file):
        #     links = h.load_menu_from_txt(url_file)
        # else:
        #     self.base_url = base_url or "https://www.sebi.gov.in/js/menu.js"
        #     links = h.find_home_action_links(self.base_url)
            
        #     base_url = "https://www.sebi.gov.in"
        #     links = [base_url + link for link in links if 'HomeAction' in link]
        #     h.save_menu_to_txt(links, url_file)
        #     self.logger.info(f"Menu data saved to {url_file}")
        
        # Track files being downloaded to avoid duplicates
        self.downloading_files = set()
        
        # Create required directories
        pathlib.Path('downloads').mkdir(exist_ok=True)
        pathlib.Path('metadata').mkdir(exist_ok=True)
        
        # Use a subset of links for testing
        # self.start_urls = links[:5]  # Uncomment for testing with fewer links
        self.start_urls = ["https://irdai.gov.in/warnings-and-penalties", 
                           "https://irdai.gov.in/life"]
        self.allowed_domains = ["irdai.gov.in"]

    def parse(self, response):
        """Parse main listing page"""
        xpath_selector = '/html/body/div[1]/div/div/section/div/div/div[1]/div/div/section/div/div/div/div[1]/div/div/div//a'
        
        # Extract directory path from breadcrumb
        t = response.xpath('//*[@id="_com_liferay_site_navigation_breadcrumb_web_portlet_SiteNavigationBreadcrumbPortlet_INSTANCE_Xu3XFCYe6VXF_breadcrumbs-defaultScreen"]/ol')
        
        directory = [x.xpath('string()').get().replace('\t', '').replace('\n', '') for x in t.xpath('./li')]
       
        directory_path = os.path.join(*directory) if directory else "default_directory"
        
        # Create directory if it doesn't exist
        os.makedirs(directory_path, exist_ok=True)
        self.logger.info(f"Directory path: {directory_path}")
        

        for item in response.xpath(xpath_selector):
            url = item.xpath('@href').get()
            
            logging.info(f"Processing URL: {url}")
            if url:
                yield scrapy.Request(
                    url=url,
                    callback=self.process_page,
                    meta={'directory_path': directory_path},
                    priority=1  # Higher priority for document pages
                )
    
    def process_page(self, response):
        xpath = '//*[@id="_com_irdai_document_media_IRDAIDocumentMediaPortlet_fileEntriesSearchContainerSearchContainer"]/table/tbody/tr'
        directory_path = response.meta['directory_path']
        
        next_page_xpath = '//*[@id="_com_irdai_document_media_IRDAIDocumentMediaPortlet_fileEntriesSearchContainerPageIteratorBottom"]/div/ul/li[3]/a/@href'
        
        # Check if there is a next page
        next_page_url = response.xpath(next_page_xpath).get()
        if next_page_url and next_page_url.startswith('http'):
            # Extract the next page number from the URL
            next_page_number = next_page_url.split('cur=')[1]
            self.logger.info(f"Next page number: {next_page_number}")
        
            yield scrapy.Request(
                url=next_page_url,
                callback=self.process_page,
                meta={'directory_path': directory_path},
                priority=1,
                dont_filter=True
            )
        
        for item in response.xpath(xpath):
            url = item.xpath('td[7]/div/div[3]/a/@href').get()
            if url:
                date_text = item.xpath('./td[4]/text()').get()
                title_text = item.xpath('./td[5]/a/u/text()').get()
                short_desc = item.xpath('td[3]/text()').get()
                
                date_text = date_text.replace('\t', '').replace('\n', '').replace('\r','') if date_text else None
                title_text = title_text.replace('\t', '').replace('\n', '').replace('\r','') if title_text else None
                short_desc = short_desc.replace('\t', '').replace('\n', '').replace('\r','') if short_desc else None
                
                
                date = date_text.strip() if date_text else 'not_f'
                title = title_text.strip() if title_text else None
                
                if title is None:
                    title = url.split('/')[-1].split('.')[0]
                
                # Generate a unique file ID
                file_id = f"{date}_{title}_{url.split('/')[-1]}"
                
                if url and date:
                    # Add file_id to tracking set
                    self.downloading_files.add(file_id)
                    
                    yield scrapy.Request(
                        url=url,
                        callback=self.save_pdf,
                        meta={
                            'date': date,
                            'title': title,
                            'short_desc': short_desc,
                            'directory_path': directory_path,
                            'file_id': file_id
                        },
                        priority=2  # Higher priority for document pages
                    )
    
       
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
        self._save_metadata(date, title, response.url, full_path, directory_path, 'pdf')
        
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
    process.crawl(IRADAISpider)
    process.start()

if __name__ == "__main__":
    main()