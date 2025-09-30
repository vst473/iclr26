import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import logging
import json
import scrapy
from scrapy.crawler import CrawlerProcess
from scrapy.spiders import CrawlSpider, Rule
from scrapy.linkextractors import LinkExtractor
from scrapy.exceptions import DropItem

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    filename='gadhkosh_scraper.log'
)

class GadhkoshSpider(CrawlSpider):
    name = 'gadhkosh'
    allowed_domains = ['gadhkosh.com']
    start_urls = ['http://gadyakosh.org/gk/']  # Replace with actual starting URL
    
    custom_settings = {
        'ROBOTSTXT_OBEY': True,
        'CONCURRENT_REQUESTS': 5,
        'DOWNLOAD_DELAY': 1,
        'RANDOMIZE_DOWNLOAD_DELAY': True,
        'RETRY_TIMES': 3,
        'RETRY_HTTP_CODES': [500, 502, 503, 504, 429],
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    rules = (
        # Initial rule to follow links from the main page
        Rule(LinkExtractor(restrict_xpaths='//*[@id="mw-content-text"]/div[3]/div/ul/li/a'), 
             callback='parse_item', follow=True),
    )

    def parse_item(self, response):
        try:
            # Extract the main content from the current page
            # item = {
            #     'url': response.url,
            #     'title': response.xpath('//title/text()').get(),
            #     # Add more fields as needed for the main content
            # }
            # yield item
            
            # Extract additional links using the second XPath pattern
            additional_links = response.xpath('//*[@id="mw-content-text"]/ul/li/a')
            
            for link in additional_links:
                link_url = link.xpath('@href').get()
                link_text = link.xpath('text()').get()
                
                if link_url:
                    # Make the URL absolute if it's relative
                    absolute_url = response.urljoin(link_url)
                    
                    # Create an item for each extracted link
                    link_item = {
                        'parent_url': response.url,
                        'link_url': absolute_url,
                        'link_text': link_text,
                        'type': 'secondary_link'
                    }
                    yield link_item
                    
                    # Optionally follow these links too
                    yield scrapy.Request(
                        url=absolute_url,
                        callback=self.parse_secondary_page
                    )
                    
        except Exception as e:
            logging.error(f"Error parsing {response.url}: {str(e)}")
            
    def parse_secondary_page(self, response):
        """Parse content from secondary pages that were found via the second XPath"""
        try:
            # Extract all paragraph texts using the XPath
            paragraphs = []
            for p in response.xpath('//*[@id="mw-content-text"]/p'):
                # Get text from each paragraph, including nested elements
                text = ' '.join(p.xpath('.//text()').getall()).strip()
                if text:  # Only add non-empty paragraphs
                    paragraphs.append(text)
            
            item = {
                'url': response.url,
                'title': response.xpath('//title/text()').get(),
                'content': paragraphs,  # Store all paragraph texts
                'type': 'secondary_content',
            }
            yield item
        except Exception as e:
            logging.error(f"Error parsing secondary page {response.url}: {str(e)}")

class GadhkoshPipeline:
    def __init__(self):
        self.items = []
        self.start_urls = ['http://gadyakosh.org/gk/']

    def process_item(self, item, spider):
        self.items.append(item)
        return item

    def close_spider(self, spider):
        df = pd.DataFrame(self.items)
        df.to_csv('gadhkosh_data.csv', index=False)
        logging.info("Data successfully saved to CSV")

if __name__ == "__main__":
    process = CrawlerProcess({
        'ITEM_PIPELINES': {
            'gadkosh.GadhkoshPipeline': 300,  # Use actual file name without .py
        }
    })
    process.crawl(GadhkoshSpider)
    process.start()
