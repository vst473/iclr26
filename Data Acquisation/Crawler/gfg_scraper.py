import requests
from bs4 import BeautifulSoup
import json
import re
import logging
from urllib.parse import urljoin

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SimpleGeeksForGeeksScraper:
    def __init__(self, base_url="https://www.geeksforgeeks.org"):
        self.base_url = base_url
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
        })
    
    def clean_text(self, text):
        """Clean text by removing extra whitespace and formatting"""
        if not text:
            return ""
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        # Remove unnecessary characters
        text = re.sub(r'[^\w\s\.\,\;\:\!\?\(\)\[\]\{\}\-\+\=\<\>\#\@\$\%\^\&\*\'\"\/\\]', '', text)
        return text.strip()
    
    def clean_code(self, code_text):
        """Clean code by preserving structure but removing extra whitespace"""
        if not code_text:
            return ""
        
        lines = code_text.strip().split('\n')
        cleaned_lines = []
        
        for line in lines:
            cleaned_line = line.strip()
            if cleaned_line:  # Skip empty lines
                cleaned_lines.append(cleaned_line)
        
        return '\n'.join(cleaned_lines)
    
    def extract_examples(self, soup):
        """Extract input/output examples from the article"""
        examples = []
        blockquotes = soup.find_all('blockquote')
        
        for blockquote in blockquotes:
            text = blockquote.get_text(strip=True)
            if 'Input:' in text and 'Output:' in text:
                # Parse the example
                example = {
                    'input': '',
                    'output': '',
                    'explanation': ''
                }
                
                # Extract input
                input_match = re.search(r'Input:\s*([^Output]*)', text)
                if input_match:
                    example['input'] = self.clean_text(input_match.group(1))
                
                # Extract output
                output_match = re.search(r'Output:\s*([^Explanation]*)', text)
                if output_match:
                    example['output'] = self.clean_text(output_match.group(1))
                
                # Extract explanation if present
                explanation_match = re.search(r'Explanation:\s*(.*)', text)
                if explanation_match:
                    example['explanation'] = self.clean_text(explanation_match.group(1))
                
                examples.append(example)
        
        return examples
    
    def extract_approaches(self, soup):
        """Extract different approaches/methods from the article"""
        approaches = []
        
        # Find all headings that contain approach information
        headings = soup.find_all(['h2', 'h3', 'h4'])
        
        for heading in headings:
            heading_text = heading.get_text(strip=True)
            if any(keyword in heading_text.lower() for keyword in ['approach', 'method', 'algorithm', 'solution']):
                approach = {
                    'title': self.clean_text(heading_text),
                    'description': '',
                    'time_complexity': '',
                    'space_complexity': ''
                }
                
                # Get content after the heading
                content_parts = []
                current = heading.next_sibling
                
                while current and current.name not in ['h2', 'h3', 'h4']:
                    if current.name == 'p' or current.name == 'blockquote':
                        text = current.get_text(strip=True)
                        if text and not text.startswith('Time Complexity') and not text.startswith('Space Complexity'):
                            content_parts.append(text)
                    elif current.name == 'ul':
                        # Add list items
                        for li in current.find_all('li'):
                            content_parts.append("• " + li.get_text(strip=True))
                    
                    current = current.next_sibling
                
                approach['description'] = ' '.join(content_parts)
                
                # Extract complexity from the description
                complexity_text = approach['description']
                
                # Time complexity
                time_match = re.search(r'Time Complexity[:\s]*([^,\n\.]*)', complexity_text, re.IGNORECASE)
                if time_match:
                    approach['time_complexity'] = self.clean_text(time_match.group(1))
                
                # Space complexity
                space_match = re.search(r'Space Complexity[:\s]*([^,\n\.]*)', complexity_text, re.IGNORECASE)
                if space_match:
                    approach['space_complexity'] = self.clean_text(space_match.group(1))
                
                approaches.append(approach)
        
        return approaches
    
    def extract_code_by_language(self, soup):
        """Extract code implementations by programming language"""
        code_implementations = {}
        
        # Find all gfg-panel elements (these contain code for different languages)
        gfg_panels = soup.find_all('gfg-panel')
        
        for panel in gfg_panels:
            lang = panel.get('data-code-lang', 'unknown')
            code_text = panel.get_text(strip=True)
            
            if code_text and len(code_text) > 30:  # Filter out very short snippets
                # Map language names to readable format
                lang_map = {
                    'cpp': 'C++',
                    'c': 'C',
                    'java': 'Java',
                    'python3': 'Python',
                    'python': 'Python',
                    'javascript': 'JavaScript',
                    'csharp': 'C#'
                }
                
                mapped_lang = lang_map.get(lang.lower(), lang.title())
                code_implementations[mapped_lang] = self.clean_code(code_text)
        
        return code_implementations
    
    def scrape_article(self, url):
        """Scrape a single article and return structured JSON data"""
        try:
            logger.info(f"Scraping article: {url}")
            response = self.session.get(url)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.content, 'html.parser')
            
            # Initialize structured data
            article_data = {
                'url': url,
                'title': '',
                'description': '',
                'last_updated': '',
                'examples': [],
                'approaches': [],
                'code_implementations': {},
                'table_of_contents': [],
                'tags': []
            }
            
            # Extract title
            title_elem = soup.find('div', class_='article-title')
            if title_elem:
                h1 = title_elem.find('h1')
                if h1:
                    article_data['title'] = self.clean_text(h1.get_text())
            
            # Extract last updated
            last_updated_elem = soup.find('div', class_='last_updated_parent')
            if last_updated_elem:
                spans = last_updated_elem.find_all('span')
                if len(spans) >= 2:
                    article_data['last_updated'] = self.clean_text(spans[1].get_text())
            
            # Extract description (first few paragraphs)
            article_elem = soup.find('article', class_=lambda x: x and 'content' in x)
            if article_elem:
                paragraphs = article_elem.find_all('p')[:3]  # First 3 paragraphs
                description_parts = []
                for p in paragraphs:
                    text = p.get_text(strip=True)
                    if text and not text.startswith('Last Updated'):
                        description_parts.append(text)
                article_data['description'] = ' '.join(description_parts)
            
            # Extract examples
            article_data['examples'] = self.extract_examples(soup)
            
            # Extract approaches
            article_data['approaches'] = self.extract_approaches(soup)
            
            # Extract code implementations
            article_data['code_implementations'] = self.extract_code_by_language(soup)
            
            # Extract table of contents
            toc_elem = soup.find('div', id='table_of_content')
            if toc_elem:
                toc_links = toc_elem.find_all('a')
                for link in toc_links:
                    article_data['table_of_contents'].append(self.clean_text(link.get_text()))
            
            # Extract tags
            article_elem = soup.find('article')
            if article_elem:
                classes = article_elem.get('class', [])
                for cls in classes:
                    if cls.startswith('tag-'):
                        tag = cls.replace('tag-', '').replace('-', ' ').title()
                        if tag not in article_data['tags']:
                            article_data['tags'].append(tag)
            
            return article_data
            
        except requests.RequestException as e:
            logger.error(f"Error scraping article {url}: {e}")
            return None
    
    def save_to_json(self, article_data, filename=None):
        """Save article data to JSON file"""
        if not article_data:
            return
        
        if not filename:
            # Create filename from title
            safe_title = re.sub(r'[^\w\s-]', '', article_data['title'])
            safe_title = re.sub(r'[-\s]+', '-', safe_title)
            filename = f"{safe_title[:50]}.json"
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(article_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Saved article data to {filename}")
        return filename

def main():
    """Main function to test the scraper"""
    scraper = SimpleGeeksForGeeksScraper()
    
    # Test with the array reverse URL
    test_url = "https://www.geeksforgeeks.org/dsa/program-to-reverse-an-array/"
    print(f"Testing with URL: {test_url}")
    
    # Scrape the article
    article_data = scraper.scrape_article(test_url)
    
    if article_data:
        # Save to JSON
        filename = scraper.save_to_json(article_data)
        print(f"Successfully scraped and saved to {filename}")
        
        # Print summary
        print(f"\nTitle: {article_data['title']}")
        print(f"Examples found: {len(article_data['examples'])}")
        print(f"Approaches found: {len(article_data['approaches'])}")
        print(f"Code implementations: {list(article_data['code_implementations'].keys())}")
        print(f"Table of contents: {len(article_data['table_of_contents'])} items")
        
    else:
        print("Failed to scrape the article")

if __name__ == "__main__":
    main()
