import scrapy
import re
from urllib.parse import urljoin


class AimeSpider(scrapy.Spider):
    name = 'aime'
    start_urls = ['https://artofproblemsolving.com/wiki/index.php/AIME_Problems_and_Solutions']
    
    custom_settings = {
        'DOWNLOAD_DELAY': 1,
        'RANDOMIZE_DOWNLOAD_DELAY': 0.5,
        'USER_AGENT': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
    }

    def parse(self, response):
        """Parse the main AIME problems page to find year links"""

        aime_links = response.xpath('//*[@id="mw-content-text"]/div/table/tbody/tr/td/a/@href')
        # aime_links = []
        
        for link in aime_links:
            full_url = urljoin(response.url, link.get())
            yield response.follow(full_url, self.parse_year)

        
    def parse_year(self, response):
        """Parse problems from a specific AIME year page"""
        # Extract year from URL
        year_match = re.search(r'(\d{4})', response.url)
        year = year_match.group(1) if year_match else "Unknown"

        problem_links = response.xpath('//*[@id="mw-content-text"]/div/ul[1]/li[2]/ul/li/a/@href').getall()
        for link in problem_links:
            yield response.follow(link, self.parse_problems, cb_kwargs={'year': year})

    def parse_problems(self, response, year):
        # Extract problem number from URL
        problem_match = re.search(r'Problem[_\s](\d+)', response.url)
        if not problem_match:
            self.logger.warning(f"Could not extract problem number from URL: {response.url}")
            return
            
        problem_number = problem_match.group(1)
        
        # Find the main content area
        content_div = response.css('div#mw-content-text .mw-parser-output')
        if not content_div:
            self.logger.warning(f"Could not find content div for {response.url}")
            return
            
        # Extract problem text using the Problem section
        problem_text = self.extract_problem_text_from_section(content_div)
        
        # Extract all solutions
        solutions = self.extract_all_solutions(content_div)
        
        # Combine all solutions
        combined_solutions = '\n\n'.join(solutions) if solutions else ''
        
        yield {
            'year': year,
            'problem_number': problem_number,
            'problem_text': problem_text.strip() if problem_text else '',
            'solutions': combined_solutions.strip(),
            'url': response.url
        }
        
        self.logger.info(f"Extracted problem {problem_number} from {year} with {len(solutions)} solutions")

    def extract_problem_text_from_section(self, content_div):
        """Extract problem text from the Problem section"""
        # Find the Problem section header
        problem_header = content_div.xpath('//h2[span[@class="mw-headline" and @id="Problem"]]').get()
        if not problem_header:
            # Try alternative: look for first paragraph after table of contents
            problem_paragraph = content_div.xpath('//div[@class="toc"]/following-sibling::p[1]').get()
            if problem_paragraph:
                return self.extract_text_with_latex(problem_paragraph)
            return ""
        
        # Get the first paragraph following the Problem header
        problem_paragraph = content_div.xpath('//h2[span[@class="mw-headline" and @id="Problem"]]/following-sibling::p[1]').get()
        if problem_paragraph:
            return self.extract_text_with_latex(problem_paragraph)
        
        return ""
    
    def extract_text_with_latex(self, html_content):
        """Extract text and LaTeX from HTML content, preserving order"""
        from scrapy import Selector
        selector = Selector(text=html_content)
        
        # Convert the HTML content to text while preserving LaTeX
        result = html_content
        
        # Find all LaTeX images and replace them with their alt text
        latex_imgs = selector.xpath('//img[@class="latex"]')
        for img in latex_imgs:
            alt_text = img.xpath('@alt').get()
            if alt_text:
                # Clean up the alt text and wrap in $ if not already wrapped
                latex_clean = alt_text.strip('$')
                latex_formatted = f'${latex_clean}$'
                
                # Replace the entire img tag with the LaTeX
                img_html = img.get()
                result = result.replace(img_html, latex_formatted)
        
        # Remove all HTML tags except the LaTeX we just added
        import re
        result = re.sub(r'<[^>]+>', ' ', result)
        
        # Clean up whitespace
        result = re.sub(r'\s+', ' ', result)
        result = re.sub(r'\n\s*\n', '\n\n', result)
        
        return result.strip()

    def extract_problem_section(self, content_div, section_name):
        """Extract content from a specific section (Problem, Solution, etc.)"""
        # Find the section header
        section_header = content_div.css(f'h2:contains("{section_name}"), h3:contains("{section_name}")').get()
        if not section_header:
            return ""
            
        # Get all content following the header until the next header
        text_content = []
        
        # Find the section using xpath
        section_xpath = f'//h2[span[@class="mw-headline" and @id="{section_name}"]]/following-sibling::*'
        following_elements = content_div.xpath(section_xpath)
        
        for elem in following_elements:
            # Stop if we hit another h2 header
            if elem.xpath('.//h2'):
                break
                
            # Extract and clean text
            element_text = self.clean_element_text(elem)
            if element_text.strip():
                text_content.append(element_text)
                
        return ' '.join(text_content)
    
    def extract_all_solutions(self, content_div):
        """Extract all solution sections"""
        solutions = []
        
        # Find all solution headers (Solution 1, Solution 2, Video Solution, etc.)
        solution_headers = content_div.xpath('//h3[span[@class="mw-headline" and contains(@id, "Solution_")]]')
        if len(solution_headers)  == 0:
            solution_headers = content_div.xpath('//h2[span[@class="mw-headline" and contains(@id, "Solution")]]')
        
        for header in solution_headers:
            solution_id = header.xpath('.//span[@class="mw-headline"]/@id').get()
            if not solution_id:
                continue
                
            # Skip video solutions
            if 'video' in solution_id.lower():
                continue
                
            # Get content following this solution header
            text_content = []
            following_elements = header.xpath('./following-sibling::*')
            
            for elem in following_elements:
                # Stop if we hit another h2 header
                if elem.root.tag in ['h2', 'h3']:
                    break
                    
                # Extract and clean text
                element_text = self.clean_element_text(elem)
                if element_text.strip():
                    text_content.append(element_text)
            
            if text_content:
                solution_text = ' '.join(text_content)
                solutions.append(f"**{solution_id.replace('_', ' ').replace('.28', '(').replace('.29', ')')}**\n{solution_text}")
                
        return solutions
    
    def clean_element_text(self, element):
        """Clean and extract text from an element, removing images and video links"""
        # Get the HTML content of the element
        element_html = element.get()
        
        # Remove video links first
        from scrapy import Selector
        temp_selector = Selector(text=element_html)
        
        # Remove video links (external links to youtube, etc.)
        video_links = temp_selector.xpath('//a[contains(@href, "youtube") or contains(@href, "youtu.be")]')
        for link in video_links:
            link_html = link.get()
            element_html = element_html.replace(link_html, '')
        
        # Now extract text with LaTeX using the same method
        return self.extract_text_with_latex(element_html)

    def extract_problem_text(self, section_element, response):
        """Extract problem text following a problem header (legacy method - keeping for compatibility)"""
        return self.extract_problem_section(response.css('div#mw-content-text .mw-parser-output'), 'Problem')


if __name__ == "__main__":
    from scrapy.crawler import CrawlerProcess
    process = CrawlerProcess()
    process.crawl(AimeSpider)
    process.start()
