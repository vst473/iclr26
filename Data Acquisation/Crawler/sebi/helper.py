import re
import json
import requests
from bs4 import BeautifulSoup
import io
from lxml import etree

def convert_js_to_html(js_string):
    """
    Convert JavaScript document.write() statements to HTML.
    
    Args:
        js_string (str): JavaScript code containing document.write() statements
    
    Returns:
        str: Resulting HTML string
    """
    # Split the input string by document.write statements
    lines = js_string.split('document.write(')
    
    html_output = ""
    
    # Process each line
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Find where the document.write statement ends
        closing_paren_index = line.find(');')
        if closing_paren_index == -1:
            continue
            
        # Extract the HTML content (remove quotes and escapes)
        html_content = line[:closing_paren_index]
        
        # Remove opening and closing quotes
        if html_content.startswith('"') and html_content.endswith('"'):
            html_content = html_content[1:-1]
        elif html_content.startswith("'") and html_content.endswith("'"):
            html_content = html_content[1:-1]
            
        # Unescape quotes
        html_content = html_content.replace('\\"', '"').replace("\\'", "'")
        
        # Add to our output
        html_output += html_content
        
    return html_output

def find_home_action_links(url):
    """
    Find all links that contain 'HomeAction' in their href attribute using BeautifulSoup and XPath
    
    Args:
        html_content (str): HTML content to search through
        
    Returns:
        list: List of URLs containing 'HomeAction'
    """
    
    js_content = requests.get(url).text
    
    html_content = convert_js_to_html(js_content)
    # Parse with BeautifulSoup first
    soup = BeautifulSoup(html_content, 'html.parser')

    # Convert to lxml for xpath
    dom = etree.parse(io.StringIO(str(soup)), etree.HTMLParser())
        
        # Find all links with HomeAction using xpath
    links = dom.xpath("//a[contains(@href, 'HomeAction')]/@href")
    
    return list(links)


def save_menu_to_txt(link_list, output_file='sebi_menu_structure.txt'):
    """
    Save the extracted menu structure to a JSON file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(link_list))
    print(f"Menu data saved to {output_file}")
    
    return output_file


def load_menu_from_txt(input_file='sebi_menu_structure.txt'):
    """
    Load the menu structure from a JSON file
    
    Args:
        input_file (str): Path to the JSON file
        
    Returns:
        dict: The menu structure loaded from the file
    """
    with open(input_file, 'r', encoding='utf-8') as f:
        link_list = f.readlines()
        link_list = [link.strip() for link in link_list]
        # Remove empty lines
        link_list = [link for link in link_list if link]
        
    return link_list

