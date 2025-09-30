import json
import os
from pathlib import Path

# ============= CONFIGURATION VARIABLES - UPDATE THESE =============
METADATA_FOLDER = "metadata_example"                 # UPDATE: Folder containing metadata JSONL files (from step 1)
OUTPUT_FOLDER = "url"                                # UPDATE: Folder to save URL files
OUTPUT_FILENAME = "archive_urls_example.txt"        # UPDATE: Name of output URL file
# ================================================================

def collect_urls_from_json():
    # Directory containing JSON files of metadata
    json_dir = Path(METADATA_FOLDER)
    urls = set()
    
    print(f"Looking for metadata files in: {json_dir}")
    if not json_dir.exists():
        print(f"ERROR: Metadata directory {json_dir} does not exist!")
        print("Make sure to run step 2 (extract_metadata.py) first.")
        return
    
    # Iterate through all JSON files in the directory
    for json_file in json_dir.glob("*.jsonl"):
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = f.readlines()
                data = [json.loads(line) for line in data]
                # Check if data is a list or single item
                if isinstance(data, list):
                    items = data
                else:
                    items = [data]
                
                # Build URLs for each identifier
                for item in items:
                    if 'identifier' in item['fields']:
                        identifier = item['fields']['identifier']
                        url = f"https://archive.org/details/{identifier}"
                        urls.add(url)
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    if not urls:
        print("No URLs found! Check if metadata files contain valid data.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    
    # Save all URLs to a text file
    output_file = os.path.join(OUTPUT_FOLDER, OUTPUT_FILENAME)
    with open(output_file, 'w', encoding='utf-8') as f:
        for url in sorted(urls):  # Sort URLs for better organization
            f.write(url + '\n')
    
    print(f"Total URLs collected: {len(urls)}")
    print(f"URLs saved to: {output_file}")
    print(f"Ready for step 4: Download PDFs using this URL file!")

if __name__ == "__main__":
    collect_urls_from_json()