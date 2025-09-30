#!/usr/bin/env python3
"""
Archive Scraper - Complete Pipeline Runner
Runs the complete scraping pipeline with error handling and progress tracking.

Usage:
    python run_pipeline.py [--config CONFIG_NAME]

Examples:
    python run_pipeline.py --config mathematics
    python run_pipeline.py --config hindi_books
"""

import os
import sys
import subprocess
import argparse
from datetime import datetime

def run_command(command, description):
    """Run a command with error handling."""
    print(f"\n🔄 {description}")
    print(f"Command: {command}")
    print("-" * 50)
    
    try:
        result = subprocess.run(command, shell=True, check=True, 
                              capture_output=False, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed with exit code {e.returncode}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run Archive Scraper Pipeline')
    parser.add_argument('--config', help='Configuration name (for logging)', default='default')
    args = parser.parse_args()
    
    print("🚀 Archive Scraper Pipeline")
    print("=" * 60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Config: {args.config}")
    print("=" * 60)
    
    # Check if configuration files exist
    required_files = ['1_extract_metadata.py', '2_build_url.py', '3_download_pdf.py']
    for file in required_files:
        if not os.path.exists(file):
            print(f"❌ Required file {file} not found!")
            sys.exit(1)
    
    # Determine the correct Python interpreter
    python_cmd = sys.executable
    
    # Run the pipeline
    steps = [
        (f"{python_cmd} 1_extract_metadata.py", "Step 1: Extract Metadata"),
        (f"{python_cmd} 2_build_url.py", "Step 2: Build URL List"),
        (f"{python_cmd} 3_download_pdf.py", "Step 3: Download Files")
    ]
    
    for i, (command, description) in enumerate(steps, 1):
        if not run_command(command, description):
            print(f"\n❌ Pipeline failed at step {i}")
            sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Pipeline completed successfully!")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

if __name__ == "__main__":
    main()
