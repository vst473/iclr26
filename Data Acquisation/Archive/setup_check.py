#!/usr/bin/env python3
"""
Archive Scraper Setup and Validation Script
Run this script to validate your configuration and setup.
"""

import os
import sys
import importlib.util

def check_dependencies():
    """Check if all required dependencies are installed."""
    required_packages = [
        'requests', 'bs4', 'asyncio', 'aiohttp', 'aiofiles', 
        'tenacity', 'aiohttp_socks', 'tqdm'
    ]
    
    missing = []
    for package in required_packages:
        spec = importlib.util.find_spec(package)
        if spec is None:
            missing.append(package)
    
    if missing:
        print("❌ Missing dependencies:")
        for pkg in missing:
            print(f"   - {pkg}")
        print("\nInstall with: pip install -r requirements.txt")
        return False
    else:
        print("✅ All dependencies installed")
        return True

def check_configuration():
    """Check if configuration files have been updated."""
    config_files = {
        '1_extract_metadata.py': ['FOLDER', 'COLLECTION', 'QUERY', 'LANGUAGES'],
        '2_build_url.py': ['METADATA_FOLDER', 'OUTPUT_FILENAME'],
        '3_download_pdf.py': ['LANGUAGE', 'URL_FILE']
    }
    
    issues = []
    for filename, vars_to_check in config_files.items():
        if not os.path.exists(filename):
            issues.append(f"❌ {filename} not found")
            continue
            
        with open(filename, 'r', encoding='utf-8') as f:
            content = f.read()
            
        for var in vars_to_check:
            if f'{var} = "your_' in content:
                issues.append(f"❌ {filename}: Update {var} variable (still has template value)")
    
    if issues:
        print("\n".join(issues))
        print("\n💡 See CONFIG_TEMPLATE.md for examples")
        return False
    else:
        print("✅ Configuration files updated")
        return True

def main():
    """Main setup validation."""
    print("🔍 Archive Scraper Setup Validation")
    print("=" * 40)
    
    deps_ok = check_dependencies()
    config_ok = check_configuration()
    
    print("\n" + "=" * 40)
    if deps_ok and config_ok:
        print("🎉 Setup complete! Ready to run the pipeline:")
        print("   1. python 1_extract_metadata.py")
        print("   2. python 2_build_url.py")
        print("   3. python 3_download_pdf.py")
    else:
        print("❌ Setup incomplete. Please fix the issues above.")
        sys.exit(1)

if __name__ == "__main__":
    main()
