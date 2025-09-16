#!/usr/bin/env python3
"""
Konfigurasi untuk Scraper Peraturan BPK Indonesia
"""

import os

# URL Configuration
BASE_URL = "https://peraturan.bpk.go.id"
SEARCH_URL = f"{BASE_URL}/Search"

# Download Configuration
DEFAULT_DOWNLOAD_DIR = "downloads"
MAX_RETRIES = 3
REQUEST_TIMEOUT = 30
DOWNLOAD_TIMEOUT = 60

# Rate Limiting Configuration
DELAY_BETWEEN_REQUESTS = 1  # seconds
DELAY_BETWEEN_PAGES = 2     # seconds
DELAY_BETWEEN_YEARS = 5     # seconds

# File Configuration
MAX_FILENAME_LENGTH = 200
ALLOWED_FILE_EXTENSIONS = ['.pdf','.doc','.docx']

# Logging Configuration
LOG_LEVEL = "INFO"
LOG_FORMAT = '%(asctime)s - %(levelname)s - %(message)s'
LOG_FILE = "scraper.log"

# User Agent
USER_AGENT = 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'

# Year Range
DEFAULT_START_YEAR = 1945
DEFAULT_END_YEAR = 2025

# Pagination
MAX_PAGES_PER_YEAR = 1000  # Safety limit

# Resume Configuration
METADATA_FILE = "metadata.json"
ENABLE_RESUME = True

# Error Handling
MAX_CONSECUTIVE_ERRORS = 5
SKIP_EXISTING_FILES = True

# Content Validation
MIN_PDF_SIZE = 1024  # bytes (1KB minimum)
VALIDATE_PDF_CONTENT = True

# Advanced Configuration
USE_SESSION = True
VERIFY_SSL = True
FOLLOW_REDIRECTS = True

# Custom Headers
CUSTOM_HEADERS = {
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8',
    'Accept-Language': 'id-ID,id;q=0.9,en;q=0.8',
    'Accept-Encoding': 'gzip, deflate, br',
    'DNT': '1',
    'Connection': 'keep-alive',
    'Upgrade-Insecure-Requests': '1',
}

# Environment Variable Overrides
def get_config_value(key, default):
    """Get configuration value from environment variable or use default"""
    env_key = f"SCRAPER_{key.upper()}"
    return os.getenv(env_key, default)

# Apply environment overrides
BASE_URL = get_config_value('base_url', BASE_URL)
DEFAULT_DOWNLOAD_DIR = get_config_value('download_dir', DEFAULT_DOWNLOAD_DIR)
REQUEST_TIMEOUT = int(get_config_value('request_timeout', REQUEST_TIMEOUT))
DOWNLOAD_TIMEOUT = int(get_config_value('download_timeout', DOWNLOAD_TIMEOUT))
DELAY_BETWEEN_REQUESTS = float(get_config_value('delay_requests', DELAY_BETWEEN_REQUESTS))
DELAY_BETWEEN_PAGES = float(get_config_value('delay_pages', DELAY_BETWEEN_PAGES))
DELAY_BETWEEN_YEARS = float(get_config_value('delay_years', DELAY_BETWEEN_YEARS))
LOG_LEVEL = get_config_value('log_level', LOG_LEVEL)
USER_AGENT = get_config_value('user_agent', USER_AGENT)

# Validation
if DELAY_BETWEEN_REQUESTS < 0.5:
    print("Warning: DELAY_BETWEEN_REQUESTS is less than 0.5 seconds. This may cause rate limiting.")

if REQUEST_TIMEOUT < 10:
    print("Warning: REQUEST_TIMEOUT is less than 10 seconds. This may cause timeouts.")

if DOWNLOAD_TIMEOUT < 30:
    print("Warning: DOWNLOAD_TIMEOUT is less than 30 seconds. Large files may fail to download.")

# Configuration Summary
def print_config():
    """Print current configuration"""
    print("Current Scraper Configuration:")
    print("=" * 40)
    print(f"Base URL: {BASE_URL}")
    print(f"Download Directory: {DEFAULT_DOWNLOAD_DIR}")
    print(f"Request Timeout: {REQUEST_TIMEOUT}s")
    print(f"Download Timeout: {DOWNLOAD_TIMEOUT}s")
    print(f"Delay Between Requests: {DELAY_BETWEEN_REQUESTS}s")
    print(f"Delay Between Pages: {DELAY_BETWEEN_PAGES}s")
    print(f"Delay Between Years: {DELAY_BETWEEN_YEARS}s")
    print(f"Log Level: {LOG_LEVEL}")
    print(f"Year Range: {DEFAULT_START_YEAR}-{DEFAULT_END_YEAR}")
    print(f"Resume Enabled: {ENABLE_RESUME}")
    print(f"Skip Existing Files: {SKIP_EXISTING_FILES}")
    print("=" * 40)

if __name__ == "__main__":
    print_config()