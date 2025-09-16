#!/usr/bin/env python3
"""
Scraper untuk mengunduh data perundang-undangan Indonesia dari BPK
Website: https://peraturan.bpk.go.id/
Tahun: 1945-2025
"""

import requests
import os
import time
import json
from urllib.parse import urljoin, urlparse
from bs4 import BeautifulSoup
import logging
from datetime import datetime
import re
from config import *

# Setup logging
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL),
    format=LOG_FORMAT,
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PeraturanScraper:
    def __init__(self, download_dir=None):
        self.base_url = BASE_URL
        self.search_url = SEARCH_URL
        self.download_dir = download_dir or DEFAULT_DOWNLOAD_DIR
        
        if USE_SESSION:
            self.session = requests.Session()
        else:
            self.session = requests
            
        # Setup headers
        headers = {'User-Agent': USER_AGENT}
        headers.update(CUSTOM_HEADERS)
        
        if USE_SESSION:
            self.session.headers.update(headers)
        
        # Create download directory
        os.makedirs(self.download_dir, exist_ok=True)
        
        # Metadata untuk tracking
        self.metadata_file = os.path.join(self.download_dir, METADATA_FILE)
        self.metadata = self.load_metadata() if ENABLE_RESUME else {}
    
    def load_metadata(self):
        """Load metadata file yang berisi daftar file yang sudah didownload"""
        if os.path.exists(self.metadata_file):
            try:
                with open(self.metadata_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def save_metadata(self):
        """Simpan metadata ke file"""
        with open(self.metadata_file, 'w', encoding='utf-8') as f:
            json.dump(self.metadata, f, ensure_ascii=False, indent=2)
    
    def sanitize_filename(self, filename):
        """Bersihkan nama file dari karakter yang tidak valid"""
        # Hapus karakter yang tidak valid untuk nama file
        filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
        # Batasi panjang nama file
        if len(filename) > 200:
            filename = filename[:200]
        return filename
    
    def get_search_results(self, tahun, page=1):
        """Ambil hasil pencarian untuk tahun dan halaman tertentu"""
        try:
            params = {
                'tahun': tahun,
                'page': page
            }
            
            # Setup request parameters
            request_kwargs = {
                'params': params,
                'timeout': REQUEST_TIMEOUT,
                'verify': VERIFY_SSL,
                'allow_redirects': FOLLOW_REDIRECTS
            }
            
            if not USE_SESSION:
                request_kwargs['headers'] = {'User-Agent': USER_AGENT}
                request_kwargs['headers'].update(CUSTOM_HEADERS)
            
            response = self.session.get(self.search_url, **request_kwargs)
            response.raise_for_status()
            
            return response.text
            
        except Exception as e:
            logger.error(f"Error mengambil data tahun {tahun} halaman {page}: {e}")
            return None
    
    def parse_search_results(self, html_content):
        """Parse hasil pencarian untuk mendapatkan link download PDF"""
        soup = BeautifulSoup(html_content, 'html.parser')
        results = []
        
        # Cari semua item peraturan
        peraturan_items = soup.find_all('div', class_='card-body')
        
        for item in peraturan_items:
            try:
                # Ambil judul peraturan dari text element pertama yang mengandung "UU" atau "PP"
                title = None
                all_text_elements = item.find_all(text=True)
                non_empty_texts = [t.strip() for t in all_text_elements if t.strip()]
                
                for text in non_empty_texts:
                    # Cari text yang mengandung pola peraturan (UU, PP, UUD, dll)
                    if any(keyword in text for keyword in ['Undang-undang (UU)', 'Undang-undang Dasar (UUD)', 'Peraturan Pemerintah (PP)', 'UU Nomor', 'PP Nomor', 'UUD Tahun']):
                        title = text
                        break
                
                if not title:
                    continue
                    
                # Cari link download PDF
                download_links = item.find_all('a', href=True)
                pdf_links = []
                
                for link in download_links:
                    href = link.get('href', '')
                    if 'download' in href.lower() or href.endswith('.pdf'):
                        full_url = urljoin(self.base_url, href)
                        pdf_links.append(full_url)
                
                if pdf_links:
                    results.append({
                        'title': title,
                        'pdf_links': pdf_links
                    })
                    
            except Exception as e:
                logger.warning(f"Error parsing item: {e}")
                continue
        
        return results
    
    def get_total_pages(self, html_content):
        """Ambil total halaman dari pagination"""
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Cari elemen pagination
        pagination = soup.find('nav', {'aria-label': 'Page navigation'}) or soup.find('ul', class_='pagination')
        
        if not pagination:
            return 1
        
        # Cari nomor halaman terakhir
        page_links = pagination.find_all('a', href=True)
        max_page = 1
        
        for link in page_links:
            text = link.get_text(strip=True)
            if text.isdigit():
                max_page = max(max_page, int(text))
        
        return max_page
    
    def download_file(self, url, filename, max_retries=None):
        """Download file dengan retry mechanism"""
        if max_retries is None:
            max_retries = MAX_RETRIES
            
        full_path = os.path.join(self.download_dir, filename)
        
        # Skip jika file sudah ada dan konfigurasi mengizinkan
        if SKIP_EXISTING_FILES and os.path.exists(full_path):
            file_size = os.path.getsize(full_path)
            if file_size >= MIN_PDF_SIZE:
                logger.info(f"File sudah ada: {filename} ({file_size} bytes)")
                return True
            else:
                logger.warning(f"File ada tapi terlalu kecil, akan didownload ulang: {filename}")
                os.remove(full_path)
            
        for attempt in range(max_retries):
            try:
                logger.info(f"Downloading: {filename} (attempt {attempt + 1})")
                
                # Setup request parameters
                request_kwargs = {
                    'stream': True,
                    'timeout': DOWNLOAD_TIMEOUT,
                    'verify': VERIFY_SSL,
                    'allow_redirects': FOLLOW_REDIRECTS
                }
                
                if not USE_SESSION:
                    request_kwargs['headers'] = {'User-Agent': USER_AGENT}
                    request_kwargs['headers'].update(CUSTOM_HEADERS)
                
                response = self.session.get(url, **request_kwargs)
                response.raise_for_status()
                
                # Check content type
                content_type = response.headers.get('content-type', '').lower()
                if 'pdf' not in content_type and 'application/octet-stream' not in content_type:
                    logger.warning(f"Unexpected content type for {filename}: {content_type}")
                
                with open(full_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                
                # Verifikasi file berhasil didownload
                if os.path.exists(full_path):
                    file_size = os.path.getsize(full_path)
                    if file_size >= MIN_PDF_SIZE:
                        logger.info(f"Berhasil download: {filename} ({file_size} bytes)")
                        return True
                    else:
                        logger.error(f"File terlalu kecil: {filename} ({file_size} bytes)")
                        os.remove(full_path)
                else:
                    logger.error(f"File tidak ditemukan setelah download: {filename}")
                        
            except Exception as e:
                logger.error(f"Error downloading {filename} (attempt {attempt + 1}): {str(e)}")
                if os.path.exists(full_path):
                    os.remove(full_path)
                    
                if attempt < max_retries - 1:
                    sleep_time = min(2 ** attempt, 30)  # Exponential backoff, max 30s
                    logger.info(f"Waiting {sleep_time}s before retry...")
                    time.sleep(sleep_time)
                    
        return False
    
    def scrape_year(self, tahun):
        """Scrape semua peraturan untuk tahun tertentu"""
        logger.info(f"Memulai scraping untuk tahun {tahun}")
        
        # Buat direktori untuk tahun
        year_dir = os.path.join(self.download_dir, str(tahun))
        os.makedirs(year_dir, exist_ok=True)
        
        total_downloaded = 0
        page = 1
        consecutive_errors = 0
        seen_titles = set()  # Track titles yang sudah dilihat untuk deteksi duplicate
        max_pages = None  # Cache max pages dari pagination
        
        while True:
            try:
                logger.info(f"Scraping tahun {tahun}, halaman {page}")
                
                # Get data untuk halaman ini
                html_content = self.get_search_results(tahun, page)
                
                if not html_content:
                    logger.info(f"Tidak ada data lagi untuk tahun {tahun}, halaman {page}")
                    break
                
                # Ambil max pages dari pagination jika belum ada
                if max_pages is None:
                    max_pages = self.get_total_pages(html_content)
                    logger.info(f"Total halaman untuk tahun {tahun}: {max_pages}")
                
                # Jika halaman saat ini melebihi max pages, stop
                if page > max_pages:
                    logger.info(f"Halaman {page} melebihi total halaman ({max_pages}) untuk tahun {tahun}")
                    break
                
                # Parse hasil
                results = self.parse_search_results(html_content)
                
                if not results:
                    logger.info(f"Tidak ada hasil untuk tahun {tahun}, halaman {page}")
                    break
                
                # Check untuk duplicate content (indikasi sudah mencapai akhir)
                current_titles = set(result['title'] for result in results)
                if seen_titles and current_titles.issubset(seen_titles):
                    logger.info(f"Duplicate content terdeteksi pada halaman {page} untuk tahun {tahun}, menghentikan scraping")
                    break
                
                # Update seen titles
                seen_titles.update(current_titles)
                
                # Download semua PDF di halaman ini
                downloaded_count = 0
                for result in results:
                    title = result['title']
                    for i, pdf_url in enumerate(result['pdf_links']):
                        # Buat nama file yang aman
                        safe_title = self.sanitize_filename(title)
                        if len(result['pdf_links']) > 1:
                            filename = f"{safe_title}_{i+1}.pdf"
                        else:
                            filename = f"{safe_title}.pdf"
                        
                        # Generate filename dengan path tahun
                        year_filename = os.path.join(str(tahun), filename)
                        
                        if self.download_file(pdf_url, year_filename):
                            downloaded_count += 1
                            total_downloaded += 1
                            
                            # Update metadata jika resume diaktifkan
                            if ENABLE_RESUME:
                                self.metadata[pdf_url] = {
                                    'title': title,
                                    'filename': year_filename,
                                    'tahun': tahun,
                                    'download_date': datetime.now().isoformat()
                                }
                                
                                # Save metadata setiap 10 file
                                if total_downloaded % 10 == 0:
                                    self.save_metadata()
                        
                        # Delay antar download
                        time.sleep(DELAY_BETWEEN_REQUESTS)
                
                logger.info(f"Downloaded {downloaded_count} files dari halaman {page}")
                
                # Reset error counter jika berhasil
                consecutive_errors = 0
                
                # Delay antar halaman
                time.sleep(DELAY_BETWEEN_PAGES)
                page += 1
                
                # Safety check untuk mencegah infinite loop
                if page > MAX_PAGES_PER_YEAR:
                    logger.warning(f"Mencapai batas maksimum halaman ({MAX_PAGES_PER_YEAR}) untuk tahun {tahun}")
                    break
                    
            except Exception as e:
                consecutive_errors += 1
                logger.error(f"Error pada tahun {tahun}, halaman {page}: {str(e)}")
                
                if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                    logger.error(f"Terlalu banyak error berturut-turut ({MAX_CONSECUTIVE_ERRORS}) untuk tahun {tahun}. Melanjutkan ke tahun berikutnya.")
                    break
                    
                # Delay lebih lama jika ada error
                time.sleep(DELAY_BETWEEN_PAGES * 2)
                page += 1
        
        # Save metadata final jika resume diaktifkan
        if ENABLE_RESUME:
            self.save_metadata()
        
        logger.info(f"Selesai scraping tahun {tahun}. Total downloaded: {total_downloaded} files")
        return total_downloaded
    
    def scrape_all_years(self, start_year=None, end_year=None):
        """Scrape semua tahun dari start_year sampai end_year"""
        if start_year is None:
            start_year = DEFAULT_START_YEAR
        if end_year is None:
            end_year = DEFAULT_END_YEAR
            
        logger.info(f"Memulai scraping dari tahun {start_year} sampai {end_year}")
        
        total_files = 0
        failed_years = []
        
        for tahun in range(start_year, end_year + 1):
            try:
                logger.info(f"\n{'='*50}")
                logger.info(f"SCRAPING TAHUN {tahun}")
                logger.info(f"{'='*50}")
                
                downloaded = self.scrape_year(tahun)
                total_files += downloaded
                
                if downloaded > 0:
                    logger.info(f"Selesai tahun {tahun}. Downloaded: {downloaded} files")
                else:
                    logger.warning(f"Tidak ada file yang berhasil didownload untuk tahun {tahun}")
                    failed_years.append(tahun)
                
                # Delay antar tahun untuk menghindari rate limiting
                if tahun < end_year:
                    logger.info(f"Menunggu {DELAY_BETWEEN_YEARS}s sebelum melanjutkan ke tahun berikutnya...")
                    time.sleep(DELAY_BETWEEN_YEARS)
                    
            except KeyboardInterrupt:
                logger.info("Scraping dihentikan oleh user")
                break
            except Exception as e:
                logger.error(f"Error pada tahun {tahun}: {str(e)}")
                failed_years.append(tahun)
                continue
        
        logger.info(f"\n{'='*50}")
        logger.info(f"SCRAPING SELESAI")
        logger.info(f"Total files downloaded: {total_files}")
        if failed_years:
            logger.warning(f"Tahun yang gagal: {failed_years}")
        logger.info(f"{'='*50}")
        
        return total_files, failed_years

def main():
    """Fungsi utama"""
    scraper = PeraturanScraper()
    
    # Scrape dari tahun 1945 sampai 2025
    scraper.scrape_all_years(1945, 2025)

if __name__ == "__main__":
    main()