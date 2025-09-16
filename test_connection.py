#!/usr/bin/env python3
"""
Script untuk menguji koneksi dan validasi website BPK
sebelum menjalankan scraper utama
"""

import requests
import time
from bs4 import BeautifulSoup
from urllib.parse import urljoin

def test_basic_connection():
    """Test koneksi dasar ke website BPK"""
    print("1. Testing basic connection...")
    
    try:
        response = requests.get("https://peraturan.bpk.go.id", timeout=10)
        if response.status_code == 200:
            print("   ✓ Website dapat diakses")
            return True
        else:
            print(f"   ✗ Website mengembalikan status code: {response.status_code}")
            return False
    except requests.RequestException as e:
        print(f"   ✗ Error koneksi: {e}")
        return False

def test_search_functionality():
    """Test fungsi pencarian"""
    print("2. Testing search functionality...")
    
    try:
        # Test pencarian untuk tahun 2025
        search_url = "https://peraturan.bpk.go.id/Search?tahun=2025"
        response = requests.get(search_url, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Cek apakah ada hasil pencarian
            result_text = response.text.lower()
            if "menemukan" in result_text and "peraturan" in result_text:
                print("   ✓ Fungsi pencarian bekerja")
                
                # Coba ambil jumlah hasil
                import re
                match = re.search(r'menemukan\s+([\d.,]+)\s+peraturan', result_text)
                if match:
                    count = match.group(1)
                    print(f"   ✓ Ditemukan {count} peraturan untuk tahun 2025")
                
                return True
            else:
                print("   ✗ Tidak ada hasil pencarian yang ditemukan")
                return False
        else:
            print(f"   ✗ Search mengembalikan status code: {response.status_code}")
            return False
            
    except requests.RequestException as e:
        print(f"   ✗ Error saat testing search: {e}")
        return False

def test_pdf_download():
    """Test kemampuan download PDF"""
    print("3. Testing PDF download capability...")
    
    try:
        # Ambil halaman pencarian untuk mencari link PDF
        search_url = "https://peraturan.bpk.go.id/Search?tahun=2025"
        response = requests.get(search_url, timeout=15)
        
        if response.status_code != 200:
            print("   ✗ Tidak dapat mengakses halaman pencarian")
            return False
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Cari link download PDF
        pdf_links = []
        for link in soup.find_all('a', href=True):
            href = link.get('href', '')
            if 'download' in href.lower() or href.endswith('.pdf'):
                full_url = urljoin("https://peraturan.bpk.go.id", href)
                pdf_links.append(full_url)
        
        if pdf_links:
            print(f"   ✓ Ditemukan {len(pdf_links)} link PDF")
            
            # Test download PDF pertama (hanya header)
            test_url = pdf_links[0]
            print(f"   Testing download: {test_url[:60]}...")
            
            head_response = requests.head(test_url, timeout=10)
            if head_response.status_code == 200:
                content_type = head_response.headers.get('content-type', '').lower()
                if 'pdf' in content_type:
                    print("   ✓ PDF dapat didownload")
                    return True
                else:
                    print(f"   ⚠ File bukan PDF (content-type: {content_type})")
                    return True  # Masih OK, mungkin server tidak set content-type dengan benar
            else:
                print(f"   ✗ PDF tidak dapat didownload (status: {head_response.status_code})")
                return False
        else:
            print("   ✗ Tidak ditemukan link PDF")
            return False
            
    except requests.RequestException as e:
        print(f"   ✗ Error saat testing PDF download: {e}")
        return False

def test_pagination():
    """Test sistem pagination"""
    print("4. Testing pagination system...")
    
    try:
        # Test halaman pertama
        search_url = "https://peraturan.bpk.go.id/Search?tahun=2025&page=1"
        response = requests.get(search_url, timeout=15)
        
        if response.status_code != 200:
            print("   ✗ Tidak dapat mengakses halaman pertama")
            return False
        
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Cari elemen pagination
        pagination = soup.find('nav', {'aria-label': 'Page navigation'}) or soup.find('ul', class_='pagination')
        
        if pagination:
            print("   ✓ Sistem pagination ditemukan")
            
            # Cari nomor halaman
            page_links = pagination.find_all('a', href=True)
            max_page = 1
            
            for link in page_links:
                text = link.get_text(strip=True)
                if text.isdigit():
                    max_page = max(max_page, int(text))
            
            if max_page > 1:
                print(f"   ✓ Ditemukan {max_page} halaman untuk tahun 2025")
                
                # Test halaman kedua
                page2_url = "https://peraturan.bpk.go.id/Search?tahun=2025&page=2"
                page2_response = requests.get(page2_url, timeout=15)
                
                if page2_response.status_code == 200:
                    print("   ✓ Halaman kedua dapat diakses")
                    return True
                else:
                    print("   ⚠ Halaman kedua tidak dapat diakses, tapi pagination terdeteksi")
                    return True
            else:
                print("   ⚠ Hanya ditemukan 1 halaman")
                return True
        else:
            print("   ⚠ Sistem pagination tidak ditemukan (mungkin hanya 1 halaman)")
            return True
            
    except requests.RequestException as e:
        print(f"   ✗ Error saat testing pagination: {e}")
        return False

def test_rate_limiting():
    """Test rate limiting dengan beberapa request berturut-turut"""
    print("5. Testing rate limiting...")
    
    try:
        base_url = "https://peraturan.bpk.go.id/Search?tahun=2025"
        
        # Lakukan 3 request berturut-turut
        for i in range(3):
            start_time = time.time()
            response = requests.get(base_url, timeout=10)
            end_time = time.time()
            
            if response.status_code == 200:
                response_time = end_time - start_time
                print(f"   Request {i+1}: OK ({response_time:.2f}s)")
            elif response.status_code == 429:  # Too Many Requests
                print(f"   Request {i+1}: Rate limited (429)")
                print("   ⚠ Server menerapkan rate limiting")
                return True
            else:
                print(f"   Request {i+1}: Error {response.status_code}")
            
            # Delay kecil antar request
            if i < 2:
                time.sleep(1)
        
        print("   ✓ Tidak ada rate limiting yang terdeteksi")
        return True
        
    except requests.RequestException as e:
        print(f"   ✗ Error saat testing rate limiting: {e}")
        return False

def main():
    """Fungsi utama untuk menjalankan semua test"""
    print("=" * 60)
    print("TEST KONEKSI DAN VALIDASI WEBSITE BPK")
    print("=" * 60)
    
    tests = [
        test_basic_connection,
        test_search_functionality,
        test_pdf_download,
        test_pagination,
        test_rate_limiting
    ]
    
    passed = 0
    total = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed += 1
            print()  # Baris kosong antar test
        except Exception as e:
            print(f"   ✗ Unexpected error: {e}")
            print()
    
    print("=" * 60)
    print(f"HASIL TEST: {passed}/{total} test berhasil")
    
    if passed == total:
        print("✓ Semua test berhasil! Website siap untuk di-scrape.")
    elif passed >= total * 0.8:  # 80% berhasil
        print("⚠ Sebagian besar test berhasil. Scraping mungkin dapat dilakukan dengan beberapa keterbatasan.")
    else:
        print("✗ Banyak test yang gagal. Periksa koneksi internet dan status website.")
    
    print("=" * 60)
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)