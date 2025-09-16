#!/usr/bin/env python3
"""
Demo script untuk menunjukkan cara kerja scraper
dengan mengambil sample data dari tahun 2025
"""

import os
import sys
from scraper_peraturan_bpk import PeraturanScraper
import logging

def setup_demo_logging():
    """Setup logging khusus untuk demo"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def demo_connection_test():
    """Demo test koneksi"""
    print("=" * 50)
    print("DEMO: Test Koneksi Website BPK")
    print("=" * 50)
    
    try:
        import requests
        from bs4 import BeautifulSoup
        
        print("1. Testing basic connection...")
        response = requests.get("https://peraturan.bpk.go.id", timeout=10)
        if response.status_code == 200:
            print("   ✓ Website dapat diakses")
        else:
            print(f"   ✗ Error: Status code {response.status_code}")
            return False
        
        print("2. Testing search for year 2025...")
        search_url = "https://peraturan.bpk.go.id/Search?tahun=2025"
        response = requests.get(search_url, timeout=15)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Cek hasil pencarian
            result_text = response.text.lower()
            if "menemukan" in result_text and "peraturan" in result_text:
                print("   ✓ Pencarian berhasil")
                
                # Ambil jumlah hasil
                import re
                match = re.search(r'menemukan\s+([\d.,]+)\s+peraturan', result_text)
                if match:
                    count = match.group(1)
                    print(f"   ✓ Ditemukan {count} peraturan untuk tahun 2025")
                
                return True
            else:
                print("   ✗ Tidak ada hasil pencarian")
                return False
        else:
            print(f"   ✗ Error: Status code {response.status_code}")
            return False
            
    except Exception as e:
        print(f"   ✗ Error: {e}")
        return False

def demo_scraping_sample():
    """Demo scraping dengan sample kecil"""
    print("\n" + "=" * 50)
    print("DEMO: Scraping Sample Data (Tahun 2024, Halaman 1)")
    print("=" * 50)
    
    try:
        # Setup scraper dengan direktori demo
        demo_dir = "demo_downloads"
        scraper = PeraturanScraper(download_dir=demo_dir)
        
        print(f"Direktori demo: {demo_dir}")
        
        # Ambil halaman pertama tahun 2024
        print("Mengambil data halaman pertama tahun 2024...")
        html_content = scraper.get_search_results(2024, 1)
        
        if not html_content:
            print("✗ Gagal mengambil data")
            return False
        
        # Parse hasil
        results = scraper.parse_search_results(html_content)
        print(f"✓ Ditemukan {len(results)} item peraturan")
        
        if results:
            print("\nSample peraturan yang ditemukan:")
            print("-" * 40)
            
            for i, result in enumerate(results[:5], 1):  # Tampilkan 5 pertama
                title = result['title'][:80] + "..." if len(result['title']) > 80 else result['title']
                pdf_count = len(result['pdf_links'])
                print(f"{i}. {title}")
                print(f"   PDF links: {pdf_count}")
                
                # Tampilkan link PDF pertama
                if result['pdf_links']:
                    first_link = result['pdf_links'][0]
                    print(f"   URL: {first_link[:60]}...")
                print()
            
            if len(results) > 5:
                print(f"... dan {len(results) - 5} item lainnya")
        
        return True
        
    except Exception as e:
        print(f"✗ Error during demo scraping: {e}")
        return False

def demo_download_single_pdf():
    """Demo download satu file PDF"""
    print("\n" + "=" * 50)
    print("DEMO: Download Single PDF")
    print("=" * 50)
    
    try:
        demo_dir = "demo_downloads"
        scraper = PeraturanScraper(download_dir=demo_dir)
        
        # Ambil data untuk mencari PDF
        html_content = scraper.get_search_results(2024, 1)
        if not html_content:
            print("✗ Gagal mengambil data")
            return False
        
        results = scraper.parse_search_results(html_content)
        
        if not results:
            print("✗ Tidak ada hasil yang ditemukan")
            return False
        
        # Ambil PDF pertama
        first_result = results[0]
        if not first_result['pdf_links']:
            print("✗ Tidak ada link PDF yang ditemukan")
            return False
        
        title = first_result['title']
        pdf_url = first_result['pdf_links'][0]
        
        print(f"Mencoba download PDF:")
        print(f"Title: {title[:60]}...")
        print(f"URL: {pdf_url[:60]}...")
        
        # Buat nama file yang aman
        safe_title = scraper.sanitize_filename(title)
        filename = f"{safe_title}.pdf"
        
        print(f"Filename: {filename[:60]}...")
        print("Downloading...")
        
        success = scraper.download_file(pdf_url, filename)
        
        if success:
            print("✓ Download berhasil!")
            
            # Cek ukuran file
            filepath = os.path.join(demo_dir, "2024", filename)
            if os.path.exists(filepath):
                size = os.path.getsize(filepath)
                size_mb = size / (1024 * 1024)
                print(f"✓ File size: {size_mb:.2f} MB")
            
            return True
        else:
            print("✗ Download gagal")
            return False
            
    except Exception as e:
        print(f"✗ Error during PDF download: {e}")
        return False

def cleanup_demo():
    """Bersihkan file demo"""
    print("\n" + "=" * 50)
    print("CLEANUP: Menghapus file demo")
    print("=" * 50)
    
    try:
        import shutil
        demo_dir = "demo_downloads"
        
        if os.path.exists(demo_dir):
            shutil.rmtree(demo_dir)
            print(f"✓ Direktori {demo_dir} berhasil dihapus")
        else:
            print(f"✓ Direktori {demo_dir} tidak ada")
            
    except Exception as e:
        print(f"✗ Error during cleanup: {e}")

def main():
    """Fungsi utama demo"""
    setup_demo_logging()
    
    print("DEMO SCRAPER PERATURAN BPK INDONESIA")
    print("=====================================")
    print("Demo ini akan menunjukkan cara kerja scraper dengan:")
    print("1. Test koneksi ke website BPK")
    print("2. Scraping sample data (tahun 2025, halaman 1)")
    print("3. Download satu file PDF sebagai contoh")
    print("4. Cleanup file demo")
    print()
    
    try:
        # Konfirmasi dari user
        confirm = input("Lanjutkan demo? (y/n): ").lower().strip()
        if confirm not in ['y', 'yes', 'ya']:
            print("Demo dibatalkan.")
            return
        
        # Jalankan demo
        success_count = 0
        total_tests = 3
        
        if demo_connection_test():
            success_count += 1
        
        if demo_scraping_sample():
            success_count += 1
        
        if demo_download_single_pdf():
            success_count += 1
        
        # Cleanup
        cleanup_demo()
        
        # Summary
        print("\n" + "=" * 50)
        print("DEMO SUMMARY")
        print("=" * 50)
        print(f"Tests passed: {success_count}/{total_tests}")
        
        if success_count == total_tests:
            print("✓ Semua demo berhasil! Scraper siap digunakan.")
            print("\nUntuk menjalankan scraper lengkap:")
            print("  make run                    # Semua tahun")
            print("  make run-year YEAR=2025     # Tahun tertentu")
            print("  make run-range START=2020 END=2025  # Rentang tahun")
        else:
            print("⚠ Beberapa demo gagal. Periksa koneksi internet dan coba lagi.")
        
    except KeyboardInterrupt:
        print("\nDemo dibatalkan oleh user.")
        cleanup_demo()
    except Exception as e:
        print(f"\nError during demo: {e}")
        cleanup_demo()

if __name__ == "__main__":
    main()