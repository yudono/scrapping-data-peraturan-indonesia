#!/usr/bin/env python3
"""
Script untuk menjalankan scraper dengan opsi yang lebih fleksibel
"""

import argparse
import sys
from scraper_peraturan_bpk import PeraturanScraper

def main():
    parser = argparse.ArgumentParser(
        description='Scraper Peraturan BPK Indonesia',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python run_scraper.py                    # Scrape semua tahun (1945-2025)
  python run_scraper.py --start 2020       # Scrape dari 2020 sampai 2025
  python run_scraper.py --start 2020 --end 2022  # Scrape tahun 2020-2022
  python run_scraper.py --year 2025        # Scrape hanya tahun 2025
  python run_scraper.py --dir my_downloads  # Simpan ke folder custom
        """
    )
    
    parser.add_argument(
        '--start', 
        type=int, 
        default=1945,
        help='Tahun mulai scraping (default: 1945)'
    )
    
    parser.add_argument(
        '--end', 
        type=int, 
        default=2025,
        help='Tahun akhir scraping (default: 2025)'
    )
    
    parser.add_argument(
        '--year', 
        type=int,
        help='Scrape hanya tahun tertentu (override --start dan --end)'
    )
    
    parser.add_argument(
        '--dir', 
        type=str, 
        default='downloads',
        help='Direktori untuk menyimpan file (default: downloads)'
    )
    
    args = parser.parse_args()
    
    # Validasi input
    if args.year:
        if args.year < 1945 or args.year > 2025:
            print("Error: Tahun harus antara 1945-2025")
            sys.exit(1)
        start_year = end_year = args.year
    else:
        if args.start < 1945 or args.start > 2025:
            print("Error: Tahun mulai harus antara 1945-2025")
            sys.exit(1)
        if args.end < 1945 or args.end > 2025:
            print("Error: Tahun akhir harus antara 1945-2025")
            sys.exit(1)
        if args.start > args.end:
            print("Error: Tahun mulai tidak boleh lebih besar dari tahun akhir")
            sys.exit(1)
        start_year = args.start
        end_year = args.end
    
    # Tampilkan informasi
    print("=" * 50)
    print("SCRAPER PERATURAN BPK INDONESIA")
    print("=" * 50)
    print(f"Tahun: {start_year} - {end_year}")
    print(f"Direktori output: {args.dir}")
    print("=" * 50)
    
    # Konfirmasi dari user
    try:
        confirm = input("Lanjutkan scraping? (y/n): ").lower().strip()
        if confirm not in ['y', 'yes', 'ya']:
            print("Scraping dibatalkan.")
            sys.exit(0)
    except KeyboardInterrupt:
        print("\nScraping dibatalkan.")
        sys.exit(0)
    
    # Mulai scraping
    try:
        scraper = PeraturanScraper(download_dir=args.dir)
        scraper.scrape_all_years(start_year, end_year)
    except KeyboardInterrupt:
        print("\nScraping dihentikan oleh user.")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()