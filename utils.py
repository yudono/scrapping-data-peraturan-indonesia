#!/usr/bin/env python3
"""
Utilitas untuk mengelola dan menganalisis hasil scraping
"""

import os
import json
import argparse
from collections import defaultdict
from datetime import datetime

def analyze_downloads(download_dir="downloads"):
    """Analisis file yang sudah didownload"""
    metadata_file = os.path.join(download_dir, "metadata.json")
    
    if not os.path.exists(metadata_file):
        print(f"File metadata tidak ditemukan: {metadata_file}")
        return
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    # Statistik per tahun
    stats_per_year = defaultdict(int)
    total_files = len(metadata)
    
    for url, info in metadata.items():
        tahun = info.get('tahun', 'Unknown')
        stats_per_year[tahun] += 1
    
    print("=" * 60)
    print("ANALISIS HASIL SCRAPING")
    print("=" * 60)
    print(f"Total file didownload: {total_files}")
    print(f"Direktori: {download_dir}")
    print("\nStatistik per tahun:")
    print("-" * 30)
    
    for tahun in sorted(stats_per_year.keys()):
        if str(tahun).isdigit():
            count = stats_per_year[tahun]
            print(f"{tahun}: {count} file")
    
    # Cek file fisik
    print("\nVerifikasi file fisik:")
    print("-" * 30)
    
    missing_files = []
    for url, info in metadata.items():
        tahun = info.get('tahun')
        filename = info.get('filename')
        if tahun and filename:
            filepath = os.path.join(download_dir, str(tahun), filename)
            if not os.path.exists(filepath):
                missing_files.append(filepath)
    
    if missing_files:
        print(f"File hilang: {len(missing_files)}")
        print("File yang hilang:")
        for file in missing_files[:10]:  # Tampilkan 10 pertama
            print(f"  - {file}")
        if len(missing_files) > 10:
            print(f"  ... dan {len(missing_files) - 10} file lainnya")
    else:
        print("Semua file metadata tersedia secara fisik ✓")

def list_files_by_year(download_dir="downloads", tahun=None):
    """List file berdasarkan tahun"""
    if tahun:
        year_dir = os.path.join(download_dir, str(tahun))
        if not os.path.exists(year_dir):
            print(f"Direktori tahun {tahun} tidak ditemukan: {year_dir}")
            return
        
        files = [f for f in os.listdir(year_dir) if f.endswith('.pdf')]
        print(f"File PDF untuk tahun {tahun}: {len(files)}")
        print("-" * 40)
        
        for i, file in enumerate(sorted(files), 1):
            filepath = os.path.join(year_dir, file)
            size = os.path.getsize(filepath)
            size_mb = size / (1024 * 1024)
            print(f"{i:3d}. {file} ({size_mb:.1f} MB)")
    else:
        # List semua tahun
        if not os.path.exists(download_dir):
            print(f"Direktori download tidak ditemukan: {download_dir}")
            return
        
        year_dirs = [d for d in os.listdir(download_dir) 
                    if os.path.isdir(os.path.join(download_dir, d)) and d.isdigit()]
        
        print("Tahun yang tersedia:")
        print("-" * 30)
        
        total_files = 0
        total_size = 0
        
        for year in sorted(year_dirs):
            year_path = os.path.join(download_dir, year)
            files = [f for f in os.listdir(year_path) if f.endswith('.pdf')]
            
            year_size = 0
            for file in files:
                filepath = os.path.join(year_path, file)
                year_size += os.path.getsize(filepath)
            
            year_size_mb = year_size / (1024 * 1024)
            total_files += len(files)
            total_size += year_size
            
            print(f"{year}: {len(files)} file ({year_size_mb:.1f} MB)")
        
        total_size_gb = total_size / (1024 * 1024 * 1024)
        print("-" * 30)
        print(f"Total: {total_files} file ({total_size_gb:.2f} GB)")

def cleanup_empty_dirs(download_dir="downloads"):
    """Hapus direktori kosong"""
    if not os.path.exists(download_dir):
        print(f"Direktori download tidak ditemukan: {download_dir}")
        return
    
    removed_dirs = []
    
    for item in os.listdir(download_dir):
        item_path = os.path.join(download_dir, item)
        if os.path.isdir(item_path) and item.isdigit():
            # Cek apakah direktori kosong
            files = os.listdir(item_path)
            if not files:
                os.rmdir(item_path)
                removed_dirs.append(item)
    
    if removed_dirs:
        print(f"Direktori kosong yang dihapus: {len(removed_dirs)}")
        for dir_name in removed_dirs:
            print(f"  - {dir_name}")
    else:
        print("Tidak ada direktori kosong yang ditemukan.")

def export_metadata_csv(download_dir="downloads", output_file="metadata.csv"):
    """Export metadata ke file CSV"""
    metadata_file = os.path.join(download_dir, "metadata.json")
    
    if not os.path.exists(metadata_file):
        print(f"File metadata tidak ditemukan: {metadata_file}")
        return
    
    with open(metadata_file, 'r', encoding='utf-8') as f:
        metadata = json.load(f)
    
    import csv
    
    with open(output_file, 'w', newline='', encoding='utf-8') as csvfile:
        fieldnames = ['url', 'title', 'filename', 'tahun', 'download_date']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        writer.writeheader()
        for url, info in metadata.items():
            row = {
                'url': url,
                'title': info.get('title', ''),
                'filename': info.get('filename', ''),
                'tahun': info.get('tahun', ''),
                'download_date': info.get('download_date', '')
            }
            writer.writerow(row)
    
    print(f"Metadata berhasil diekspor ke: {output_file}")
    print(f"Total record: {len(metadata)}")

def main():
    parser = argparse.ArgumentParser(
        description='Utilitas untuk mengelola hasil scraping',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Contoh penggunaan:
  python utils.py analyze                    # Analisis hasil download
  python utils.py list                       # List semua tahun
  python utils.py list --year 2025           # List file tahun 2025
  python utils.py cleanup                    # Hapus direktori kosong
  python utils.py export                     # Export metadata ke CSV
        """
    )
    
    parser.add_argument(
        'command',
        choices=['analyze', 'list', 'cleanup', 'export'],
        help='Perintah yang akan dijalankan'
    )
    
    parser.add_argument(
        '--dir',
        default='downloads',
        help='Direktori download (default: downloads)'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        help='Tahun tertentu untuk command list'
    )
    
    parser.add_argument(
        '--output',
        default='metadata.csv',
        help='File output untuk command export (default: metadata.csv)'
    )
    
    args = parser.parse_args()
    
    if args.command == 'analyze':
        analyze_downloads(args.dir)
    elif args.command == 'list':
        list_files_by_year(args.dir, args.year)
    elif args.command == 'cleanup':
        cleanup_empty_dirs(args.dir)
    elif args.command == 'export':
        export_metadata_csv(args.dir, args.output)

if __name__ == "__main__":
    main()