#!/usr/bin/env python3

import os
from pathlib import Path
from PyPDF2 import PdfReader

INPUT_DIR = "downloads"
OUTPUT_DIR = "dataset_txt"
DATASET_FILE = "dataset_law.txt"

def pdf_to_text(pdf_path, txt_path):
    """Ekstrak teks dari file PDF dan simpan ke file txt"""
    try:
        reader = PdfReader(pdf_path)
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        
        # Buat direktori jika belum ada
        os.makedirs(os.path.dirname(txt_path), exist_ok=True)
        
        with open(txt_path, "w", encoding="utf-8") as f:
            f.write(text)
        
        print(f"✓ Berhasil ekstrak: {pdf_path} -> {txt_path}")
        return True
        
    except Exception as e:
        print(f"✗ Error {pdf_path}: {e}")
        return False

def extract_all_pdfs():
    """Ekstrak semua file PDF dari folder downloads ke dataset_txt"""
    print(f"Memulai ekstraksi PDF dari folder '{INPUT_DIR}'...")
    
    # Buat direktori output
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    total_files = 0
    success_count = 0
    
    # Iterasi semua file PDF
    for root, _, files in os.walk(INPUT_DIR):
        for file in files:
            if file.endswith(".pdf"):
                total_files += 1
                pdf_path = os.path.join(root, file)
                
                # Buat path relatif untuk struktur folder yang sama
                rel_path = os.path.relpath(pdf_path, INPUT_DIR)
                txt_path = os.path.join(OUTPUT_DIR, Path(rel_path).with_suffix(".txt"))
                
                if pdf_to_text(pdf_path, txt_path):
                    success_count += 1
    
    print(f"\nSelesai! {success_count}/{total_files} file berhasil diekstrak.")
    return success_count

def combine_all_texts():
    """Gabungkan semua file txt menjadi satu dataset"""
    print(f"\nMenggabungkan semua file txt ke '{DATASET_FILE}'...")
    
    file_count = 0
    
    with open(DATASET_FILE, "w", encoding="utf-8") as outfile:
        for root, _, files in os.walk(OUTPUT_DIR):
            for file in files:
                if file.endswith(".txt"):
                    file_path = os.path.join(root, file)
                    try:
                        with open(file_path, "r", encoding="utf-8") as infile:
                            text = infile.read().strip()
                            if text:
                                # Tambahkan header dengan nama file
                                outfile.write(f"=== {file} ===\n")
                                outfile.write(text + "\n\n")
                                file_count += 1
                    except Exception as e:
                        print(f"✗ Error membaca {file_path}: {e}")
    
    print(f"✓ {file_count} file berhasil digabungkan ke '{DATASET_FILE}'")
    return file_count

def main():
    """Fungsi utama untuk ekstraksi PDF"""
    print("=" * 50)
    print("EKSTRAKSI PDF KE TEKS")
    print("=" * 50)
    
    # Cek apakah folder downloads ada
    if not os.path.exists(INPUT_DIR):
        print(f"✗ Folder '{INPUT_DIR}' tidak ditemukan!")
        print("Pastikan Anda sudah menjalankan scraper terlebih dahulu.")
        return
    
    # Hitung jumlah PDF
    pdf_count = sum(1 for root, _, files in os.walk(INPUT_DIR) 
                   for file in files if file.endswith(".pdf"))
    
    if pdf_count == 0:
        print(f"✗ Tidak ada file PDF di folder '{INPUT_DIR}'")
        return
    
    print(f"Ditemukan {pdf_count} file PDF untuk diekstrak.\n")
    
    # Ekstrak semua PDF
    success_count = extract_all_pdfs()
    
    if success_count > 0:
        # Gabungkan semua teks
        combine_all_texts()
        
        print("\n" + "=" * 50)
        print("EKSTRAKSI SELESAI!")
        print("=" * 50)
        print(f"📁 File teks individual: folder '{OUTPUT_DIR}'")
        print(f"📄 Dataset gabungan: '{DATASET_FILE}'")
    else:
        print("\n✗ Tidak ada file yang berhasil diekstrak.")

if __name__ == "__main__":
    main()