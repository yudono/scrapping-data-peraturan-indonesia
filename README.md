# Scraper Peraturan BPK Indonesia

Script Python untuk mengunduh data perundang-undangan Indonesia dari website BPK (https://peraturan.bpk.go.id/) dari tahun 1945 sampai 2025.

## Fitur

- Scraping otomatis semua peraturan dari tahun 1945-2025
- Download PDF peraturan secara otomatis
- Penanganan pagination untuk mengambil semua data
- Logging untuk monitoring proses
- Resume capability (melanjutkan download yang terputus)
- Metadata tracking untuk file yang sudah didownload

## Instalasi

1. Clone atau download repository ini
2. Install dependencies:

```bash
pip install -r requirements.txt
```

## Penggunaan

### Quick Start (Menggunakan Makefile)

```bash
# 1. Install dependencies dan test koneksi
make quick-start

# 2. Coba demo interaktif (opsional)
make demo

# 3. Jalankan scraper untuk semua tahun (1945-2025)
make run

# Atau jalankan untuk tahun tertentu
make run-year YEAR=2025

# Atau jalankan untuk rentang tahun
make run-range START=2020 END=2025
```

### Demo Interaktif

Sebelum menjalankan scraper lengkap, Anda dapat mencoba demo untuk memahami cara kerja scraper:

```bash
make demo
```

Demo akan:

1. Test koneksi ke website BPK
2. Scraping sample data (tahun 2025, halaman 1)
3. Download satu file PDF sebagai contoh
4. Cleanup otomatis file demo

### Menjalankan Scraper Manual

```bash
# Install dependencies
pip install -r requirements.txt

# Test koneksi website
python test_connection.py

# Jalankan scraper
python scraper_peraturan_bpk.py
# atau dengan opsi
python run_scraper.py --start 2020 --end 2025
```

### Perintah Makefile Lengkap

```bash
make help           # Tampilkan bantuan
make install        # Install dependencies
make test           # Test koneksi ke website BPK
make run            # Scrape semua tahun (1945-2025)
make run-year       # Scrape tahun tertentu (YEAR=xxxx)
make run-range      # Scrape rentang tahun (START=xxxx END=yyyy)
make analyze        # Analisis file yang sudah didownload
make list           # List semua file berdasarkan tahun
make list-year      # List file untuk tahun tertentu (YEAR=xxxx)
make cleanup        # Hapus direktori kosong
make export         # Export metadata ke CSV
make extract-pdf    # Ekstrak semua PDF ke format teks
make clean          # Hapus semua file download dan log
```

Script akan:

1. Membuat folder `downloads` untuk menyimpan file PDF
2. Membuat subfolder untuk setiap tahun
3. Mengunduh semua PDF peraturan yang tersedia
4. Menyimpan metadata dalam file `metadata.json`
5. Membuat log dalam file `scraper.log`

### Struktur Folder Output

```
downloads/
├── metadata.json          # Metadata file yang sudah didownload
├── scraper.log           # Log file
├── 1945/                 # Folder untuk tahun 1945
│   ├── peraturan1.pdf
│   └── peraturan2.pdf
├── 1946/                 # Folder untuk tahun 1946
│   └── ...
└── 2025/                 # Folder untuk tahun 2025
    └── ...
```

## Ekstraksi PDF ke Teks

Setelah mendownload file PDF, Anda dapat mengekstrak konten teks untuk analisis lebih lanjut:

### Menggunakan Makefile (Recommended)

```bash
# Ekstrak semua PDF ke format teks
make extract-pdf
```

### Menjalankan Script Manual

```bash
# Install dependency tambahan (jika belum)
pip install "PyPDF2>=3.0.0"

# Jalankan ekstraksi
python extract_pdf_to_text.py
```

### Output Ekstraksi

Script akan membuat:

1. **Folder `dataset_txt/`**: Berisi file teks individual untuk setiap PDF

   ```
   dataset_txt/
   ├── 1945/
   │   ├── peraturan1.txt
   │   └── peraturan2.txt
   ├── 1946/
   │   └── ...
   └── 2025/
       └── ...
   ```

2. **File `dataset_law.txt`**: Dataset gabungan semua teks peraturan
   - Format: Setiap peraturan dipisahkan dengan header nama file
   - Encoding: UTF-8
   - Siap untuk analisis teks, machine learning, atau penelitian

### Fitur Ekstraksi

- ✅ **Batch Processing**: Ekstrak semua PDF sekaligus
- ✅ **Error Handling**: Skip file yang corrupt/tidak bisa dibaca
- ✅ **Progress Tracking**: Menampilkan progress ekstraksi
- ✅ **Struktur Folder**: Mempertahankan struktur folder asli
- ✅ **Dataset Gabungan**: Otomatis menggabungkan semua teks
- ✅ **UTF-8 Encoding**: Mendukung karakter Indonesia dengan baik

### Contoh Penggunaan Dataset

```python
# Baca dataset gabungan
with open('dataset_law.txt', 'r', encoding='utf-8') as f:
    all_laws = f.read()

# Analisis teks, word frequency, dll
print(f"Total karakter: {len(all_laws)}")
print(f"Total kata: {len(all_laws.split())}")
```

## Konfigurasi

### File Konfigurasi (config.py)

Scraper menggunakan file `config.py` untuk pengaturan. Anda dapat menyesuaikan:

- **Timeout dan Retry**: `REQUEST_TIMEOUT`, `DOWNLOAD_TIMEOUT`, `MAX_RETRIES`
- **Rate Limiting**: `DELAY_BETWEEN_REQUESTS`, `DELAY_BETWEEN_PAGES`, `DELAY_BETWEEN_YEARS`
- **Direktori Download**: `DEFAULT_DOWNLOAD_DIR`
- **Logging**: `LOG_LEVEL`, `LOG_FILE`
- **User Agent**: `USER_AGENT`
- **Rentang Tahun**: `DEFAULT_START_YEAR`, `DEFAULT_END_YEAR`

### Environment Variables

Anda juga dapat menggunakan environment variables:

```bash
export SCRAPER_DOWNLOAD_DIR="/path/to/downloads"
export SCRAPER_REQUEST_TIMEOUT="60"
export SCRAPER_DELAY_REQUESTS="2"
export SCRAPER_LOG_LEVEL="DEBUG"
```

### Melihat Konfigurasi Saat Ini

```bash
python config.py
```

## Kustomisasi

### Mengubah Rentang Tahun

Edit fungsi `main()` dalam script:

```python
def main():
    scraper = PeraturanBPKScraper()

    # Ubah rentang tahun sesuai kebutuhan
    scraper.scrape_all_years(2020, 2025)  # Contoh: hanya 2020-2025
```

### Mengubah Direktori Download

```python
scraper = PeraturanBPKScraper(download_dir="folder_custom")
```

## Fitur Keamanan

- **Rate Limiting**: Delay otomatis antar request untuk menghindari overload server
- **Resume Capability**: Script dapat melanjutkan download yang terputus
- **Error Handling**: Penanganan error yang robust
- **Logging**: Log lengkap untuk monitoring dan debugging

## Catatan Penting

1. **Waktu Eksekusi**: Proses scraping membutuhkan waktu yang lama (bisa beberapa jam/hari) karena:

   - Jumlah data yang sangat besar (1945-2025)
   - Rate limiting untuk menghormati server
   - Ukuran file PDF yang bervariasi

2. **Ruang Disk**: Pastikan memiliki ruang disk yang cukup (bisa mencapai beberapa GB)

3. **Koneksi Internet**: Pastikan koneksi internet stabil

4. **Etika Scraping**: Script sudah mengimplementasikan delay yang wajar untuk tidak membebani server

## File dan Utilitas

### File Utama

- `scraper_peraturan_bpk.py` - Script scraper utama
- `run_scraper.py` - Script dengan opsi command line yang fleksibel
- `utils.py` - Utilitas untuk analisis dan manajemen file
- `test_connection.py` - Script untuk test koneksi website
- `demo.py` - Demo interaktif untuk memahami cara kerja scraper
- `config.py` - File konfigurasi untuk pengaturan scraper
- `Makefile` - Automation commands untuk kemudahan penggunaan
- `requirements.txt` - Dependencies Python yang diperlukan

### Utilitas Tambahan

```bash
# Analisis file yang sudah didownload
python utils.py analyze

# List file berdasarkan tahun
python utils.py list
python utils.py list --year 2025

# Hapus direktori kosong
python utils.py cleanup

# Export metadata ke CSV
python utils.py export --output my_metadata.csv
```

## Troubleshooting

### Error "Connection timeout"

- Periksa koneksi internet
- Jalankan `make test` untuk test koneksi
- Coba jalankan ulang script (akan melanjutkan dari yang terakhir)

### Error "Permission denied"

- Pastikan memiliki permission write di direktori
- Jalankan dengan `sudo` jika diperlukan (Linux/Mac)

### Script berhenti tiba-tiba

- Cek file `scraper.log` untuk detail error
- Jalankan `make analyze` untuk cek status download
- Jalankan ulang script (akan melanjutkan otomatis)

### Website tidak dapat diakses

- Jalankan `make test` untuk diagnosa
- Cek apakah website BPK sedang maintenance
- Coba lagi beberapa saat kemudian

## Kontribusi

Silakan buat issue atau pull request untuk perbaikan dan penambahan fitur.

## Disclaimer

Script ini dibuat untuk tujuan penelitian dan edukasi. Pastikan untuk mematuhi terms of service website BPK dan menggunakan data yang diunduh sesuai dengan ketentuan yang berlaku.

ini command untuk training:
bash -lc 'python fine_tune_gpt2_lora.py --dataset_dir dataset_txt/1945 --output_dir outputs/hukum_1945 --model_name distilgpt2 --epochs 1 --max_train_steps 150 --batch_size 1 --grad_accum 8 --block_size 128 --eval_steps 50 --save_steps 50 --logging_steps 10 --save_total_limit 2 --sample_after_train --sample_prompt "Undang-undang"'

python fine_tune_gpt2_lora.py --dataset_dir dataset_txt/1945 --output_dir outputs/hukum_1945_rerun --model_name distilgpt2 --epochs 1 --max_train_steps 30 --batch_size 1 --grad_accum 8 --block_size 128 --eval_steps 15 --save_steps 1000 --logging_steps 5 --save_total_limit 2 --sample_after_train --sample_prompt "Undang-undang"

python fine_tune_gpt2_lora.py --dataset_dir dataset_txt/1945 --output_dir outputs/hukum_1945_gpt2 --model_name gpt2 --epochs 1 --max_train_steps 180 --batch_size 1 --grad_accum 8 --block_size 256 --eval_steps 30 --save_steps 60 --logging_steps 10 --save_total_limit 3 --sample_after_train --sample_prompt "Undang-undang tentang peraturan perpajakan"

python fine_tune_gpt2_lora.py --dataset_path dataset_law.txt --epochs 1 --batch_size 1 --max_train_steps 10 --save_steps 10 --eval_steps 5 --logging_steps 2

python fine_tune_gpt2_lora.py --dataset_path dataset_law.txt --epochs 2 --batch_size 2

ini buat runing:
bash -lc 'python run_hukum_1945.py --prompt "Undang-undang tentang peraturan perpajakan" --adapter_dir outputs/hukum_1945 --base_model distilgpt2 --max_new_tokens 80 --top_p 0.95 --temperature 0.8'

bash -lc 'python run_hukum_1945.py --prompt "Undang-undang tentang peraturan perpajakan" --adapter_dir outputs/hukum_1945_rerun --base_model distilgpt2 --max_new_tokens 80 --top_p 0.95 --temperature 0.8'

bash -lc 'python run_hukum_1945.py --prompt "Undang-undang tentang peraturan perpajakan" --adapter_dir outputs/hukum_1945_gpt2 --base_model distilgpt2 --max_new_tokens 80 --top_p 0.95 --temperature 0.8'

python run_law_model.py \
 --prompt "Undang-undang tentang peraturan perpajakan" \
 --adapter_dir outputs/gpt2-lora-law \
 --base_model gpt2 \
 --max_new_tokens 200 \
 --top_p 0.95 \
 --temperature 0.8

# Basic full fine-tuning

python fine_tune_gpt2_full.py --dataset_path dataset_law.txt --epochs 2 --batch_size 1

# Dengan parameter lebih detail

python fine_tune_gpt2_full.py \
 --dataset_path dataset_law.txt \
 --output_dir outputs/gpt2-full-law \
 --epochs 3 --batch_size 2 --grad_accum 4 \
 --block_size 256 --save_steps 500

# Untuk testing (lebih cepat)

python fine_tune_gpt2_full.py \
 --dataset_path dataset_law.txt \
 --epochs 1 --batch_size 1 --limit_chars 50000

# Single generation

python run_full_law_model.py \
 --model_dir outputs/gpt2-full-law \
 --prompt "Undang-undang tentang" --max_tokens 200

# Interactive mode

python run_full_law_model.py \
 --model_dir outputs/gpt2-full-law --interactive

# Batch generation

python run_full_law_model.py \
 --model_dir outputs/gpt2-full-law \
 --prompts "Pasal 1" "Ayat (1)" "Ketentuan umum"
