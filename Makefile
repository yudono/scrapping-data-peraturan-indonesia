# Makefile untuk Scraper Peraturan BPK Indonesia

.PHONY: help install test run analyze list cleanup export clean

# Default target
help:
	@echo "Scraper Peraturan BPK Indonesia"
	@echo "==============================="
	@echo ""
	@echo "Available commands:"
	@echo "  make install    - Install dependencies"
	@echo "  make test       - Test connection to BPK website"
	@echo "  make demo       - Run interactive demo"
	@echo "  make run        - Run scraper for all years (1945-2025)"
	@echo "  make run-year   - Run scraper for specific year (use YEAR=xxxx)"
	@echo "  make run-range  - Run scraper for year range (use START=xxxx END=yyyy)"
	@echo "  make analyze    - Analyze downloaded files"
	@echo "  make list       - List all downloaded files by year"
	@echo "  make list-year  - List files for specific year (use YEAR=xxxx)"
	@echo "  make cleanup    - Remove empty directories"
	@echo "  make export     - Export metadata to CSV"
	@echo "  make extract-pdf - Extract all PDF files to text format"
	@echo "  make clean      - Clean all downloaded files and logs"
	@echo "  make config     - Show current configuration"
	@echo ""
	@echo "Quick start:"
	@echo "  make quick-start  # Install + test"
	@echo "  make demo         # Try interactive demo"
	@echo ""
	@echo "Examples:"
	@echo "  make run-year YEAR=2025"
	@echo "  make run-range START=2020 END=2025"
	@echo "  make list-year YEAR=2025"

install:
	@echo "Installing dependencies..."
	pip install -r requirements.txt
	@echo "Dependencies installed successfully!"

test:
	@echo "Testing connection to BPK website..."
	python test_connection.py

run:
	@echo "Running scraper for all years (1945-2025)..."
	@echo "This will take a very long time. Press Ctrl+C to cancel."
	@sleep 3
	python run_scraper.py

run-year:
	@if [ -z "$(YEAR)" ]; then \
		echo "Error: Please specify YEAR. Example: make run-year YEAR=2025"; \
		exit 1; \
	fi
	@echo "Running scraper for year $(YEAR)..."
	python run_scraper.py --year $(YEAR)

run-range:
	@if [ -z "$(START)" ] || [ -z "$(END)" ]; then \
		echo "Error: Please specify START and END. Example: make run-range START=2020 END=2025"; \
		exit 1; \
	fi
	@echo "Running scraper for years $(START)-$(END)..."
	python run_scraper.py --start $(START) --end $(END)

analyze:
	@echo "Analyzing downloaded files..."
	python utils.py analyze

list:
	@echo "Listing all downloaded files..."
	python utils.py list

list-year:
	@if [ -z "$(YEAR)" ]; then \
		echo "Error: Please specify YEAR. Example: make list-year YEAR=2025"; \
		exit 1; \
	fi
	@echo "Listing files for year $(YEAR)..."
	python utils.py list --year $(YEAR)

cleanup:
	@echo "Cleaning up empty directories..."
	python utils.py cleanup

export:
	@echo "Exporting metadata to CSV..."
	python utils.py export
	@echo "Metadata exported to metadata.csv"

clean:
	@echo "WARNING: This will delete all downloaded files and logs!"
	@echo "Press Ctrl+C within 5 seconds to cancel..."
	@sleep 5
	@echo "Cleaning up..."
	rm -rf downloads/
	rm -f scraper.log
	rm -f metadata.csv
	@echo "Cleanup completed!"

# PDF extraction commands
extract-pdf:
	@echo "Extracting PDF files to text format..."
	python extract_pdf_to_text.py

# Quick start commands
quick-test: install test
	@echo "Quick test completed!"

quick-start: install test
	@echo "Setup completed! You can now run 'make run' to start scraping."

# Demo commands
demo:
	@echo "Running interactive demo..."
	python demo.py

# Configuration commands
config:
	@echo "Showing current configuration..."
	python config.py

# Development commands
dev-test:
	@echo "Running development test (year 2025 only)..."
	python run_scraper.py --year 2025 --dir test_downloads

dev-clean:
	@echo "Cleaning development files..."
	rm -rf test_downloads/
	rm -rf demo_downloads/