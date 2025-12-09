"""
KB Document Extraction Pipeline - TEXT ONLY
Extracts text from PDFs for all customers with change detection
No OCR - Digital PDFs only (born-digital documents)
Optimized for voicebot RAG systems
"""

import json
import sys
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Add parent directory to import paths_config
sys.path.insert(0, str(Path(__file__).parent.parent))
from paths_config import PathsConfig

try:
    from dotenv import load_dotenv
    DOTENV_AVAILABLE = True
except ImportError:
    DOTENV_AVAILABLE = False
    print("[WARN] python-dotenv not installed")

import fitz  # PyMuPDF

# Configure logging - module level logger
logger = logging.getLogger(__name__)


class KBDocumentExtractor:
    """Extract text from PDF documents for KB ingestion (text-only, no OCR)"""

    def __init__(self, pdf_path, output_base_dir="kb_documents"):
        self.pdf_path = Path(pdf_path)
        self.doc = fitz.open(pdf_path)

        # Create output folder with PDF filename (no extension)
        pdf_name = self.pdf_path.stem
        self.output_dir = Path(output_base_dir) / pdf_name
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"Extracting text from: {pdf_path}")

    def extract_page(self, page_num):
        """Extract text from a single page"""
        page = self.doc[page_num]

        # Extract PDF text
        pdf_text = page.get_text().strip()

        # Get page metadata
        page_metadata = {
            'page_num': page_num + 1,
            'width': page.rect.width,
            'height': page.rect.height,
            'text_length': len(pdf_text),
            'has_text': bool(pdf_text)
        }

        return {
            'page_num': page_num + 1,
            'text': pdf_text,
            'metadata': page_metadata
        }

    def extract_all(self):
        """Extract all pages"""
        pages = []
        total_text_chars = 0
        pages_with_text = 0

        for page_num in range(len(self.doc)):
            try:
                page_data = self.extract_page(page_num)
                pages.append(page_data)

                text_length = len(page_data['text'])
                total_text_chars += text_length

                if text_length > 0:
                    pages_with_text += 1

            except Exception as e:
                logger.error(f"Error extracting page {page_num + 1}: {e}")
                continue

        return {
            'metadata': {
                'source_pdf': self.pdf_path.name,
                'total_pages': len(self.doc),
                'pages_with_text': pages_with_text,
                'total_text_chars': total_text_chars,
                'extraction_method': 'text_only',
                'extracted_at': datetime.now().isoformat()
            },
            'pages': pages
        }

    def save_results(self, extraction_result):
        """Save extraction results to JSON"""
        output_file = self.output_dir / 'content.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(extraction_result, f, indent=2, ensure_ascii=False)
        return output_file


class ExtractionPipeline:
    """Manage extraction for all customers"""

    def __init__(self, server_root: Path):
        self.server_root = Path(server_root)

        # Create logs directory FIRST
        logs_dir = self.server_root / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Configure logging if not already configured
        if not logging.getLogger().handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - EXTRACTION - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler(str(logs_dir / 'extraction.log'), encoding='utf-8'),
                    logging.StreamHandler()
                ]
            )

        logger.info(f"Extraction pipeline initialized with root: {self.server_root}")

    def get_pdf_metadata(self, pdf_path: Path) -> Dict:
        """Get PDF file metadata (size, modified time)"""
        stat = pdf_path.stat()
        return {
            'size': stat.st_size,
            'modified': stat.st_mtime,
            'filename': pdf_path.name
        }

    def load_extraction_metadata(self, customer_dir: Path) -> Dict:
        """Load extraction metadata for a customer"""
        metadata_path = customer_dir / "extraction_metadata.json"

        if metadata_path.exists():
            try:
                with open(metadata_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load extraction metadata: {e}")
                return {'extracted_pdfs': {}}

        return {'extracted_pdfs': {}}

    def save_extraction_metadata(self, customer_dir: Path, metadata: Dict):
        """Save extraction metadata for a customer"""
        metadata_path = customer_dir / "extraction_metadata.json"
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)

    def should_extract(self, pdf_path: Path, extracted_metadata: Dict) -> tuple:
        """
        Check if PDF should be extracted
        Returns: (should_extract, reason)
        """
        pdf_name = pdf_path.name
        current_metadata = self.get_pdf_metadata(pdf_path)

        # Check if already extracted
        if pdf_name not in extracted_metadata.get('extracted_pdfs', {}):
            return True, "New PDF"

        # Check if file has changed
        existing_meta = extracted_metadata['extracted_pdfs'][pdf_name]
        if current_metadata['size'] != existing_meta.get('size'):
            return True, "File size changed"

        if current_metadata['modified'] != existing_meta.get('modified'):
            return True, "File modified"

        return False, "Already extracted"

    def extract_customer(self, customer_name: str, customer_id: str) -> Dict:
        """Extract all PDFs for one customer"""
        customer_dir = self.server_root / "customers" / customer_id
        pdfs_dir = customer_dir / "pdfs"

        if not pdfs_dir.exists():
            logger.warning(f"No PDFs folder for {customer_name}")
            return {
                'customer_name': customer_name,
                'customer_id': customer_id,
                'status': 'no_pdfs',
                'total_pdfs': 0,
                'extracted': 0,
                'skipped': 0,
                'failed': 0
            }

        # Load existing extraction metadata
        extraction_meta = self.load_extraction_metadata(customer_dir)

        extracted_count = 0
        skipped_count = 0
        failed_count = 0
        pdf_list = list(pdfs_dir.glob("*.pdf"))

        print(f"\n[EXTRACT] Starting extraction for customer: {customer_name}")
        print("=" * 70)

        for pdf_path in sorted(pdf_list):
            should_extract, reason = self.should_extract(pdf_path, extraction_meta)

            if not should_extract:
                print(f"  [SKIP] {pdf_path.name} ({reason})")
                skipped_count += 1
                continue

            try:
                print(f"  [{reason}] {pdf_path.name}")

                extractor = KBDocumentExtractor(str(pdf_path), str(customer_dir))
                result = extractor.extract_all()

                # Add customer_id to metadata
                result['metadata']['customer_id'] = customer_id

                output_file = extractor.save_results(result)

                # Update extraction metadata
                pdf_meta = self.get_pdf_metadata(pdf_path)
                pdf_meta['extracted_at'] = datetime.now().isoformat()
                pdf_meta['pages'] = result['metadata']['total_pages']
                pdf_meta['pages_with_text'] = result['metadata']['pages_with_text']
                pdf_meta['text_chars'] = result['metadata']['total_text_chars']

                extraction_meta['extracted_pdfs'][pdf_path.name] = pdf_meta

                print(f"    [OK] Extracted {result['metadata']['total_pages']} pages, "
                      f"{result['metadata']['pages_with_text']} with text")

                extracted_count += 1

            except Exception as e:
                logger.error(f"Failed to extract {pdf_path.name}: {e}")
                print(f"    [ERROR] {e}")
                failed_count += 1

        # Save updated metadata
        self.save_extraction_metadata(customer_dir, extraction_meta)

        print("=" * 70)
        print(f"[SUMMARY] {customer_name}")
        print(f"  Extracted: {extracted_count}")
        print(f"  Skipped: {skipped_count}")
        print(f"  Failed: {failed_count}")
        print(f"  Total: {len(pdf_list)}")

        return {
            'customer_name': customer_name,
            'customer_id': customer_id,
            'status': 'success' if failed_count == 0 else 'partial',
            'total_pdfs': len(pdf_list),
            'extracted': extracted_count,
            'skipped': skipped_count,
            'failed': failed_count
        }

    def extract_all_customers(self, customer_filter: Optional[List[str]] = None) -> Dict:
        """Extract PDFs for all customers"""
        print("\n" + "=" * 70)
        print("KB DOCUMENT EXTRACTION PIPELINE (TEXT-ONLY)")
        print("=" * 70)

        customers_dir = self.server_root / "customers"

        if not customers_dir.exists():
            logger.error(f"Customers directory not found: {customers_dir}")
            return {}

        # Discover all customers
        all_customers = []
        for customer_folder in sorted(customers_dir.glob("*")):
            if customer_folder.is_dir():
                pdfs_dir = customer_folder / "pdfs"
                if pdfs_dir.exists() and any(pdfs_dir.glob("*.pdf")):
                    # Convert folder name to customer name (reverse of downloader logic)
                    customer_name = customer_folder.name.replace("_", " ").title()
                    all_customers.append((customer_name, customer_folder.name))

        print(f"\n[DISCOVER] Found {len(all_customers)} customers with PDFs")
        for customer_name, customer_id in all_customers:
            print(f"  - {customer_name}")

        # Filter if specified
        if customer_filter:
            all_customers = [(name, cid) for name, cid in all_customers
                            if name in customer_filter or cid in customer_filter]
            print(f"\n[FILTER] Processing {len(all_customers)} selected customers")

        # Extract for each customer
        results = {}
        for customer_name, customer_id in all_customers:
            result = self.extract_customer(customer_name, customer_id)
            results[customer_name] = result

        # Print summary
        self._print_summary(results)

        return results

    def _print_summary(self, results: Dict):
        """Print final extraction summary"""
        print("\n" + "=" * 70)
        print("EXTRACTION SUMMARY")
        print("=" * 70)

        total_pdfs = sum(r['total_pdfs'] for r in results.values())
        total_extracted = sum(r['extracted'] for r in results.values())
        total_skipped = sum(r['skipped'] for r in results.values())
        total_failed = sum(r['failed'] for r in results.values())

        for customer_name, result in sorted(results.items()):
            status_icon = "[OK]" if result['status'] == 'success' else "[WARN]"
            print(f"{status_icon} {customer_name:25} {result['extracted']}/{result['total_pdfs']} extracted")

        print("\n" + "-" * 70)
        print(f"Total PDFs: {total_pdfs}")
        print(f"Extracted: {total_extracted}")
        print(f"Skipped: {total_skipped}")
        print(f"Failed: {total_failed}")
        print("=" * 70)


def main():
    if DOTENV_AVAILABLE:
        load_dotenv()

    # Get configuration
    server_root = PathsConfig.RAG_DIR

    try:
        pipeline = ExtractionPipeline(Path(server_root))

        # Parse command line arguments
        import argparse
        parser = argparse.ArgumentParser(description='KB Document Extraction Pipeline (Text-Only)')
        parser.add_argument('--customers', nargs='+', help='Specific customers to process')
        args = parser.parse_args()

        results = pipeline.extract_all_customers(customer_filter=args.customers)

        # Exit with status
        failed = sum(1 for r in results.values() if r['status'] not in ['success', 'no_pdfs'])
        sys.exit(1 if failed > 0 else 0)

    except Exception as e:
        logger.error(f"Extraction pipeline failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
