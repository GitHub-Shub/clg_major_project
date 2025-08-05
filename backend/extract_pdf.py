# backend/extract_pdf.py
"""
Enhanced Multi-PDF Text Extractor for Legal Documents
====================================================

This script extracts text from multiple PDF files and combines them into a single output file.
It's designed to be beginner-friendly while providing the best possible
text extraction results for RAG applications.

Features:
- Process multiple PDFs from a folder
- Detailed progress tracking for each file
- Smart text cleaning and formatting
- Table extraction support
- Error handling with helpful messages
- Quality validation
- Multiple extraction strategies
- Document separation markers
"""

import pdfplumber
import os
import sys
from pathlib import Path
import re
import glob
from datetime import datetime


class MultiPDFTextExtractor:
    """
    A comprehensive multi-PDF text extractor that prioritizes data quality and clarity.
    
    This class handles PDF text extraction from multiple files with strategies to ensure
    we get the highest quality text output for our RAG system.
    """
    
    def __init__(self):
        """Initialize the PDF extractor with default settings."""
        self.total_files = 0
        self.processed_files = 0
        self.total_pages = 0
        self.processed_pages = 0
        self.combined_text = ""
        self.extraction_stats = {
            'total_characters': 0,
            'total_words': 0,
            'total_files': 0,
            'successful_files': 0,
            'failed_files': 0,
            'empty_pages': 0,
            'pages_with_tables': 0,
            'successful_pages': 0,
            'file_details': []
        }
    
    def find_pdf_files(self, pdf_folder):
        """
        Find all PDF files in the specified folder.
        
        Args:
            pdf_folder (str): Path to the folder containing PDF files
            
        Returns:
            list: List of PDF file paths
        """
        if not os.path.exists(pdf_folder):
            print(f"❌ Error: PDF folder not found at '{pdf_folder}'")
            return []
        
        # Find all PDF files
        pdf_pattern = os.path.join(pdf_folder, "*.pdf")
        pdf_files = glob.glob(pdf_pattern, recursive=False)
        
        # Also check for case variations
        pdf_pattern_upper = os.path.join(pdf_folder, "*.PDF")
        pdf_files.extend(glob.glob(pdf_pattern_upper, recursive=False))
        
        # Remove duplicates and sort
        pdf_files = sorted(list(set(pdf_files)))
        
        if not pdf_files:
            print(f"❌ Error: No PDF files found in '{pdf_folder}'")
            print("   Please ensure PDF files are present in the folder.")
            return []
        
        print(f"✅ Found {len(pdf_files)} PDF files:")
        for i, pdf_file in enumerate(pdf_files, 1):
            file_name = os.path.basename(pdf_file)
            file_size = os.path.getsize(pdf_file)
            print(f"   {i}. {file_name} ({file_size:,} bytes)")
        
        return pdf_files
    
    def validate_pdf_file(self, pdf_path):
        """
        Check if the PDF file exists and is accessible.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            bool: True if file is valid, False otherwise
        """
        pdf_file = Path(pdf_path)
        
        if not pdf_file.exists():
            print(f"❌ Error: PDF file not found at '{pdf_path}'")
            return False
        
        if not pdf_file.suffix.lower() == '.pdf':
            print(f"❌ Error: '{pdf_path}' is not a PDF file")
            return False
        
        if pdf_file.stat().st_size == 0:
            print(f"❌ Error: PDF file '{pdf_path}' is empty")
            return False
        
        return True
    
    def extract_text_from_page(self, page, page_num, total_pages_in_file):
        """
        Extract text from a single PDF page with multiple strategies.
        
        Args:
            page: pdfplumber page object
            page_num (int): Page number for logging
            total_pages_in_file (int): Total pages in current file
            
        Returns:
            str: Extracted text from the page
        """
        print(f"📄 Processing page {page_num}/{total_pages_in_file}...", end=" ")
        
        page_text = ""
        
        try:
            # Strategy 1: Extract regular text
            regular_text = page.extract_text()
            if regular_text:
                page_text += regular_text
                page_text += "\n"
            
            # Strategy 2: Extract tables if present
            tables = page.extract_tables()
            if tables:
                self.extraction_stats['pages_with_tables'] += 1
                print("(contains tables)", end=" ")
                
                for table in tables:
                    # Convert table to readable text format
                    table_text = self.format_table_as_text(table)
                    page_text += table_text + "\n"
            
            # Check if page had content
            if page_text.strip():
                self.extraction_stats['successful_pages'] += 1
                print("✅")
            else:
                self.extraction_stats['empty_pages'] += 1
                print("⚠️ (empty)")
                
        except Exception as e:
            print(f"❌ (error: {str(e)})")
            
        return page_text
    
    def format_table_as_text(self, table):
        """
        Convert a table structure to readable text format.
        
        Args:
            table: Table data from pdfplumber
            
        Returns:
            str: Formatted table text
        """
        if not table:
            return ""
        
        table_text = "\n[TABLE START]\n"
        
        for row in table:
            # Clean and join row cells
            clean_row = []
            for cell in row:
                if cell:
                    # Clean cell text
                    cleaned_cell = str(cell).strip().replace('\n', ' ')
                    clean_row.append(cleaned_cell)
                else:
                    clean_row.append("")
            
            # Join cells with | separator for clarity
            table_text += " | ".join(clean_row) + "\n"
        
        table_text += "[TABLE END]\n"
        
        return table_text
    
    def clean_extracted_text(self, text):
        """
        Clean and format the extracted text for better quality.
        
        Args:
            text (str): Raw extracted text
            
        Returns:
            str: Cleaned and formatted text
        """
        print("   🧹 Cleaning extracted text...", end=" ")
        
        # Remove excessive whitespace
        text = re.sub(r'\n\s*\n\s*\n', '\n\n', text)  # Max 2 consecutive newlines
        text = re.sub(r' +', ' ', text)  # Multiple spaces to single space
        
        # Fix common OCR issues
        text = text.replace('—', '-')  # Em dash to hyphen
        text = text.replace('\'', "'")  # Smart quote to regular quote
        text = text.replace('"', '"').replace('"', '"')  # Smart quotes
        
        # Remove page numbers and headers/footers (common patterns)
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            
            # Skip likely page numbers (just numbers or "Page X")
            if re.match(r'^(Page\s+)?\d+\s*$', line, re.IGNORECASE):
                continue
            
            # Skip very short lines that might be artifacts
            if len(line) < 3:
                continue
                
            cleaned_lines.append(line)
        
        cleaned_text = '\n'.join(cleaned_lines)
        
        print(f"✅ (removed {len(lines) - len(cleaned_lines)} unnecessary lines)")
        
        return cleaned_text
    
    def extract_text_from_single_pdf(self, pdf_path):
        """
        Extract text from a single PDF file.
        
        Args:
            pdf_path (str): Path to the PDF file
            
        Returns:
            tuple: (success: bool, extracted_text: str, file_stats: dict)
        """
        file_name = os.path.basename(pdf_path)
        print(f"\n📖 Processing: {file_name}")
        print("-" * 50)
        
        # Validate file
        if not self.validate_pdf_file(pdf_path):
            return False, "", {"error": "File validation failed"}
        
        file_stats = {
            'file_name': file_name,
            'file_path': pdf_path,
            'file_size': os.path.getsize(pdf_path),
            'pages_processed': 0,
            'pages_successful': 0,
            'pages_empty': 0,
            'pages_with_tables': 0,
            'characters_extracted': 0,
            'words_extracted': 0,
            'success': False,
            'error': None
        }
        
        try:
            with pdfplumber.open(pdf_path) as pdf:
                total_pages_in_file = len(pdf.pages)
                print(f"   Found {total_pages_in_file} pages to process")
                
                file_text = ""
                pages_successful_start = self.extraction_stats['successful_pages']
                pages_empty_start = self.extraction_stats['empty_pages']
                pages_tables_start = self.extraction_stats['pages_with_tables']
                
                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    self.processed_pages += 1
                    page_text = self.extract_text_from_page(page, page_num, total_pages_in_file)
                    file_text += page_text
                
                # Clean the extracted text
                cleaned_text = self.clean_extracted_text(file_text)
                
                # Calculate file statistics
                file_stats.update({
                    'pages_processed': total_pages_in_file,
                    'pages_successful': self.extraction_stats['successful_pages'] - pages_successful_start,
                    'pages_empty': self.extraction_stats['empty_pages'] - pages_empty_start,
                    'pages_with_tables': self.extraction_stats['pages_with_tables'] - pages_tables_start,
                    'characters_extracted': len(cleaned_text),
                    'words_extracted': len(cleaned_text.split()),
                    'success': True
                })
                
                print(f"   ✅ Successfully extracted {file_stats['characters_extracted']:,} characters, {file_stats['words_extracted']:,} words")
                
                return True, cleaned_text, file_stats
                
        except Exception as e:
            error_msg = f"Failed to process {file_name}: {str(e)}"
            print(f"   ❌ {error_msg}")
            file_stats['error'] = error_msg
            return False, "", file_stats
    
    def create_document_separator(self, file_name, file_stats):
        """
        Create a separator between documents in the combined output.
        
        Args:
            file_name (str): Name of the PDF file
            file_stats (dict): Statistics for this file
            
        Returns:
            str: Formatted separator text
        """
        separator = f"""

{'='*80}
DOCUMENT SOURCE: {file_name}
FILE PATH: {file_stats['file_path']}
EXTRACTED: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
PAGES: {file_stats['pages_processed']}
CHARACTERS: {file_stats['characters_extracted']:,}
WORDS: {file_stats['words_extracted']:,}
{'='*80}

"""
        return separator
    
    def print_extraction_summary(self):
        """Print a detailed summary of the extraction process."""
        print("\n" + "="*80)
        print("📊 MULTI-PDF EXTRACTION SUMMARY")
        print("="*80)
        print(f"Total files found: {self.extraction_stats['total_files']}")
        print(f"Successfully processed: {self.extraction_stats['successful_files']}")
        print(f"Failed files: {self.extraction_stats['failed_files']}")
        print(f"Total pages processed: {self.processed_pages}")
        print(f"Successful pages: {self.extraction_stats['successful_pages']}")
        print(f"Empty pages: {self.extraction_stats['empty_pages']}")
        print(f"Pages with tables: {self.extraction_stats['pages_with_tables']}")
        print(f"Combined characters: {self.extraction_stats['total_characters']:,}")
        print(f"Combined words: {self.extraction_stats['total_words']:,}")
        
        print(f"\n📋 File Details:")
        for i, file_detail in enumerate(self.extraction_stats['file_details'], 1):
            status = "✅" if file_detail['success'] else "❌"
            print(f"   {i}. {status} {file_detail['file_name']} - {file_detail['words_extracted']:,} words")
            if not file_detail['success'] and 'error' in file_detail:
                print(f"      Error: {file_detail['error']}")
        
        if self.extraction_stats['total_words'] > 0:
            print(f"\n✅ Multi-PDF extraction completed successfully!")
        else:
            print(f"\n⚠️ Warning: No text was extracted from any files.")
        
        print("="*80)
    
    def extract_text_from_multiple_pdfs(self, pdf_folder, output_path):
        """
        Main method to extract text from multiple PDFs with comprehensive error handling.
        
        Args:
            pdf_folder (str): Path to folder containing PDF files
            output_path (str): Path to output text file
            
        Returns:
            bool: True if extraction successful, False otherwise
        """
        print("🚀 Starting Multi-PDF Text Extraction")
        print("="*80)
        
        # Step 1: Find all PDF files
        pdf_files = self.find_pdf_files(pdf_folder)
        if not pdf_files:
            return False
        
        self.total_files = len(pdf_files)
        self.extraction_stats['total_files'] = self.total_files
        
        # Step 2: Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Created output directory: {output_dir}")
        
        # Step 3: Process each PDF file
        print(f"\n🔄 Processing {self.total_files} PDF files...")
        
        combined_text = ""
        
        for pdf_file in pdf_files:
            self.processed_files += 1
            
            # Extract text from this PDF
            success, file_text, file_stats = self.extract_text_from_single_pdf(pdf_file)
            
            # Update statistics
            self.extraction_stats['file_details'].append(file_stats)
            
            if success:
                self.extraction_stats['successful_files'] += 1
                
                # Add document separator
                separator = self.create_document_separator(file_stats['file_name'], file_stats)
                combined_text += separator + file_text + "\n"
                
            else:
                self.extraction_stats['failed_files'] += 1
                print(f"   ⚠️ Skipping {file_stats['file_name']} due to errors")
        
        # Step 4: Final text processing
        print(f"\n🔧 Finalizing combined document...")
        
        # Add header to combined document
        header = f"""COMBINED LEGAL DOCUMENTS EXTRACTION
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Total Files Processed: {self.extraction_stats['successful_files']}/{self.extraction_stats['total_files']}
Total Pages: {self.processed_pages}

This document contains text extracted from multiple PDF files in the Motor Vehicles Act collection.
Each document section is clearly marked with source information.

"""
        
        self.combined_text = header + combined_text
        
        # Calculate final statistics
        self.extraction_stats['total_characters'] = len(self.combined_text)
        self.extraction_stats['total_words'] = len(self.combined_text.split())
        
        # Step 5: Save combined output
        try:
            print(f"💾 Saving combined text to: {output_path}")
            with open(output_path, 'w', encoding='utf-8') as f:
                f.write(self.combined_text)
            
            # Step 6: Print summary
            self.print_extraction_summary()
            
            return True
            
        except Exception as e:
            print(f"\n❌ Failed to save combined output:")
            print(f"   {str(e)}")
            return False


def main():
    """
    Main function to run the multi-PDF extraction process.
    """
    print("🔍 Multi-PDF Text Extractor for Legal Documents")
    print("=" * 80)
    
    # Configure file paths
    pdf_folder = 'data/pdf'  # Input folder containing PDF files
    output_path = 'data/output/raw_data.txt'   # Output combined text file
    
    # Check if custom paths were provided via command line
    if len(sys.argv) > 1:
        pdf_folder = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    print(f"Input PDF folder: {pdf_folder}")
    print(f"Output combined file: {output_path}")
    
    # Create extractor and run extraction
    extractor = MultiPDFTextExtractor()
    success = extractor.extract_text_from_multiple_pdfs(pdf_folder, output_path)
    
    if success:
        print(f"\n🎉 SUCCESS! Combined text extracted and saved to '{output_path}'")
        print("\n📝 Next steps:")
        print("   1. Review the combined text file")
        print("   2. Run 'python data_cleaner.py' to further process the text")
        print("   3. Start the chatbot with 'python app.py'")
    else:
        print(f"\n💥 FAILED! Could not extract text from PDFs in '{pdf_folder}'")
        print("   Please check the error messages above and try again.")
    
    return success


if __name__ == '__main__':
    # Run the extraction process
    success = main()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)