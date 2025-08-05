
# backend/extract_pdf.py
"""
Enhanced PDF Text Extractor for Motor Vehicles Act
==================================================

This script extracts text from PDF files with high quality and clarity.
It's designed to be beginner-friendly while providing the best possible
text extraction results for RAG applications.

Features:
- Detailed progress tracking
- Smart text cleaning and formatting
- Table extraction support
- Error handling with helpful messages
- Quality validation
- Multiple extraction strategies
"""

import pdfplumber
import os
import sys
from pathlib import Path
import re


class PDFTextExtractor:
    """
    A comprehensive PDF text extractor that prioritizes data quality and clarity.
    
    This class handles PDF text extraction with multiple strategies to ensure
    we get the highest quality text output for our RAG system.
    """
    
    def __init__(self):
        """Initialize the PDF extractor with default settings."""
        self.total_pages = 0
        self.processed_pages = 0
        self.extracted_text = ""
        self.extraction_stats = {
            'total_characters': 0,
            'total_words': 0,
            'empty_pages': 0,
            'pages_with_tables': 0,
            'successful_pages': 0
        }
    
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
            print("   Please check the file path and try again.")
            return False
        
        if not pdf_file.suffix.lower() == '.pdf':
            print(f"❌ Error: '{pdf_path}' is not a PDF file")
            return False
        
        if pdf_file.stat().st_size == 0:
            print(f"❌ Error: PDF file '{pdf_path}' is empty")
            return False
        
        print(f"✅ PDF file validated: {pdf_file.name}")
        print(f"   File size: {pdf_file.stat().st_size:,} bytes")
        
        return True
    
    def extract_text_from_page(self, page, page_num):
        """
        Extract text from a single PDF page with multiple strategies.
        
        Args:
            page: pdfplumber page object
            page_num (int): Page number for logging
            
        Returns:
            str: Extracted text from the page
        """
        print(f"📄 Processing page {page_num}/{self.total_pages}...", end=" ")
        
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
        print("\n🧹 Cleaning extracted text...")
        
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
        
        print(f"✅ Text cleaning completed")
        print(f"   Removed {len(lines) - len(cleaned_lines)} unnecessary lines")
        
        return cleaned_text
    
    def calculate_stats(self, text):
        """
        Calculate statistics about the extracted text.
        
        Args:
            text (str): Extracted text
        """
        self.extraction_stats['total_characters'] = len(text)
        self.extraction_stats['total_words'] = len(text.split())
    
    def print_extraction_summary(self):
        """Print a detailed summary of the extraction process."""
        print("\n" + "="*60)
        print("📊 EXTRACTION SUMMARY")
        print("="*60)
        print(f"Total pages processed: {self.processed_pages}")
        print(f"Successful pages: {self.extraction_stats['successful_pages']}")
        print(f"Empty pages: {self.extraction_stats['empty_pages']}")
        print(f"Pages with tables: {self.extraction_stats['pages_with_tables']}")
        print(f"Total characters extracted: {self.extraction_stats['total_characters']:,}")
        print(f"Total words extracted: {self.extraction_stats['total_words']:,}")
        
        if self.extraction_stats['total_words'] > 0:
            print("✅ Extraction appears successful!")
        else:
            print("⚠️ Warning: No text was extracted. Check your PDF file.")
        
        print("="*60)
    
    def extract_text_from_pdf(self, pdf_path, output_path):
        """
        Main method to extract text from PDF with comprehensive error handling.
        
        Args:
            pdf_path (str): Path to input PDF file
            output_path (str): Path to output text file
            
        Returns:
            bool: True if extraction successful, False otherwise
        """
        print("🚀 Starting PDF Text Extraction")
        print("="*60)
        
        # Step 1: Validate input file
        if not self.validate_pdf_file(pdf_path):
            return False
        
        # Step 2: Create output directory if needed
        output_dir = os.path.dirname(output_path)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir)
            print(f"📁 Created output directory: {output_dir}")
        
        try:
            # Step 3: Open and process PDF
            print(f"\n📖 Opening PDF file...")
            with pdfplumber.open(pdf_path) as pdf:
                self.total_pages = len(pdf.pages)
                print(f"   Found {self.total_pages} pages to process")
                
                all_text = ""
                
                # Process each page
                for page_num, page in enumerate(pdf.pages, 1):
                    self.processed_pages = page_num
                    page_text = self.extract_text_from_page(page, page_num)
                    all_text += page_text
                
                # Step 4: Clean the extracted text
                self.extracted_text = self.clean_extracted_text(all_text)
                
                # Step 5: Calculate statistics
                self.calculate_stats(self.extracted_text)
                
                # Step 6: Save to output file
                print(f"\n💾 Saving extracted text to: {output_path}")
                with open(output_path, 'w', encoding='utf-8') as f:
                    f.write(self.extracted_text)
                
                # Step 7: Print summary
                self.print_extraction_summary()
                
                return True
                
        except Exception as e:
            print(f"\n❌ Critical Error during extraction:")
            print(f"   {str(e)}")
            print("\n💡 Troubleshooting tips:")
            print("   - Ensure the PDF is not corrupted")
            print("   - Check if the PDF is password protected")
            print("   - Try with a different PDF file")
            print("   - Make sure you have write permissions for the output directory")
            
            return False


def main():
    """
    Main function to run the PDF extraction process.
    
    This function handles the command-line interface and coordinates
    the extraction process.
    """
    print("🔍 Motor Vehicles Act PDF Text Extractor")
    print("=" * 60)
    
    # Configure file paths
    # You can modify these paths as needed
    pdf_path = 'MV Act English.pdf'  # Input PDF file
    output_path = 'raw_mv_act.txt'   # Output text file
    
    # Check if custom paths were provided via command line
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
    if len(sys.argv) > 2:
        output_path = sys.argv[2]
    
    print(f"Input PDF: {pdf_path}")
    print(f"Output text file: {output_path}")
    
    # Create extractor and run extraction
    extractor = PDFTextExtractor()
    success = extractor.extract_text_from_pdf(pdf_path, output_path)
    
    if success:
        print(f"\n🎉 SUCCESS! Text extracted and saved to '{output_path}'")
        print("\n📝 Next steps:")
        print("   1. Review the extracted text file")
        print("   2. Run 'python data_cleaner.py' to further process the text")
        print("   3. Start the chatbot with 'python app.py'")
    else:
        print(f"\n💥 FAILED! Could not extract text from '{pdf_path}'")
        print("   Please check the error messages above and try again.")
    
    return success


if __name__ == '__main__':
    # Run the extraction process
    success = main()
    
    # Exit with appropriate code
    sys.exit(0 if success else 1)