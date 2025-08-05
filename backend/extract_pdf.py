# backend/extract_pdf.py
import pdfplumber
import os

def extract_text_from_pdf(pdf_path, output_path):
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = ""
            for page in pdf.pages:
                text += page.extract_text() + "\n"
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(text)
        print(f"Text extracted and saved to {output_path}")
    except Exception as e:
        print(f"Error extracting text from PDF: {str(e)}")

if __name__ == '__main__':
    pdf_path = 'backend\\MV Act English.pdf'  # Replace with the actual PDF filename
    output_path = 'backend/raw_mv_act.txt'
    extract_text_from_pdf(pdf_path, output_path)