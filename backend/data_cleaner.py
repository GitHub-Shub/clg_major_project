# backend/data_cleaner.py
import re
import os

def clean_text(text):
    # Remove unwanted characters, headers, footers, and repeated "से 51"
    text = re.sub(r'से 51\]', '', text)  # Remove repeated "से 51"
    text = re.sub(r'\d+/\d+/MVL Section', '', text)  # Remove document IDs
    text = re.sub(r'THE GAZETTE OF INDIA EXTRAORDINARY.*?\n', '', text)  # Remove headers
    text = re.sub(r'\[PART II-.*?\n', '', text)  # Remove section headers
    text = re.sub(r'\.\.\.(truncated \d+ characters)\.\.\.', '', text)  # Remove truncation markers
    text = re.sub(r'\n\s*\n', '\n', text)  # Remove multiple newlines
    text = re.sub(r'[^\w\s.,;()-]', '', text)  # Keep alphanumeric, spaces, and basic punctuation
    return text.strip()

def process_mv_act_document(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        raw_text = f.read()
    
    cleaned_text = clean_text(raw_text)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(cleaned_text)

if __name__ == '__main__':
    input_file = 'raw_mv_act.txt'  # Save the provided document text as raw_mv_act.txt
    output_file = 'mv_act_cleaned.txt'
    process_mv_act_document(input_file, output_file)
    print(f"Cleaned text saved to {output_file}")