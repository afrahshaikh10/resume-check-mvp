import pdfplumber
import docx2txt
import re
import streamlit as st # Keep if you're specifically targeting Streamlit
from typing import Optional

def clean_text(text: str) -> str:
    """Clean and normalize text"""
    if not text:
        return ""
    
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^\w\s\.,;:()\-+#@]', ' ', text)
    return text.lower().strip()

def extract_text_from_file(file) -> Optional[str]:
    if file is None:
        return None

    file_type = file.type
    raw_text = ""

    try:
        if file_type == "application/pdf":
            with pdfplumber.open(file) as pdf:
                for page in pdf.pages:
                    page_text = page.extract_text()
                    if page_text:
                        raw_text += page_text + "\n"
        elif file_type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            raw_text = docx2txt.process(file)
        else:
            if st:
                st.error(f"Unsupported file type: {file_type}")
            return None
    except Exception as e:
        if st:
            st.error(f"Error reading file: {str(e)}")
        return None
    
    return clean_text(raw_text)

def test_extraction():
    print("Text extraction module loaded successfully!")
    print("Supported formats: PDF, DOCX")

if __name__ == "__main__":
    test_extraction()
