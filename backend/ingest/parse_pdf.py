import pdfplumber
import os
import docx 

def parse_pdf(file_path:str) -> str:
    """
    Extracts text from a PDF file.
    Returns the full text as a string.
    
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text() or ""
            text += page_text + "\n"
    
    return text.strip()

def parse_docx(file_path:str) -> str:
    """
    Extracts text from a DOCX file.
    Returns the full text as a string.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    
    doc = docx.Document(file_path)
    text = "\n".join([para.text for para in doc.paragraphs])
    
    return text.strip()