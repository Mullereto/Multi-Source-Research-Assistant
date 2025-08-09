import re
from typing import List, Dict
import tiktoken
import re
import os

def remove_headers_footers_per_page(text: str) -> str:
    """
    Removes common headers and footers on a per-page basis.

    """
    # Split pages by form feed (PDFs often use this after extraction)
    pages = text.split("\f")
    cleaned_pages = []

    for page in pages:
        lines = page.split("\n")
        if len(lines) <= 2:
            cleaned_pages.append(page)  # nothing to remove
            continue

        # Detect likely header/footer
        header = lines[0].strip()
        footer = lines[-1].strip()

        # Remove header/footer if they look like repeated text or page numbers
        new_lines = []
        for i, line in enumerate(lines):
            if i == 0 and (re.match(r"^\s*\d+\s*$", header) or len(header) < 80):
                # Skip header if short or just a page number
                continue
            if i == len(lines) - 1 and (re.match(r"^\s*\d+\s*$", footer) or len(footer) < 80):
                # Skip footer if short or just a page number
                continue
            new_lines.append(line)

        cleaned_pages.append("\n".join(new_lines))

    return "\n".join(cleaned_pages)



def clean_text(text:str) -> str:
    """
    Cleans the input text by removing extra whitespace, newlines
    """
    text = re.sub(r'\s+', ' ', text)

    return text.strip()

def chunk_text_by_tokens(text: str, source: str, chunk_size: int = 500, overlap: int = 100, model_name="gpt-3.5-turbo") -> List[Dict]:
    """
    Splits text into chunks with overlap (measured in tokens).
    Returns list of dicts with text, id, and metadata.
    """
    encoding = tiktoken.encoding_for_model(model_name)
    tokens = encoding.encode(text)
    
    chunks = []
    start = 0
    chunk_id = 1
    
    while start < len(tokens):
        end = start + chunk_size
        chunk = tokens[start:end]
        chunk_text = encoding.decode(chunk)

        chunks.append({
            "id": f"{os.path.basename(source)}_chunk{chunk_id}",
            "text": chunk_text,
            "metadata": {
                "source": os.path.basename(source),
                "chunk_id": chunk_id,
                "text_preview": chunk_text[:100] + '...'  # first 100 chars
            }
        })

        chunk_id += 1
        start += chunk_size - overlap

    return chunks