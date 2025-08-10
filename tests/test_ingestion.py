import argparse
import os
import logging
from backend.ingest.parse_pdf import parse_pdf, parse_docx
from backend.ingest.text_utils import remove_headers_footers_per_page, clean_text, chunk_text_by_tokens
from backend.ingest.save_chunks import save_chunks
from backend.vectorstore.chroma_store import add_chunk
from backend.vectorstore.chroma_store import retreive, list_indexed_docs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def ingest_file(file_path: str, chunk_size: int = 500, overlap: int = 100):
    """
    Ingests a single PDF or DOCX file, cleans text, chunks it, and saves to processed folder.
    """
    if not os.path.exists(file_path):
        logging.error(f"File not found: {file_path}")
        return

    # Step 1: Parse
    logging.info(f"Parsing file: {file_path}")
    if file_path.lower().endswith(".pdf"):
        raw_text = parse_pdf(file_path)
    elif file_path.lower().endswith(".docx"):
        raw_text = parse_docx(file_path)
    else:
        logging.error("Unsupported file type. Please provide PDF or DOCX.")
        return

    logging.info(f"Raw text length: {len(raw_text)} characters")

    # Step 2: Remove headers/footers
    no_headers = remove_headers_footers_per_page(raw_text)

    # Step 3: Clean text
    cleaned_text = clean_text(no_headers)

    # Step 4: Chunk text
    chunks = chunk_text_by_tokens(cleaned_text, os.path.basename(file_path), chunk_size=chunk_size, overlap=overlap)

    logging.info(f"Generated {len(chunks)} chunks.")

    # Step 5: Save chunks
    os.makedirs("data/processed", exist_ok=True)
    output_path = os.path.join("data/processed", f"{os.path.basename(file_path)}_chunks.json")
    save_chunks(chunks, output_path)
    logging.info(f"Chunks saved to {output_path}")
    # Step 6: Add chunks to ChromaDB
    add_chunk(chunks, os.path.basename(file_path))




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest PDF or DOCX into text chunks.")
    parser.add_argument("--file", type=str, required=True, help="Path to the PDF or DOCX file.")
    parser.add_argument("--chunk_size", type=int, default=500, help="Chunk size in tokens.")
    parser.add_argument("--overlap", type=int, default=100, help="Token overlap between chunks.")
    args = parser.parse_args()

    ingest_file(args.file, args.chunk_size, args.overlap)
    
    
    #Run The cli_vectorstore.py


    print("Indexed documents:")
    for doc in list_indexed_docs():
        print("-", doc)

    while True:
        q = input("\nEnter query (or 'exit'): ")
        if q.lower() == "exit":
            break
        results = retreive(q, top_k=2)
        for r in results:
            print(f"\n[Source: {r['metadata']['source']}]")
            print(r['text'])

