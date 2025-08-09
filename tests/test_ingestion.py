from backend.ingest.parse_pdf import *
from backend.ingest.text_utils import *
from backend.ingest.save_chunks import *
from pathlib import Path
from backend.vectorstore.chroma_store import add_chunk

# Change this to your file path
file_path = r"D:\project\Python\DL(Mostafa saad)\Project\Multi-Source-Research-Assistant\ml_test.docx"  # or "sample.docx"

# Step 1: Parse
if file_path.lower().endswith(".pdf"):
    raw_text = parse_pdf(file_path)
elif file_path.lower().endswith(".docx"):
    raw_text = parse_docx(file_path)
else:
    raise ValueError("Unsupported file type.")

print(f"Raw text length: {len(raw_text)} characters")

# Step 2: Remove headers/footers
no_headers = remove_headers_footers_per_page(raw_text)

# Step 3: Clean
cleaned_text = clean_text(no_headers)

# Step 4: Chunk by tokens
source = Path(file_path).name
chunks = chunk_text_by_tokens(cleaned_text, source=source, chunk_size=5, overlap=2)
print(f"Generated {len(chunks)} chunks.")

add_chunk(chunks, source=os.path.basename(file_path))

# Step 5: Save
save_chunks(chunks, "data/processed/sample_chunks.json")

