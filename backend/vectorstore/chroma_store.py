import os
import chromadb
from chromadb.utils import embedding_functions

embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name="sentence-transformers/all-MiniLM-L12-v2")

CHROMA_PATH = os.path.join(r"D:\project\Python\DL(Mostafa saad)\Project\Multi-Source-Research-Assistant\data\processed\chroma_db") 
client = chromadb.PersistentClient(path=CHROMA_PATH)

collection = client.get_or_create_collection(
    name="research_chunks",
    embedding_function=embedding_fn,

)

def add_chunk(chunks, source):
    """
    Add chunks to ChromaDB.
    chunks: list of dict with keys ['id', 'text']
    """
    ids = []
    document = []
    meta_data = []
    
    for chunk in chunks:
        ids.append(chunk["id"])
        document.append(chunk["text"])
        meta_data.append({
            "source": source,
            "chunk_id": chunk["id"],
            "text_preview": chunk["text"][:50] + "..."
        })
    
    collection.add(
        ids=ids,
        documents=document,
        metadatas=meta_data
    )
    print(f"✅ Added {len(chunks)} chunks from {source} to ChromaDB.")


def retreive(quere, top_k=1):
    result = collection.query(
        query_texts=[quere],
        n_results=top_k
    )
    
    passages = []
    
    for doc, meta in zip(result["documents"][0], result["metadatas"][0]):
        passages.append({
            "text": doc,
            "metadata": meta
        })
    return passages


def list_indexed_docs():
    # Extract unique filenames from metadata
    print("Total chunks in collection:", collection.count())

    all_meta = collection.get()["metadatas"]
    docs = sorted({m["source"] for m in all_meta})
    return docs