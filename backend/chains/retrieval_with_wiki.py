
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from typing import List, Dict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from backend.wiki.wiki_fetcher import wiki_search
from backend.chains.retriever import get_retriever
from langchain_ollama import OllamaLLM
from backend.chains.prompts import RETRIEVAL_PROMPT

EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100


def build_faiss_for_pages(pages: List[Dict]):
    if not pages:
        print("there is no pages")
        return None

    spliter = RecursiveCharacterTextSplitter(chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP)
    embeddings = HuggingFaceEmbeddings(model_name=EMBED_MODEL)
    docs = []
    
    for page in pages:
        chunks = spliter.split_text(page['content'])
        for i, chunk in enumerate(chunks):
            meta = {
                "source": page.get("url"),
                "title": page.get("title"),
                "crawl_date": page.get("crawl_date"),
                "source_type": "wiki",
                "web_chunk_id": f"{page.get('title')}#chunk{i}"
            }
            docs.append(Document(page_content=chunk, metadata=meta))
    
    if not docs:
        print("there is no docs")
        return None
    faiss_index = FAISS.from_documents(docs, embeddings)
    return faiss_index


def retriever_multi(query: str, top_k_local: int = 5, top_k_wiki: int = 3):
    """
    Returns merged results from local vectorstore and Wikipedia faiss index.
    Each returned item: {text, metadata, source_type, score}
    """
    merged = []
    
    #local retriever
    local_retriever = get_retriever(top_k=top_k_local)
    local_docs = local_retriever.invoke(query)
    
    for doc in local_docs:
        meta = dict(doc.metadata or {})
        meta["source_type"] = "local"
        merged.append({"text": doc.page_content, "metadata": meta, "source_type": "local", "score": None})
        
        
    
    #wiki retriever
    wiki_pages = wiki_search(query, top_k=top_k_wiki)
    faiss_index = build_faiss_for_pages(wiki_pages)
    if faiss_index:
        wiki_docs = faiss_index.similarity_search(query, k=top_k_wiki)
        for doc in wiki_docs:
            m = dict(doc.metadata or {})
            m["source_type"] = "wiki"
            merged.append({"text": doc.page_content, "metadata": m, "source_type": "wiki", "score": None})

    return merged

def build_context_from_merged(merged: List[Dict], max_chars: int = 4000):
    """
    Build a single context string by concatenating top merged results until max_chars reached.
    Include short citations like [1], [2] that map to metadata.
    Returns the context string and a mapping list.
    """
    
    parts = []
    mapping = []
    total = 0
    
    for i, item in enumerate(merged, start=1):
        snippet = item["text"].strip().replace("\n", " ")
        snippet = snippet[:2000]
        
        meta = item.get("metadata", {})         
        citation = f"[{i}]"
        parts.append(f"{snippet} {citation}")
        mapping.append({"id":i, "metadata": meta})
        
        total += len(snippet)
        if total > max_chars:
            break
        
    context = "\n\n".join(parts)
    return context, mapping


def answer_multi_source(top_k_local: int = 5, top_k_wiki: int = 3):
    chat_history = []  # (question, answer, sources)
    llm = OllamaLLM(model="llama3.2:1b", temperature=0)

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break

        # Merge local + wiki results
        merged = retriever_multi(query, top_k_local, top_k_wiki)
        if not merged:
            print("Bot: I couldn't find relevant information in local docs or Wikipedia.")
            continue

        # Build context and track mapping
        context, mapping = build_context_from_merged(merged)

        # Prepare chat history for prompt
        history_str = "\n".join([f"User: {q}\nBot: {a}" for q, a, _ in chat_history])

        # Build the final prompt
        prompt = RETRIEVAL_PROMPT.format(
            context=context,
            question=query,
            chat_history=history_str
        )

        # Get LLM answer
        answer = llm.invoke(prompt)

        # Prepare source display
        seen = set()
        sources_display = []
        for m in mapping:
            meta = m["metadata"]
            src_type = meta.get("source_type", "unknown")
            title = meta.get("title", "Untitled")
            url = meta.get("source", "")

            key = (title, src_type, url)
            if key not in seen:
                seen.add(key)
                sources_display.append(f"[{m['id']}] {title} ({src_type}) {url}")

        # Show bot's answer
        print("\nBot:", answer)
        print("Sources:")
        for s in sources_display:
            print(s)

        # Save to history
        chat_history.append((query, answer, sources_display))



    