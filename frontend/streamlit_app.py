import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import sys
# Add project root to Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import tempfile
import streamlit as st
from typing import List, Dict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from backend.wiki.wiki_fetcher import wiki_search
from backend.chains.retriever import get_retriever
from langchain_ollama import OllamaLLM
from backend.chains.prompts import RETRIEVAL_PROMPT

# If you have your own parsing utilities:
from backend.ingest.parse_pdf import parse_pdf, parse_docx
from backend.ingest.text_utils import remove_headers_footers_per_page, clean_text, chunk_text_by_tokens
from backend.vectorstore.chroma_store import add_chunk

EMBED_MODEL = "sentence-transformers/all-MiniLM-L12-v2"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# ===================
# FUNCTIONS
# ===================
from langchain_community.document_loaders import WikipediaLoader




def build_faiss_for_pages(pages: List[Dict]):
    if not pages:
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
        return None
    return FAISS.from_documents(docs, embeddings)

def retriever_multi(query: str, top_k_local: int = 5, top_k_wiki: int = 3):
    merged = []
    # Local retriever
    local_retriever = get_retriever(top_k=top_k_local)
    local_docs = local_retriever.invoke(query)
    for doc in local_docs:
        meta = dict(doc.metadata or {})
        meta["source_type"] = "local"
        merged.append({"text": doc.page_content, "metadata": meta, "source_type": "local", "score": None})

    # Wiki retriever
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
    parts = []
    mapping = []
    total = 0
    for i, item in enumerate(merged, start=1):
        snippet = item["text"].strip().replace("\n", " ")
        snippet = snippet[:2000]
        meta = item.get("metadata", {})
        citation = f"[{i}]"
        parts.append(f"{snippet} {citation}")
        mapping.append({"id": i, "metadata": meta})
        total += len(snippet)
        if total > max_chars:
            break
    return "\n\n".join(parts), mapping

# ===================
# STREAMLIT APP
# ===================
st.set_page_config(page_title="Multi-source QA", layout="wide")
st.title("📚 Multi-source QA Bot")

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "processed_files" not in st.session_state:
    st.session_state.processed_files = set()

llm = OllamaLLM(model="llama3.2:1b", temperature=0)

# --- FILE UPLOAD ---
st.sidebar.header("📂 Upload Documents")
uploaded_file = st.sidebar.file_uploader("Upload a PDF or DOCX", type=["pdf", "docx"])

if uploaded_file is not None:
    file_name = uploaded_file.name
    if file_name not in st.session_state.processed_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_name)[1]) as tmp_file:
            tmp_file.write(uploaded_file.read())
            temp_file_path = tmp_file.name

        if file_name.lower().endswith(".pdf"):
            pages_text = parse_pdf(temp_file_path)
        elif file_name.lower().endswith(".docx"):
            pages_text = parse_docx(temp_file_path)
        else:
            st.error("Unsupported file format")
            st.stop()

        cleaned_pages = remove_headers_footers_per_page(pages_text)
        cleaned_text = clean_text(cleaned_pages)
        chunks = chunk_text_by_tokens(cleaned_text, file_name, chunk_size=500, overlap=100)

        add_chunk(chunks, file_name)  # This should insert into your FAISS/local DB

        st.session_state.processed_files.add(file_name)
        st.sidebar.success(f"✅ '{file_name}' processed and added to vector store!")

# --- MAIN CHAT FORM ---
with st.form("qa_form", clear_on_submit=True):
    query = st.text_input("Ask me anything:", "")
    submitted = st.form_submit_button("Send")

if submitted and query.strip():
    merged = retriever_multi(query)
    if not merged:
        st.warning("I couldn't find relevant information in local docs or Wikipedia.")
    else:
        context, mapping = build_context_from_merged(merged)
        history_str = "\n".join([f"User: {q}\nBot: {a}" for q, a, _ in st.session_state.chat_history])
        prompt = RETRIEVAL_PROMPT.format(context=context, question=query, chat_history=history_str)
        answer = llm.invoke(prompt)

        sources_display = []
        seen = set()
        for m in mapping:
            meta = m["metadata"]
            src_type = meta.get("source_type", "unknown")
            title = meta.get("title", "Untitled")
            url = meta.get("source", "")
            key = (title, src_type, url)
            if key not in seen:
                seen.add(key)
                sources_display.append(f"[{m['id']}] {title} ({src_type}) {url}")

        st.session_state.chat_history.append((query, answer, sources_display))

# --- DISPLAY CHAT HISTORY ---
for q, a, sources in reversed(st.session_state.chat_history):
    with st.container():
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")
        if sources:
            st.markdown("**Sources:**")
            for s in sources:
                st.markdown(f"- {s}")
        st.markdown("---")
