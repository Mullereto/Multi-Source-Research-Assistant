from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings

PATH = r"D:\project\Python\DL(Mostafa saad)\Project\Multi-Source-Research-Assistant\data\processed\chroma_db"
def get_retriever(chroma_dir=PATH, top_k=5):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings, collection_name="research_chunks")
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    
    return retriever
