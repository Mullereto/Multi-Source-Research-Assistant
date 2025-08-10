from langchain_chroma import Chroma
#from langchain_openai import OpenAIEmbeddings
from langchain_community.embeddings import HuggingFaceEmbeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")


PATH = r"D:\project\Python\DL(Mostafa saad)\Project\Multi-Source-Research-Assistant\data\processed\chroma_db"
def get_retriever(chroma_dir=PATH, top_k=5):
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L12-v2")
    db = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
    retriever = db.as_retriever(search_kwargs={"k": top_k})
    
    return retriever
