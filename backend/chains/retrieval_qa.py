from langchain.chains import RetrievalQA
from langchain_ollama import OllamaLLM
from backend.chains.prompts import RETRIEVAL_PROMPT
from backend.chains.retriever import get_retriever


def build_retrieval_qa(top_k=5):
    retriever = get_retriever(top_k=top_k)
    llm = OllamaLLM(
        model="llama3.2:1b",
        temperature=0,
        num_predict=500,
        )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RETRIEVAL_PROMPT},
        return_source_documents=True
    )
    return qa_chain
