from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI
from backend.chains.prompts import RETRIEVAL_PROMPT
from backend.chains.retriever import get_retriever


def build_retrieval_qa(top_k=5):
    retriever = get_retriever(top_k=top_k)
    llm = ChatOpenAI(
        model_name="gpt-3.5-turbo", 
        temperature=0,
        max_tokens=500
        )
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": RETRIEVAL_PROMPT},
        return_source_documents=True
    )
    return qa_chain
