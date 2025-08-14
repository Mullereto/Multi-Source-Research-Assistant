from langchain_chroma import Chroma
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from backend.chains.prompts import RETRIEVAL_PROMPT


PATH = r"D:\project\Python\DL(Mostafa saad)\Project\Multi-Source-Research-Assistant\data\processed\chroma_db"


def Chat():

    vector_DB = Chroma(
        collection_name="research_chunks",
        persist_directory=PATH,
    )
    print(vector_DB._collection.count())
    retriever = vector_DB.as_retriever(search_kwargs={"k": 5})


    memory = ConversationBufferWindowMemory(
        k=5,
        memory_key="chat_history",
        return_messages=True,

    )

    llm = OllamaLLM(
        model="llama3.2:1b",
        temperature=0,
    )

    QA_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": RETRIEVAL_PROMPT}
    )

    return QA_chain