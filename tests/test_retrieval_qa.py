# tests/test_retrieval_qa.py
from backend.chains.retrieval_qa import build_retrieval_qa

if __name__ == "__main__":
    qa = build_retrieval_qa(top_k=1)
    query = "What are the main findings of the research on AI safety?"
    result = qa.invoke({"query": query})

    print("\n=== Answer ===")
    print(result["result"])

    print("\n=== Sources ===")
    for doc in result["source_documents"]:
        print(f"{doc.metadata.get('source', 'unknown')} - {doc.page_content[:100]}...")


