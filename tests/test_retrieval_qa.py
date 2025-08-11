# tests/test_retrieval_qa.py
from backend.chains.retrieval_qa import build_retrieval_qa

if __name__ == "__main__":
    qa = build_retrieval_qa(top_k=5)
    query = "who is mark zuckerberg?"
    result = qa.invoke({"query": query}, return_source_documents=True)

    print(result)


    print("\n=== Answer ===")
    print(result["result"])

    print("\n=== Sources ===")
    for doc in result["source_documents"]:
        print(f"{doc.metadata.get('source')} - {doc.page_content[:100]}...")


