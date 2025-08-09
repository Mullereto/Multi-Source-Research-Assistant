from backend.vectorstore.chroma_store import retreive, list_indexed_docs

print("Indexed documents:")
for doc in list_indexed_docs():
    print("-", doc)

while True:
    q = input("\nEnter query (or 'exit'): ")
    if q.lower() == "exit":
        break
    results = retreive(q, top_k=2)
    for r in results:
        print(f"\n[Source: {r['metadata']['source']}]")
        print(r['text'])
