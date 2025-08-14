from backend.chains.conversational_chain import Chat

if __name__ == "__main__":
    qa_chain = Chat()
    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            break
        
        result = qa_chain.invoke({"chat_history": [], "question": query})
        print(result)
        print("Bot:", result["answer"])


