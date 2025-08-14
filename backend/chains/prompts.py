from langchain.prompts import PromptTemplate

RETRIEVAL_PROMPT = PromptTemplate(
    input_variables=["context", "chat_history", "question"],
    template="""
You are a knowledgeable assistant. 
You must answer strictly based on the provided context, which comes from both local documents and Wikipedia.
Each excerpt in the context is labeled with a number in square brackets like [1], [2].

Instructions:
- Always cite sources inline in your answer using their number from the context. Example: "Deep learning is widely used [1]."
- If multiple sources support the same point, cite them together like [1][3].
- If the answer is not in the context, respond exactly with: "I don't know".
- Do not make up information or add citations that are not in the context.

Conversation history:
{chat_history}

Context (numbered excerpts from Wikipedia or local documents):
{context}

Question:
{question}

Answer (with inline citations and clear, concise, and factual):
"""
)
