from langchain.prompts import PromptTemplate

RETRIEVAL_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""You are a helpful research assistant.
Answer the question using ONLY the context provided. 
If you are not sure, say "I don't know."

Context:
{context}

Question: {question}

Rules:
- Cite sources at the end of each sentence in square brackets with their filename or ID.
- Do not make up information.
- Keep answers concise but informative.
"""
)