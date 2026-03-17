"""
llm.py
Calls the Mistral API to generate context-grounded answers via RAG.
"""

import os
from typing import List

from mistralai import Mistral

_client: Mistral | None = None


def _get_client() -> Mistral:
    global _client
    if _client is None:
        api_key = os.environ.get("MISTRAL_API_KEY")
        if not api_key:
            raise EnvironmentError(
                "MISTRAL_API_KEY environment variable is not set. "
                "Please export it before running the app."
            )
        _client = Mistral(api_key=api_key)
    return _client


SYSTEM_PROMPT = """You are a helpful AI knowledge assistant. 
Answer the user's question based ONLY on the provided context chunks.
If the answer is not in the context, say "I don't have enough information in the uploaded documents to answer that."
Be concise, accurate, and helpful. Format your answer in clear paragraphs."""


def generate_answer(question: str, context_chunks: List[str]) -> str:
    """
    Generate a context-grounded answer using Mistral.

    Args:
        question:       The user's natural language question.
        context_chunks: Retrieved document chunks from Endee.

    Returns:
        A string containing the generated answer.
    """
    if not context_chunks:
        return "No relevant context found in your documents for this question."

    context_text = "\n\n---\n\n".join(
        f"[Chunk {i+1}]\n{chunk}" for i, chunk in enumerate(context_chunks)
    )

    user_message = f"""Context:
{context_text}

Question: {question}

Answer:"""

    client = _get_client()
    response = client.chat.complete(
        model="mistral-small-latest",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": user_message},
        ],
        temperature=0.2,
        max_tokens=1024,
    )

    return response.choices[0].message.content.strip()
