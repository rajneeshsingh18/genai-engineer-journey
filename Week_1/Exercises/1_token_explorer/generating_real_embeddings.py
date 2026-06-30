# pip install google-genai numpy
import os
import numpy as np
from google import genai
from google.genai import types
from typing import List

from dotenv import load_dotenv
load_dotenv()

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))


def get_embedding(
    text: str,
    model: str = "gemini-embedding-001",
    task_type: str = "RETRIEVAL_DOCUMENT",
) -> List[float]:
    """
    Generate embedding vector for a given text using Gemini.

    Args:
        text: Input string to embed
        model: Gemini embedding model to use
        task_type: Optimizes the embedding for its intended use.
                   Use "RETRIEVAL_QUERY" for search queries and
                   "RETRIEVAL_DOCUMENT" for the documents being searched.

    Returns:
        List of floats representing the embedding vector
    """
    # Always clean the text — newlines hurt embedding quality
    text = text.replace("\n", " ").strip()

    response = client.models.embed_content(
        model=model,
        contents=text,
        config=types.EmbedContentConfig(task_type=task_type),
    )
    return response.embeddings[0].values


def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Measure similarity between two vectors.
    Returns value between -1 (opposite) and 1 (identical).
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# --- Demo: Semantic similarity ---
if __name__ == "__main__":
    sentences = [
        "How do I pay my electricity bill online?",   # Query
        "Steps to pay BESCOM bill using UPI",          # Very relevant
        "How to transfer money via NEFT?",             # Somewhat relevant
        "Best restaurants in Bengaluru for biryani",   # Not relevant
    ]

    # Embed the query with RETRIEVAL_QUERY, everything else with
    # RETRIEVAL_DOCUMENT — this asymmetry improves search-style matching.
    query_embedding = get_embedding(sentences[0], task_type="RETRIEVAL_QUERY")

    print(f"Query: {sentences[0]}\n")
    for sentence in sentences[1:]:
        emb = get_embedding(sentence, task_type="RETRIEVAL_DOCUMENT")
        score = cosine_similarity(query_embedding, emb)
        print(f"Score: {score:.4f} | Text: {sentence}")