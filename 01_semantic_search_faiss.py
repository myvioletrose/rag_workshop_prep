"""Exercise 1: semantic retrieval using OpenAI embeddings and raw FAISS.

This deliberately avoids LangChain so you can see the core mechanics:
text -> embeddings -> FAISS index -> query embedding -> nearest chunks.
"""

from __future__ import annotations

import os
from pathlib import Path

import faiss
import numpy as np
from dotenv import load_dotenv
from openai import OpenAI


ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "workshop_handbook.txt"


def load_passages(path: Path) -> list[str]:
    text = path.read_text(encoding="utf-8")
    passages = [part.strip() for part in text.split("\n\n") if part.strip()]
    if not passages:
        raise ValueError(f"No passages found in {path}")
    return passages


def embed_texts(client: OpenAI, texts: list[str], model: str) -> np.ndarray:
    response = client.embeddings.create(model=model, input=texts)
    ordered = sorted(response.data, key=lambda item: item.index)
    matrix = np.asarray([item.embedding for item in ordered], dtype="float32")
    faiss.normalize_L2(matrix)  # inner product now acts like cosine similarity
    return matrix


def main() -> None:
    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is missing. Copy .env.example to .env and add your key.")

    embedding_model = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    client = OpenAI()
    passages = load_passages(DATA_PATH)

    print(f"Loaded {len(passages)} passages.")
    passage_vectors = embed_texts(client, passages, embedding_model)

    index = faiss.IndexFlatIP(passage_vectors.shape[1])
    index.add(passage_vectors)
    print(f"FAISS index contains {index.ntotal} vectors of dimension {index.d}.\n")

    default_question = "Can I use VS Code, and what should I bring for power?"
    question = input(f"Question [{default_question}]: ").strip() or default_question
    query_vector = embed_texts(client, [question], embedding_model)

    top_k = min(3, len(passages))
    scores, positions = index.search(query_vector, top_k)

    print("\nTop retrieved passages (higher cosine score is better):")
    for rank, (score, position) in enumerate(zip(scores[0], positions[0]), start=1):
        preview = passages[int(position)].replace("\n", " ")
        print(f"\n{rank}. score={score:.3f}, passage={int(position)}")
        print(preview)

    print("\nTry again with questions that use different words from the document.")


if __name__ == "__main__":
    main()
