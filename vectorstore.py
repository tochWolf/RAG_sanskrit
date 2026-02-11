from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from .config import (
    INDEX_DIR,
    EMBEDDING_MODEL_NAME,
)


class VectorStore:
    """
    Simple FAISS-based vector store for text chunks.
    """

    def __init__(self, model_name: str = EMBEDDING_MODEL_NAME) -> None:
        self.model = SentenceTransformer(model_name)
        self.index = None  # type: ignore
        self.text_chunks: List[str] = []

    def build(self, chunks: List[str]) -> None:
        """
        Build FAISS index from provided text chunks.
        """
        self.text_chunks = chunks
        embeddings = self._encode(chunks)
        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

        INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(INDEX_DIR / "faiss.index"))
        np.save(INDEX_DIR / "chunks.npy", np.array(self.text_chunks, dtype=object))

    def load(self) -> None:
        """
        Load FAISS index and chunks from disk.
        """
        index_path = INDEX_DIR / "faiss.index"
        chunks_path = INDEX_DIR / "chunks.npy"
        if not index_path.exists() or not chunks_path.exists():
            raise FileNotFoundError("Index files not found; build the index first.")
        self.index = faiss.read_index(str(index_path))
        self.text_chunks = np.load(chunks_path, allow_pickle=True).tolist()

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Encode a list of texts into embeddings.
        """
        emb = self.model.encode(texts, show_progress_bar=True, convert_to_numpy=True)
        return emb.astype("float32")

    def search(self, query: str, top_k: int) -> List[Tuple[str, float]]:
        """
        Return top_k (chunk, distance) for a query.
        """
        if self.index is None:
            raise ValueError("Index is not loaded or built.")
        query_emb = self._encode([query])
        distances, indices = self.index.search(query_emb, top_k)
        results: List[Tuple[str, float]] = []
        for i, d in zip(indices[0], distances[0]):
            results.append((self.text_chunks[int(i)], float(d)))
        return results
