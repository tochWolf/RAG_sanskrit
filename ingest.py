from pathlib import Path
from typing import List
from .config import DATA_DIR, CHUNK_SIZE, CHUNK_OVERLAP


def load_sanskrit_files() -> List[str]:
    """
    Load all .txt files from the data folder and return their contents as a list of strings.
    Each element in the list is the full text of one file.
    """
    texts: List[str] = []
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    for path in DATA_DIR.glob("*.txt"):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback if encoding is different
            text = path.read_text(encoding="utf-16", errors="ignore")
        texts.append(text)
    return texts


def chunk_text(text: str) -> List[str]:
    """
    Split a long text into overlapping chunks for retrieval.
    Very simple character-based splitter.
    """
    chunks: List[str] = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + CHUNK_SIZE, length)
        chunk = text[start:end]
        chunks.append(chunk)
        if end == length:
            break
        start = end - CHUNK_OVERLAP
    return chunks


def build_corpus_chunks() -> List[str]:
    """
    Load all Sanskrit files and return a single list of text chunks.
    """
    all_texts = load_sanskrit_files()
    all_chunks: List[str] = []
    for doc in all_texts:
        all_chunks.extend(chunk_text(doc))
    return all_chunks
