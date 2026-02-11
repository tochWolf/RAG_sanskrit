from typing import List, Tuple
from .config import TOP_K
from .ingest import build_corpus_chunks
from .vectorstore import VectorStore
from .generator import SanskritGenerator

def build_or_load_index() -> Tuple[VectorStore, int]:
    """
    Build a fresh index over the current data folder and return
    the VectorStore and number of chunks.
    For simplicity, this always rebuilds; in a real system we could load from disk.
    """
    chunks = build_corpus_chunks()
    if not chunks:
        raise RuntimeError("No .txt files found in data/ folder. Please add Sanskrit documents.")
    vs = VectorStore()
    vs.build(chunks)
    return vs, len(chunks)


def answer_query(
    vs: VectorStore,
    gen: SanskritGenerator,
    query: str,
    top_k: int = TOP_K,
) -> Tuple[str, List[str]]:
    """
    Run retrieval + generation for a single query.
    Returns (answer_text, contexts_list).
    """
    results = vs.search(query, top_k=top_k)
    contexts = [text for text, _ in results]
    if not contexts:
        return "No relevant context found in the documents.", []
    answer = gen.generate_answer(query, contexts)
    return answer, contexts

def run_rag_app() -> None:
    """
    Console version (unchanged behaviour): build index and interact via terminal.
    """
    print("Building vector index for Sanskrit corpus (first run may take time)...")
    vs, num_chunks = build_or_load_index()
    print(f"Index built over {num_chunks} chunks.")

    gen = SanskritGenerator()
    print("Generator model loaded on CPU.")

    while True:
        query = input("\nEnter query (or 'exit'): ").strip()
        if query.lower() in {"exit", "quit"}:
            print("Exiting RAG app.")
            break

        print("\nGenerating answer...")
        answer, contexts = answer_query(vs, gen, query)

        print("\n===== ANSWER =====")
        print(answer)
        print("==================")

        print("\n(Top retrieved context snippet for debugging):")
        if contexts:
            print(contexts[0][:400].replace("\n", " "))
