import sys
from pathlib import Path

# Make sure we can import from the "code" folder
PROJECT_ROOT = Path(__file__).resolve().parent
CODE_DIR = PROJECT_ROOT / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

from rag_sanskrit.pipeline import run_rag_app

def main() -> None:
    """
    Entry point for the Sanskrit RAG system.
    For now, it will just call a simple demo function.
    Later we will connect this to real document ingestion and querying.
    """
    run_rag_app()

if __name__ == "__main__":
    main()

