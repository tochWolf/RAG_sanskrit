from pathlib import Path

# Paths
PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
INDEX_DIR: Path = PROJECT_ROOT / "data" / "index"

# Index / retrieval settings
CHUNK_SIZE: int = 400
CHUNK_OVERLAP: int = 50
TOP_K: int = 5

# Model names (CPU friendly)
EMBEDDING_MODEL_NAME: str = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
GENERATION_MODEL_NAME: str = "google/byt5-small"  # multilingual, byte-level, CPU friendly


# Generation settings
MAX_NEW_TOKENS: int = 128
TEMPERATURE: float = 0.7
