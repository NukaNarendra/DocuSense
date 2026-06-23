import os
from pathlib import Path


class AppConfig:
    BASE_DIR = Path(__file__).resolve().parent.parent
    DATA_PROCESSED_DIR = os.path.join(BASE_DIR, "data", "processed")
    CHROMA_DB_DIR = os.path.join(BASE_DIR, "data", "chroma_db")
    BM25_INDEX_DIR = os.path.join(BASE_DIR, "data", "bm25_indices")

    EMBEDDING_MODEL = "all-MiniLM-L6-v2"
    VECTOR_COLLECTION_NAME = "docusense_collection"

    TOP_K_VECTOR = 10
    TOP_K_BM25 = 10
    TOP_K_FINAL = 5
    RRF_K = 60

    os.makedirs(CHROMA_DB_DIR, exist_ok=True)
    os.makedirs(BM25_INDEX_DIR, exist_ok=True)