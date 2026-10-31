import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()


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

    NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
    # Upgraded from deprecated nemotron-3-super-120b-a12b to the fast 3.5 Lightning MoE model
    # Alternatives: "nvidia/nemotron-3.5-super-vl-120b-a12b" or "nvidia/nemotron-3-ultra-550b-a55b"
    LLM_MODEL_NAME = "nvidia/nemotron-3.5-super-vl-120b-a12b"