import os
from pathlib import Path
from dotenv import load_dotenv

# override=False ensures that if you set $env:NVIDIA_API_KEY in PowerShell,
# an old .env file won't overwrite your terminal key!
load_dotenv(override=False)


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

    # Read from terminal environment ($env:NVIDIA_API_KEY) or .env
    _raw_key = os.environ.get("NVIDIA_API_KEY", "")
    NVIDIA_API_KEY = _raw_key.strip().strip('"').strip("'") if _raw_key else None

    # Fast MoE model (or change to "nvidia/nemotron-3.5-super-vl-120b-a12b" if preferred)
    LLM_MODEL_NAME = "nvidia/nemotron-3.5-super-vl-120b-a12b"