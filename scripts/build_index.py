import os
import sys
import json
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AppConfig
from src.retrieval.vector_store import VectorStore
from src.retrieval.keyword_store import KeywordStore


def main():
    print("Starting Index Building Process...")

    processed_dir = AppConfig.DATA_PROCESSED_DIR
    if not os.path.exists(processed_dir):
        print(f"Error: Processed data directory not found at {processed_dir}")
        return

    json_files = [f for f in os.listdir(processed_dir) if f.endswith('_chunks.json')]

    if not json_files:
        print("No chunk JSON files found. Run ingest_data.py first.")
        return

    vector_store = VectorStore(
        db_dir=AppConfig.CHROMA_DB_DIR,
        collection_name=AppConfig.VECTOR_COLLECTION_NAME,
        model_name=AppConfig.EMBEDDING_MODEL
    )

    keyword_store = KeywordStore(
        index_dir=AppConfig.BM25_INDEX_DIR
    )

    for filename in json_files:
        library_name = filename.replace('_chunks.json', '')
        file_path = os.path.join(processed_dir, filename)

        print(f"\nProcessing {library_name}...")

        with open(file_path, 'r', encoding='utf-8') as f:
            chunks = json.load(f)

        print(f"Loaded {len(chunks)} chunks for {library_name}.")

        print("Building BM25 Keyword Index...")
        keyword_store.build_and_save_index(library_name, chunks)

        print("Building ChromaDB Vector Index (This may take a while)...")
        batch_size = 5000
        for i in tqdm(range(0, len(chunks), batch_size), desc="Vectorizing"):
            batch = chunks[i:i + batch_size]
            vector_store.add_chunks(batch)

    print("\nAll indices built successfully!")
    print(f"Vector DB Path: {AppConfig.CHROMA_DB_DIR}")
    print(f"Keyword DB Path: {AppConfig.BM25_INDEX_DIR}")


if __name__ == "__main__":
    main()