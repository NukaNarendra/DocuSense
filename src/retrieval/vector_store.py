import chromadb
from chromadb.utils import embedding_functions
from typing import List, Dict, Any


class VectorStore:
    def __init__(self, db_dir: str, collection_name: str, model_name: str):
        self.client = chromadb.PersistentClient(path=db_dir)
        self.embedding_fn = embedding_functions.SentenceTransformerEmbeddingFunction(model_name=model_name)
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def add_chunks(self, chunks: List[Dict[str, Any]], batch_size: int = 5000):
        total_chunks = len(chunks)
        for i in range(0, total_chunks, batch_size):
            batch = chunks[i:i + batch_size]
            ids = [chunk["chunk_id"] for chunk in batch]
            documents = [chunk["content"] for chunk in batch]
            metadatas = [{
                "library": chunk["library"],
                "source_file": chunk["source_file"],
                "title": chunk["title"],
                "chunk_index": chunk["chunk_index"]
            } for chunk in batch]

            self.collection.upsert(
                ids=ids,
                documents=documents,
                metadatas=metadatas
            )

    def search(self, query: str, library_filter: str = None, top_k: int = 5) -> List[Dict[str, Any]]:
        where_clause = {"library": library_filter} if library_filter else None

        results = self.collection.query(
            query_texts=[query],
            n_results=top_k,
            where=where_clause
        )

        formatted_results = []
        if results["ids"] and results["ids"][0]:
            for i in range(len(results["ids"][0])):
                formatted_results.append({
                    "chunk_id": results["ids"][0][i],
                    "content": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": results["distances"][0][i]
                })
        return formatted_results