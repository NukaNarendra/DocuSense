from typing import List, Dict, Any


class HybridRetriever:
    def __init__(self, vector_store, keyword_store, rrf_k: int = 60):
        self.vector_store = vector_store
        self.keyword_store = keyword_store
        self.rrf_k = rrf_k

    def search(self, query: str, library_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        vector_results = self.vector_store.search(
            query=query,
            library_filter=library_name,
            top_k=top_k * 2
        )

        bm25_results = self.keyword_store.search(
            query=query,
            library_name=library_name,
            top_k=top_k * 2
        )

        rrf_scores = {}
        chunk_data = {}

        for rank, item in enumerate(vector_results):
            chunk_id = item["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0
                chunk_data[chunk_id] = item
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank + 1)

        for rank, item in enumerate(bm25_results):
            chunk_id = item["chunk_id"]
            if chunk_id not in rrf_scores:
                rrf_scores[chunk_id] = 0
                chunk_data[chunk_id] = item
            rrf_scores[chunk_id] += 1.0 / (self.rrf_k + rank + 1)

        sorted_chunks = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        final_results = []
        for chunk_id, score in sorted_chunks[:top_k]:
            data = chunk_data[chunk_id].copy()
            data["hybrid_score"] = score
            final_results.append(data)

        return final_results