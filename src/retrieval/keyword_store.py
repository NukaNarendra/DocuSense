import os
import pickle
import string
from typing import List, Dict, Any
from rank_bm25 import BM25Okapi


class KeywordStore:
    def __init__(self, index_dir: str):
        self.index_dir = index_dir
        self.indices = {}
        self.corpus_data = {}

    def preprocess_text(self, text: str) -> List[str]:
        text = text.lower()
        text = text.translate(str.maketrans('', '', string.punctuation))
        return text.split()

    def build_and_save_index(self, library_name: str, chunks: List[Dict[str, Any]]):
        tokenized_corpus = []
        corpus_lookup = {}

        for chunk in chunks:
            tokens = self.preprocess_text(chunk["content"])
            tokenized_corpus.append(tokens)
            corpus_lookup[chunk["chunk_id"]] = {
                "chunk_id": chunk["chunk_id"],
                "content": chunk["content"],
                "metadata": {
                    "library": chunk["library"],
                    "source_file": chunk["source_file"],
                    "title": chunk["title"],
                    "chunk_index": chunk["chunk_index"]
                }
            }

        bm25 = BM25Okapi(tokenized_corpus)

        index_path = os.path.join(self.index_dir, f"{library_name}_bm25.pkl")
        data_path = os.path.join(self.index_dir, f"{library_name}_corpus.pkl")

        with open(index_path, 'wb') as f:
            pickle.dump(bm25, f)

        with open(data_path, 'wb') as f:
            pickle.dump(corpus_lookup, f)

        self.indices[library_name] = bm25
        self.corpus_data[library_name] = corpus_lookup

    def load_index(self, library_name: str):
        index_path = os.path.join(self.index_dir, f"{library_name}_bm25.pkl")
        data_path = os.path.join(self.index_dir, f"{library_name}_corpus.pkl")

        if os.path.exists(index_path) and os.path.exists(data_path):
            with open(index_path, 'rb') as f:
                self.indices[library_name] = pickle.load(f)
            with open(data_path, 'rb') as f:
                self.corpus_data[library_name] = pickle.load(f)
        else:
            raise FileNotFoundError(f"BM25 index for {library_name} not found.")

    def search(self, query: str, library_name: str, top_k: int = 5) -> List[Dict[str, Any]]:
        if library_name not in self.indices:
            self.load_index(library_name)

        bm25 = self.indices[library_name]
        corpus = self.corpus_data[library_name]

        tokenized_query = self.preprocess_text(query)
        chunk_ids = list(corpus.keys())

        scores = bm25.get_scores(tokenized_query)

        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]

        results = []
        for idx in top_indices:
            chunk_id = chunk_ids[idx]
            score = scores[idx]
            if score > 0:
                result_item = corpus[chunk_id].copy()
                result_item["score"] = score
                results.append(result_item)

        return results