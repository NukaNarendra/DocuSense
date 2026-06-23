import os
import sys
from typing import List, Dict, Any, Tuple
from langchain_nvidia_ai_endpoints import ChatNVIDIA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.correction.scorer import RelevanceScorer
from src.query.multi_query import MultiQueryEngine


class CRAGManager:
    def __init__(self, model_name: str, api_key: str, multi_query_engine: MultiQueryEngine, scorer: RelevanceScorer):
        self.client = ChatNVIDIA(
            model=model_name,
            api_key=api_key,
            temperature=0.5,
            top_p=0.95,
            max_tokens=256,
        )
        self.multi_query_engine = multi_query_engine
        self.scorer = scorer

    def rewrite_query(self, failed_query: str) -> str:
        prompt = (
            "You are an expert technical query rewriting system. The following question failed to return relevant documentation. "
            "Rewrite it to be broader, focusing strictly on the core software engineering concepts. "
            "Translate vague terms into proper programming terminology. "
            "Remove any highly specific constraints that might be causing a database miss.\n\n"
            f"Original Question: {failed_query}\n\n"
            "Rewritten Query (just the raw query, no extra words):"
        )

        try:
            response = self.client.invoke([{"role": "user", "content": prompt}])
            rewritten = response.content.strip().replace('"', '')
            print(f"\n[CRAG Rewrote Query To]: {rewritten}")
            print("-" * 50)
            return rewritten
        except Exception as e:
            print(f"\nCRAG Rewrite error: {str(e)}")
            return failed_query

    def execute_with_correction(self, question: str, library_name: str, top_k: int) -> Tuple[
        List[Dict[str, Any]], List[str], bool]:
        print("\n--- Initial Retrieval Attempt ---")
        chunks, queries = self.multi_query_engine.execute_retrieval(question, library_name, top_k)

        is_relevant = self.scorer.score_context(question, chunks)

        if is_relevant:
            return chunks, queries, False

        print("\n[CRAG ALERT] 🚨 Retrieved context deemed insufficient. Triggering self-correction...")
        rewritten_query = self.rewrite_query(question)

        print(f"\n--- Second Retrieval Attempt (CRAG Corrected) ---")
        new_chunks, new_queries = self.multi_query_engine.execute_retrieval(rewritten_query, library_name, top_k)

        all_queries_tried = queries + new_queries
        return new_chunks, all_queries_tried, True