import os
import sys
import time
from typing import Dict, Any, Tuple

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.config import AppConfig
from src.retrieval.vector_store import VectorStore
from src.retrieval.keyword_store import KeywordStore
from src.retrieval.hybrid import HybridRetriever
from src.query.multi_query import LLMVariantGenerator, MultiQueryEngine
from src.generation.generator import AnswerGenerator
from src.correction.scorer import RelevanceScorer
from src.correction.crag import CRAGManager


class DocuSensePipeline:
    def __init__(self):
        self._initialize_components()

    def _initialize_components(self):
        print("Initializing DocuSense End-to-End Pipeline...")

        self.vector_store = VectorStore(
            db_dir=AppConfig.CHROMA_DB_DIR,
            collection_name=AppConfig.VECTOR_COLLECTION_NAME,
            model_name=AppConfig.EMBEDDING_MODEL
        )

        self.keyword_store = KeywordStore(
            index_dir=AppConfig.BM25_INDEX_DIR
        )

        self.hybrid_retriever = HybridRetriever(
            vector_store=self.vector_store,
            keyword_store=self.keyword_store,
            rrf_k=AppConfig.RRF_K
        )

        model_name = getattr(AppConfig, 'LLM_MODEL_NAME', "nvidia/nemotron-3-super-120b-a12b")

       api_key = getattr(AppConfig, 'NVIDIA_API_KEY', "")
        if not api_key:
            api_key = ""

        self.variant_generator = LLMVariantGenerator(
            model_name=model_name,
            api_key=api_key
        )

        self.multi_query_engine = MultiQueryEngine(
            hybrid_retriever=self.hybrid_retriever,
            generator=self.variant_generator
        )

        self.answer_generator = AnswerGenerator(
            model_name=model_name,
            api_key=api_key
        )

       self.relevance_scorer = RelevanceScorer(
            model_name=model_name,
            api_key=api_key
        )

        self.crag_manager = CRAGManager(
            model_name=model_name,
            api_key=api_key,
            multi_query_engine=self.multi_query_engine,
            scorer=self.relevance_scorer
        )

        print("Pipeline initialization complete.\n")

    def run(self, question: str, target_library: str) -> Dict[str, Any]:
        start_time = time.time()

        print("=" * 70)
        print(f"QUERY: {question}")
        print(f"TARGET LIBRARY: {target_library.upper()}")
        print("=" * 70)

        retrieved_chunks, generated_queries, crag_triggered = self.crag_manager.execute_with_correction(
            question=question,
            library_name=target_library,
            top_k=AppConfig.TOP_K_FINAL
        )

        if crag_triggered:
            print("\n CRAG successfully intervened and provided corrected context!")

        print(f"\nSuccessfully retrieved and merged {len(retrieved_chunks)} unique chunks.")

        final_answer = self.answer_generator.generate_answer(
            question=question,
            chunks=retrieved_chunks
        )

        execution_time = time.time() - start_time

        print(f"\nTotal Pipeline Execution Time: {execution_time:.2f} seconds")
        print("=" * 70)

        return {
            "question": question,
            "library": target_library,
            "queries_executed": generated_queries,
            "retrieved_context": retrieved_chunks,
            "crag_triggered": crag_triggered,
            "answer": final_answer,
            "execution_time": execution_time
        }


if __name__ == "__main__":
    try:
        pipeline = DocuSensePipeline()
    except Exception as e:
        print(f"Failed to start pipeline: {str(e)}")
        sys.exit(1)

    questions = [
        {
            "lib": "fastapi",
            "q": "How do I create a dependency injection function that requires database access?"
        },
        {
            "lib": "scikit-learn",
            "q": "I want to do the tree thingy but with the validator cross thing on my data. How?"
        }
    ]

    for item in questions:
        pipeline.run(question=item["q"], target_library=item["lib"])
        print("\n\n")
