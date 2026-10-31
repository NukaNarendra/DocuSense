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

        model_name = getattr(AppConfig, 'LLM_MODEL_NAME', "nvidia/nemotron-3.5-lightning-30b-a3b")

        # 1. Check terminal environment variable first ($env:NVIDIA_API_KEY)
        api_key = os.environ.get("NVIDIA_API_KEY") or getattr(AppConfig, 'NVIDIA_API_KEY', None)

        # 2. Fallback to Streamlit secrets if running on Streamlit Cloud
        if not api_key:
            try:
                import streamlit as st
                api_key = st.secrets.get("NVIDIA_API_KEY")
            except Exception:
                pass

        if not api_key:
            raise ValueError(
                "NVIDIA_API_KEY is missing! Run: $env:NVIDIA_API_KEY=\"nvapi-...\" in your terminal before starting Streamlit."
            )

        api_key = api_key.strip().strip('"').strip("'")
        os.environ["NVIDIA_API_KEY"] = api_key
        print(f"Loaded NVIDIA API Key: {api_key[:10]}... (len={len(api_key)})")

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
            print("\n✅ CRAG successfully intervened and provided corrected context!")

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