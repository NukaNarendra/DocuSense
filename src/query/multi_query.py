import os
import sys
import re
import concurrent.futures
from typing import List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from langchain_nvidia_ai_endpoints import ChatNVIDIA

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import AppConfig
from src.retrieval.vector_store import VectorStore
from src.retrieval.keyword_store import KeywordStore
from src.retrieval.hybrid import HybridRetriever


class QueryGenerationError(Exception):
    pass


class RetrievedChunk(BaseModel):
    chunk_id: str
    content: str
    metadata: Dict[str, Any]
    hybrid_score: float
    found_by_queries: List[str] = Field(default_factory=list)


class LLMVariantGenerator:
    def __init__(self, model_name: str, api_key: str):
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set or passed.")

        self.client = ChatNVIDIA(
            model=model_name,
            api_key=api_key,
            temperature=0.7,
            top_p=0.95,
            max_tokens=256,
        )
        self.num_variants = 3

    def _build_prompt(self, original_query: str) -> str:
        prompt = (
            "You are an AI assistant specialized in developer documentation retrieval. "
            "Your task is to generate alternative search queries to improve retrieval accuracy. "
            "Users often use different terminology than the official documentation. "
            f"Generate exactly {self.num_variants} different versions of the following question.\n\n"
            "Rules:\n"
            "1. Focus on technical synonyms and related concepts.\n"
            "2. Do not include introductory text, numbers, or bullet points.\n"
            "3. Output exactly one query per line.\n\n"
            f"Original Question: {original_query}\n\n"
            "Alternative Queries:\n"
        )
        return prompt

    def _parse_response(self, text: str) -> List[str]:
        lines = text.strip().split('\n')
        cleaned_lines = []
        for line in lines:
            cleaned = re.sub(r'^[\d\.\-\*\s]+', '', line).strip()
            if cleaned and len(cleaned) > 5 and not cleaned.lower().startswith("here are"):
                cleaned_lines.append(cleaned)

        if not cleaned_lines:
            raise QueryGenerationError("Failed to parse valid queries from LLM output.")

        return cleaned_lines[:self.num_variants]

    def generate(self, original_query: str) -> List[str]:
        prompt_content = self._build_prompt(original_query)
        messages = [{"role": "user", "content": prompt_content}]

        print(f"\n[Generating Query Variants for: '{original_query}']")

        try:
            response = self.client.invoke(messages)
            full_content = response.content
            print(full_content)
            print("-" * 50)

            if not full_content.strip():
                return [original_query]

            variants = self._parse_response(full_content)
            return variants

        except Exception as e:
            print(f"\nAPI Error during variant generation: {str(e)}")
            return [original_query]


class ResultDeduplicator:
    def __init__(self):
        self.merged_results: Dict[str, RetrievedChunk] = {}

    def add_result(self, query_source: str, result_dict: Dict[str, Any]):
        chunk_id = result_dict.get("chunk_id")
        if not chunk_id:
            return

        if chunk_id in self.merged_results:
            existing = self.merged_results[chunk_id]
            if result_dict["hybrid_score"] > existing.hybrid_score:
                existing.hybrid_score = result_dict["hybrid_score"]
            if query_source not in existing.found_by_queries:
                existing.found_by_queries.append(query_source)
        else:
            new_chunk = RetrievedChunk(
                chunk_id=chunk_id,
                content=result_dict.get("content", ""),
                metadata=result_dict.get("metadata", {}),
                hybrid_score=result_dict.get("hybrid_score", 0.0),
                found_by_queries=[query_source]
            )
            self.merged_results[chunk_id] = new_chunk

    def get_ranked_results(self, top_k: int) -> List[Dict[str, Any]]:
        sorted_chunks = sorted(
            self.merged_results.values(),
            key=lambda x: x.hybrid_score,
            reverse=True
        )

        final_list = []
        for chunk in sorted_chunks[:top_k]:
            final_list.append({
                "chunk_id": chunk.chunk_id,
                "content": chunk.content,
                "metadata": chunk.metadata,
                "hybrid_score": chunk.hybrid_score,
                "found_by_queries": chunk.found_by_queries
            })

        return final_list


class MultiQueryEngine:
    def __init__(self, hybrid_retriever: HybridRetriever, generator: LLMVariantGenerator):
        self.retriever = hybrid_retriever
        self.generator = generator

    def _fetch_single_query(self, query: str, library_name: str, top_k: int) -> Tuple[str, List[Dict[str, Any]]]:
        results = self.retriever.search(query=query, library_name=library_name, top_k=top_k)
        return query, results

    def execute_retrieval(self, original_query: str, library_name: str, top_k: int = 5) -> Tuple[
        List[Dict[str, Any]], List[str]]:
        queries_to_run = [original_query]

        try:
            generated_variants = self.generator.generate(original_query)
            queries_to_run.extend(generated_variants)
        except Exception as e:
            print(f"Warning: Variant generation failed. Details: {str(e)}")

        deduplicator = ResultDeduplicator()
        queries_to_run = list(dict.fromkeys(queries_to_run))

        print(f"Executing parallel retrieval across {len(queries_to_run)} query variations...")

        # This replaces the slow sequential loop with blazing fast parallel threads
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(queries_to_run)) as executor:
            future_to_query = {
                executor.submit(self._fetch_single_query, q, library_name, top_k): q
                for q in queries_to_run
            }

            for future in concurrent.futures.as_completed(future_to_query):
                query, results = future.result()
                for res in results:
                    deduplicator.add_result(query_source=query, result_dict=res)

        final_ranked_results = deduplicator.get_ranked_results(top_k=top_k)
        return final_ranked_results, queries_to_run


def setup_test_environment() -> MultiQueryEngine:
    vector_store = VectorStore(
        db_dir=AppConfig.CHROMA_DB_DIR,
        collection_name=AppConfig.VECTOR_COLLECTION_NAME,
        model_name=AppConfig.EMBEDDING_MODEL
    )

    keyword_store = KeywordStore(
        index_dir=AppConfig.BM25_INDEX_DIR
    )

    hybrid_retriever = HybridRetriever(
        vector_store=vector_store,
        keyword_store=keyword_store,
        rrf_k=AppConfig.RRF_K
    )

    # Fallbacks added here to prevent AttributeError and inject your exact config
    model_name = getattr(AppConfig, 'LLM_MODEL_NAME', "nvidia/nemotron-3-super-120b-a12b")
    api_key = getattr(AppConfig, 'NVIDIA_API_KEY',
                      "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")

    generator = LLMVariantGenerator(
        model_name=model_name,
        api_key=api_key
    )

    engine = MultiQueryEngine(
        hybrid_retriever=hybrid_retriever,
        generator=generator
    )

    return engine


def run_standalone_test():
    try:
        engine = setup_test_environment()
    except ValueError as e:
        print(f"Initialization Error: {str(e)}")
        sys.exit(1)

    test_library = "fastapi"
    test_question = "How do I create a dependency injection function that requires database access?"

    print("=" * 60)
    print("PHASE 3: MULTI-QUERY EXPANSION TEST")
    print("=" * 60)
    print(f"Target Library: {test_library}")
    print(f"Original Question: {test_question}")

    results, queries_used = engine.execute_retrieval(
        original_query=test_question,
        library_name=test_library,
        top_k=3
    )

    print("\n" + "=" * 60)
    print("FINAL MERGED & DEDUPLICATED RESULTS")
    print("=" * 60)

    print(f"\nQueries Processed ({len(queries_used)} total):")
    for idx, q in enumerate(queries_used, 1):
        print(f"  {idx}. {q}")

    print("\nTop Retrieved Chunks:")
    for idx, res in enumerate(results, 1):
        print(f"\n--- Rank {idx} ---")
        print(f"Source File: {res['metadata'].get('source_file')}")
        print(f"Found via {len(res['found_by_queries'])} queries: {res['found_by_queries']}")
        print(f"Hybrid Score: {res['hybrid_score']:.4f}")
        preview = res['content'][:250].replace('\n', ' ')
        print(f"Preview: {preview}...")


if __name__ == "__main__":
    run_standalone_test()