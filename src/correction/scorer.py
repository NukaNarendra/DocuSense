import os
import sys
from typing import List, Dict, Any
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class RelevanceScorer:
    def __init__(self, model_name: str, api_key: str):
        self.client = ChatNVIDIA(
            model=model_name,
            api_key=api_key,
            temperature=0.0,
            top_p=0.90,
            max_tokens=50,  # Increased slightly to allow the instruction to complete
        )

    def score_context(self, question: str, chunks: List[Dict[str, Any]]) -> bool:
        if not chunks:
            return False

        # We use the top 3 chunks to evaluate relevance to save time and tokens
        context_str = "\n\n".join(
            [f"--- Chunk {i + 1} ---\n{c.get('content', '')[:800]}" for i, c in enumerate(chunks[:3])])

        prompt = (
            "You are a strict evaluator. Does the context contain the answer to the question?\n"
            "Respond with exactly one word: YES or NO. Do not explain.\n\n"
            f"Question: {question}\n\n"
            f"Context chunks:\n{context_str}\n\n"
            "Verdict:"
        )

        try:
            # We use invoke instead of stream here because it is a single word output
            response = self.client.invoke([{"role": "user", "content": prompt}])

            content = response.content.strip().upper()
            print(f"\n[Scorer Verdict]: {content}")
            print("-" * 50)

            # Return True if the LLM graded it as YES
            return "YES" in content

        except Exception as e:
            print(f"\nCRAG Evaluation error: {str(e)}")
            return True