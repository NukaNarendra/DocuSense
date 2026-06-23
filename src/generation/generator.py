import os
import sys
from typing import List, Dict, Any, Tuple
from langchain_nvidia_ai_endpoints import ChatNVIDIA


class AnswerGenerator:
    def __init__(self, model_name: str, api_key: str):
        if not api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is not set or passed.")

        self.client = ChatNVIDIA(
            model=model_name,
            api_key=api_key,
            temperature=0.2,
            top_p=0.90,
            max_tokens=4096,
            reasoning_budget=2048,
            chat_template_kwargs={"enable_thinking": True},
        )

    def _format_context(self, chunks: List[Dict[str, Any]]) -> Tuple[str, Dict[int, str]]:
        context_str = ""
        source_map = {}

        for idx, chunk in enumerate(chunks, 1):
            source_file = chunk.get("metadata", {}).get("source_file", "Unknown Source")
            source_map[idx] = source_file
            content = chunk.get("content", "").strip()

            context_str += f"--- Document [{idx}] ---\n"
            context_str += f"Source Path: {source_file}\n"
            context_str += f"Content:\n{content}\n\n"

        return context_str, source_map

    def _build_prompt(self, question: str, context: str) -> str:
        prompt = (
            "You are DocuSense, an expert developer assistant specialized in reading and interpreting open-source documentation. "
            "You are answering a user's technical question based STRICTLY on the provided context documents.\n\n"
            "CRITICAL INSTRUCTIONS:\n"
            "1. You must answer the question using ONLY the provided context.\n"
            "2. If the context does not contain the answer, you must state: 'I do not have enough information in the provided documentation to answer this.' Do not guess.\n"
            "3. You MUST cite your sources inline using bracketed numbers corresponding to the Document numbers provided (e.g., [1], [2]).\n"
            "4. Include code snippets exactly as they appear in the documentation if they are relevant to the user's question.\n"
            "5. At the very end of your response, create a 'Sources Referenced' section listing the document numbers and their paths.\n\n"
            f"CONTEXT DOCUMENTS:\n{context}\n"
            f"USER QUESTION: {question}\n\n"
            "CITED ANSWER:\n"
        )
        return prompt

    def generate_answer(self, question: str, chunks: List[Dict[str, Any]]) -> str:
        if not chunks:
            return "No relevant documentation chunks were found to answer this question."

        context_str, source_map = self._format_context(chunks)
        prompt_content = self._build_prompt(question, context_str)
        messages = [{"role": "user", "content": prompt_content}]

        full_content = ""
        print("\n[Nemotron Reasoning for Answer Generation]")
        print("-" * 50)

        try:
            for chunk in self.client.stream(messages):
                if chunk.additional_kwargs and "reasoning_content" in chunk.additional_kwargs:
                    print(chunk.additional_kwargs["reasoning_content"], end="", flush=True)

                if chunk.content:
                    if not full_content:
                        print("\n\n[Final Answer Generation]")
                        print("-" * 50)
                    full_content += chunk.content
                    print(chunk.content, end="", flush=True)

        except Exception as e:
            if "total_tokens" in str(e):
                if not full_content.strip():
                    try:
                        response = self.client.invoke(messages)
                        if response.additional_kwargs and "reasoning_content" in response.additional_kwargs:
                            print(response.additional_kwargs["reasoning_content"], end="", flush=True)

                        full_content = response.content
                        print("\n\n[Final Answer Generation]")
                        print("-" * 50)
                        print(full_content, end="", flush=True)
                    except Exception as invoke_e:
                        print(f"\nAPI Error during fallback generation: {str(invoke_e)}")
                        return "Failed to generate answer."
                else:
                    pass
            else:
                print(f"\nAPI Error during answer generation: {str(e)}")
                return "Failed to generate answer."

        print("\n" + "-" * 50)
        return full_content