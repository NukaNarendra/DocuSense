import os
import re
import json
import uuid
import hashlib
import urllib.request
import zipfile
import shutil
from pathlib import Path
from typing import List, Dict, Any
from langchain_text_splitters import RecursiveCharacterTextSplitter


class Config:
    DATA_RAW_DIR = os.path.join(os.getcwd(), "data", "raw")
    DATA_PROCESSED_DIR = os.path.join(os.getcwd(), "data", "processed")
    CHUNK_SIZE = 1000
    CHUNK_OVERLAP = 200

    TARGET_LIBRARIES = [
        {
            "name": "fastapi",
            "url": "https://github.com/fastapi/fastapi/archive/refs/heads/master.zip",
            "docs_path": "fastapi-master/docs/en/docs",
            "extensions": [".md", ".mdx"]
        },
        {
            "name": "langchain",
            "url": "https://github.com/langchain-ai/docs/archive/refs/heads/main.zip",
            "docs_path": "docs-main",
            "extensions": [".md", ".mdx"]
        },
        {
            "name": "scikit-learn",
            "url": "https://github.com/scikit-learn/scikit-learn/archive/refs/heads/main.zip",
            "docs_path": "scikit-learn-main/doc",
            "extensions": [".rst"]
        },
        {
            "name": "pytorch",
            "url": "https://github.com/pytorch/pytorch/archive/refs/heads/main.zip",
            "docs_path": "pytorch-main/docs/source",
            "extensions": [".rst", ".md"]
        },
        {
            "name": "tensorflow",
            "url": "https://github.com/tensorflow/docs/archive/refs/heads/master.zip",
            "docs_path": "docs-master/site/en",
            "extensions": [".md", ".ipynb"]
        }
    ]


class TextCleaner:
    @staticmethod
    def clean_markdown(text: str) -> str:
        text = re.sub(r'<!--.*?-->', '', text, flags=re.DOTALL)
        text = re.sub(r'<[^>]+>', '', text)
        text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
        text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r'^\s*$', '', text, flags=re.MULTILINE)
        return text.strip()

    @staticmethod
    def clean_rst(text: str) -> str:
        text = re.sub(r'\.\. .*?\n', '', text)
        text = re.sub(r':[a-zA-Z0-9_-]+:`.*?`', '', text)
        text = re.sub(r'\n{3,}', '\n\n', text)
        return text.strip()

    @staticmethod
    def clean_general(text: str) -> str:
        text = text.replace('\r\n', '\n')
        text = text.replace('\t', '    ')
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)
        return text


class DocProcessor:
    def __init__(self):
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=Config.CHUNK_SIZE,
            chunk_overlap=Config.CHUNK_OVERLAP,
            length_function=len,
            separators=["\n\n", "\n", " ", ""]
        )

    def extract_title_from_md(self, text: str) -> str:
        lines = text.split('\n')
        for line in lines:
            if line.startswith('# '):
                return line.replace('# ', '').strip()
        return "Untitled Document"

    def process_file(self, file_path: str, lib_name: str) -> List[Dict[str, Any]]:
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            try:
                with open(file_path, 'r', encoding='latin-1') as f:
                    content = f.read()
            except Exception:
                return []

        ext = Path(file_path).suffix.lower()
        content = TextCleaner.clean_general(content)

        if ext in ['.md', '.mdx']:
            content = TextCleaner.clean_markdown(content)
            title = self.extract_title_from_md(content)
        elif ext == '.rst':
            content = TextCleaner.clean_rst(content)
            title = "RST Document"
        else:
            title = "Document"

        if len(content.strip()) < 50:
            return []

        chunks = self.text_splitter.split_text(content)
        processed_chunks = []

        for index, chunk in enumerate(chunks):
            chunk_id = hashlib.md5(f"{file_path}_{index}".encode()).hexdigest()
            relative_path = str(Path(file_path).relative_to(Config.DATA_RAW_DIR))

            chunk_data = {
                "chunk_id": chunk_id,
                "library": lib_name,
                "source_file": relative_path,
                "title": title,
                "chunk_index": index,
                "content": chunk.strip()
            }
            processed_chunks.append(chunk_data)

        return processed_chunks


class IngestionPipeline:
    def __init__(self):
        self.processor = DocProcessor()
        self.setup_directories()

    def setup_directories(self):
        os.makedirs(Config.DATA_RAW_DIR, exist_ok=True)
        os.makedirs(Config.DATA_PROCESSED_DIR, exist_ok=True)

    def download_and_extract(self, lib_config: Dict[str, Any]) -> str:
        lib_name = lib_config["name"]
        url = lib_config["url"]
        zip_path = os.path.join(Config.DATA_RAW_DIR, f"{lib_name}.zip")
        extract_path = os.path.join(Config.DATA_RAW_DIR, lib_name)
        docs_target_path = os.path.join(extract_path, lib_config["docs_path"])

        if not os.path.exists(docs_target_path):
            print(f"Downloading {lib_name} from {url}...")
            req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
            with urllib.request.urlopen(req) as response, open(zip_path, 'wb') as out_file:
                shutil.copyfileobj(response, out_file)

            print(f"Extracting {lib_name} docs specifically...")
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                target_prefix = lib_config["docs_path"].replace("\\", "/")

                # Targeted extraction: Only extract files that belong to the docs path
                # This prevents MAX_PATH OS errors from extracting deep nested test folders
                for member in zip_ref.namelist():
                    if member.startswith(target_prefix):
                        zip_ref.extract(member, extract_path)

            if os.path.exists(zip_path):
                os.remove(zip_path)

        return docs_target_path

    def collect_files(self, docs_path: str, extensions: List[str]) -> List[str]:
        target_files = []
        if not os.path.exists(docs_path):
            print(f"Path not found: {docs_path}")
            return target_files

        for root, dirs, files in os.walk(docs_path):
            for file in files:
                if any(file.endswith(ext) for ext in extensions):
                    target_files.append(os.path.join(root, file))
        return target_files

    def run(self):
        total_chunks_system = 0

        for lib in Config.TARGET_LIBRARIES:
            lib_name = lib["name"]
            output_json_path = os.path.join(Config.DATA_PROCESSED_DIR, f"{lib_name}_chunks.json")

            if os.path.exists(output_json_path):
                print(f"Skipping {lib_name}, already processed.")
                continue

            try:
                docs_path = self.download_and_extract(lib)
                files_to_process = self.collect_files(docs_path, lib["extensions"])

                print(f"Found {len(files_to_process)} files for {lib_name}")

                library_chunks = []
                for file_path in files_to_process:
                    file_chunks = self.processor.process_file(file_path, lib_name)
                    library_chunks.extend(file_chunks)

                if library_chunks:
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(library_chunks, f, indent=2, ensure_ascii=False)
                    print(f"Successfully saved {len(library_chunks)} chunks for {lib_name}")
                    total_chunks_system += len(library_chunks)
                else:
                    print(f"Warning: No valid chunks generated for {lib_name}")

            except Exception as e:
                print(f"Error processing library {lib_name}: {str(e)}")

        print(f"\nPhase 1 Complete. Total chunks processed and saved: {total_chunks_system}")


if __name__ == "__main__":
    pipeline = IngestionPipeline()
    pipeline.run()