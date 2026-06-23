📚 DocuSense — Self-Correcting Developer Docs Assistant

DocuSense is an advanced, self-correcting Retrieval-Augmented Generation (RAG) system designed to answer technical questions by pulling directly from open-source documentation.

Unlike naive RAG implementations that hallucinate or fail on vague queries, DocuSense utilizes a multi-layered architecture including Hybrid Search, Multi-Query Expansion, and Corrective RAG (CRAG) to guarantee accurate, verifiable, and strictly cited answers.

🌟 Key Features & Architecture

This pipeline is built to enterprise standards, implementing several advanced RAG methodologies:

Hybrid Retrieval Backbone (RRF): Fuses semantic meaning (ChromaDB Vector Search) with exact-keyword matching (Rank-BM25) using Reciprocal Rank Fusion. This ensures variable names and code syntax are never lost in vector space.

Multi-Query Expansion: Before searching, the system uses an LLM to dynamically generate 3-4 highly technical variants of the user's prompt, maximizing the retrieval "catch radius" in parallel threads.

Corrective RAG (CRAG) Layer: An "LLM-as-a-Judge" rigorously evaluates retrieved chunks. If the context is deemed irrelevant to the query, the system automatically rejects it, rewrites the user's prompt using proper software engineering terminology, and triggers a second retrieval phase.

Strictly Grounded Citations: The generation module acts as a forensic technical writer, enforcing inline citations (e.g., [1], [2]) that map directly back to the original GitHub .md or .rst file paths.

🛠️ Tech Stack

Core: Python 3.10+

Orchestration: LangChain

Vector Database: ChromaDB

Keyword Indexing: Rank-BM25

Embeddings: sentence-transformers (all-MiniLM-L6-v2)

LLM Provider: NVIDIA AI Endpoints (Nemotron-3-Super-120B)

UI: Streamlit

🗂️ Supported Documentation

DocuSense is capable of ingesting both Markdown and reStructuredText formats. By default, the ingestion engine parses and chunks:

FastAPI

LangChain

Scikit-Learn

PyTorch

TensorFlow

🚀 Setup & Installation

1. Clone the Repository
```
git clone [https://github.com/YOUR_USERNAME/DocuSense.git](https://github.com/YOUR_USERNAME/DocuSense.git)
cd DocuSense
```

2. Set Up the Environment
```
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

3. Configure API Keys

Create a .env file in the root directory and add your NVIDIA API Key (or configure the project to use OpenAI/Anthropic via LangChain):
```
NVIDIA_API_KEY="your_api_key_here"
```

4. Build the Databases

First, download and chunk the raw documentation from GitHub:

```
python scripts/ingest_data.py
```

Next, build the ChromaDB Vector Index and the BM25 Keyword Index (this runs locally using CPU embeddings):

```
python scripts/build_index.py
```

5. Launch the Application

Start the Streamlit UI to interact with the pipeline:

```
streamlit run app/streamlit_app.py
```

🧠 Diagnostic Mode

The Streamlit interface includes a "Pipeline Diagnostics" expander. Use this during demos to visually prove to reviewers when the CRAG layer intervenes, how the Multi-Query engine expands questions, and how long the parallelized retrieval took.
