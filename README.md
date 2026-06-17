# RAG: Contextual Retrieval Framework

A Retrieval-Augmented Generation (RAG) system built to handle dense, technical documents where data integrity is critical. While the pipeline can process any text, it is benchmarked against SARS-CoV-2 genomic datasets.

## The Problem
Standard RAG systems suffer from "contextual noise". In areas like bioinformatics, missing a single mutation coordinate or misidentifying an accession number completely invalidates the analysis.

## The Solution
To eliminate issues and guarantee data accuracy, this system uses a two-stage retrieval pipeline:
* **Stage 1 (Vector Search):** ChromaDB and OpenAI Embeddings execute a fast initial sweep to pull the 10 most relevant text chunks based on cosine similarity.
* **Stage 2 (Reranking):** A lightweight Flashrank Cross-Encoder performs a deep semantic analysis on the initial chunks. It re-scores and compresses them, guaranteeing the exact data needed is prioritized at the top of the context window.

## Tech Stack
* **Languages:** Python
* **Frameworks/Libraries:** LangChain, ChromaDB, Flashrank, PyPDF
* **Models:** OpenAI `gpt-4o`, OpenAI Embeddings

## Pipeline Architecture
1. **Document Ingestion:** Recursively splits large genomic PDFs into 1000-character chunks with a 150-character overlap to preserve contextual explanations.
2. **Vectorization:** Converts text into float embeddings and persists them locally using ChromaDB.
3. **Contextual Compression:** Executes the two-stage filtering via LangChain's `ContextualCompressionRetriever`.
4. **Generation:** Forces the compressed, highly-relevant chunks into the LLM prompt, returning both the precise answer and the source document metadata for verification.

## How to Run Locally

1. **Clone the repository:**
   `git clone https://github.com/mspsoccer/rag-contextual-retrieval.git`
2. **Install dependencies:**
   `pip install langchain langchain-openai langchain-community chromadb flashrank pypdf python-dotenv`
3. **Environment Setup:**
   Create a `key.env` file in the root directory and add your API key:
   `OPENAI_API_KEY=your_key_here`
4. **Add Data:**
   Place your target PDF files into the working directory (or update the loader paths in the script).
5. **Execute:**
   `python main.py`
