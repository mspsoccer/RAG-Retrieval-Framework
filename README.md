This is a high-accuracy Retrieval-Augmented Generation (RAG) system built to handle dense, technical documents where data integrity is critical. While the pipeline can process any text, I benchmarked it against SARS-CoV-2 genomic datasets.

# The Problem
Standard RAG systems suffer from "contextual noise." If an LLM is flooded with vaguely related text, it hallucinates. In bioinformatics, missing a single mutation coordinate or misidentifying an accession number completely invalidates the analysis.

# The Solution
To fix this, I built a two-stage retrieval pipeline:

Stage 1 (Vector Search): ChromaDB and OpenAI Embeddings execute a fast initial sweep to pull the 10 most relevant text chunks.

Stage 2 (Reranking): A lightweight Flashrank Cross-Encoder performs a deep semantic analysis on those 10 chunks. It re-scores them to guarantee the exact data needed is prioritized at the top of the context window.# Overview
secure API management```
