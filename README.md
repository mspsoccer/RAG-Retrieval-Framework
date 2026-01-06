# Overview
This project is a high-accuracy Retrieval-Augmented Generation (RAG) system designed to bridge the gap between "fuzzy" vector searches and the strict data integrity required for technical research.

While the system is completely data-agnostic—capable of ingestng everything from legal contracts to software documentation—I have primarily benchmarked and tested its performance using SARS-CoV-2 genomic datasets from the FIRE Research Program.

# The Challenge
In bioinformatics, missing a single mutation coordinate or misidentifying an accession number can invalidate a phylogenetic analysis. Standard RAG systems often suffer from "contextual noise," where the LLM is overwhelmed by irrelevant bunches of text.

To solve this, I implemented a Two-Stage Retrieval Architecture:

'''Stage 1: The Librarian (Vector Search): Uses ChromaDB and OpenAIEmbeddings to quickly sweep the dataset and find the 10 most mathematically similar chunks.'''


'''Stage 2: The Judge (Flashrank Reranker): A lightweight Cross-Encoder model that performs deep semantic analysis on those 10 chunks. It re-scores them to ensure the specific data needed (like a mutation presence) is moved to the top of the context window.'''

# Benchmarking with Genomic Data
I validated the framework using two complex datasets:


Phylogenetic Tree Legend: 101 viral sequences from November 2020, aligned to reference NC045512.


Sequence Metadata Table: Mapping accession numbers (e.g., MW653491) to geographic locations and mutation presence.


Performance results: The system successfully mapped the B.1.1.7 (Alpha) variant and its specific indicators—the P681H mutation and 69/70 deletion—without hallucination.

# Tech Stack
'''Framework: LangChain

Vector Store: ChromaDB (with persistence)

Embeddings/LLM: OpenAI GPT-4o

Reranker: Flashrank

Environment: Python-dotenv for secure API management'''
