import os
from dotenv import load_dotenv 
# loads API key
load_dotenv("key.env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("API Key not found. Check your key.env file.")
# pip install langchain langchain-openai langchain-community chromadb flashrank pypdf
# pip install python-dotenv


from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import RetrievalQA

# Loading   Research Data
loader_tree = PyPDFLoader("asn5 tree visualization with legend.pdf")
loader_table = PyPDFLoader("Sequence table - sequences (1).pdf")
documents = loader_tree.load() + loader_table.load() 

# Recursive Character Splitting
# 1000 chunk size with 150 overlap to keep explanations together
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
chunks = text_splitter.split_documents(documents)

# Creating the Vector Space
# Using embedding model to convert text to floats and saves them into the 'persist_directory'
vector_db = Chroma.from_documents(
    documents=chunks,
    embedding=OpenAIEmbeddings(openai_api_key=api_key), 
    persist_directory="./chroma_db"
)

#Two-Stage Filtering
# fetches 10 neighbors based on cosine similarity
base_retriever = vector_db.as_retriever(search_kwargs={"k": 10})

# Cross-encoder model with flashrankRerank
compressor = FlashrankRerank()

# Executes 'base_retriever' first then passes results to 'compressor' 
# to re-check relevance scores
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor, 
    base_retriever=base_retriever
)

# RAG Chain
# Forcing compressed chunks (10 max) in prompt for LLM
# Explicitly pass the key and return source documents to prevent crashes
llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key)
rag_chain = RetrievalQA.from_chain_type(
    llm=ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=api_key),
    chain_type="stuff",
    retriever=compression_retriever,
    return_source_documents=True 
)

# test exec
query = "Which accession numbers are associated with the P681H mutation?"
result = rag_chain.invoke(query)

print(f"\nAI Answer: {result['result']}")
print("\nSources Used:")
for doc in result["source_documents"]:
    print(f"- {doc.metadata['source']} (Page {doc.metadata.get('page', 1)})")
