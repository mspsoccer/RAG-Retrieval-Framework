import os
import glob
from dotenv import load_dotenv 

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import Chroma
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import FlashrankRerank
from langchain.chains import RetrievalQA

class BioRAGPipeline:
    def __init__(self, data_dir="sample_data", persist_dir="./chroma_db"):
        load_dotenv("key.env")
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("API Key not found. Check your key.env file.")
        
        self.data_dir = data_dir
        self.persist_dir = persist_dir
        self.rag_chain = None

    def load_and_process_documents(self):
        pdf_files = glob.glob(os.path.join(self.data_dir, "*.pdf"))
        
        if not pdf_files:
            print(f"Warning: No PDF files found in the '{self.data_dir}' directory.")
            return []

        documents = []
        for file in pdf_files:
            loader = PyPDFLoader(file)
            documents.extend(loader.load())
            
        print(f"Loaded {len(pdf_files)} document(s).")

        # large chunk size with 150 overlap 
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150)
        chunks = text_splitter.split_documents(documents)
        print(f"Split into {len(chunks)} chunks.")
        
        return chunks

    def build_system(self):
        """Constructs the vector database, reranker, and execution chain."""
        chunks = self.load_and_process_documents()
        
        if not chunks:
            print("System build aborted: No data to process.")
            return

        print("Building vector database with ChromaDB...")
        vector_db = Chroma.from_documents(
            documents=chunks,
            embedding=OpenAIEmbeddings(openai_api_key=self.api_key), 
            persist_directory=self.persist_dir
        )

        # TwoStage Filtering
        base_retriever = vector_db.as_retriever(search_kwargs={"k": 10})
        compressor = FlashrankRerank()

        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor, 
            base_retriever=base_retriever
        )

        # RAG 
        llm = ChatOpenAI(model_name="gpt-4o", temperature=0, openai_api_key=self.api_key)
        self.rag_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=compression_retriever,
            return_source_documents=True 
        )
        print("System ready.\n")

    def query(self, user_question):
        """Executes a query against the RAG chain and formats the output."""
        if not self.rag_chain:
            print("Error: RAG chain is not built. Call build_system() first.")
            return

        print(f"Querying: '{user_question}'\nProcessing...")
        result = self.rag_chain.invoke(user_question)

        print("-" * 50)
        print(f"AI Answer: {result['result']}")
        print("-" * 50)
        print("Sources Used:")
        
        for doc in result["source_documents"]:
            source_name = os.path.basename(doc.metadata.get('source', 'Unknown Document'))
            page_num = doc.metadata.get('page', 1)
            print(f"- {source_name} (Page {page_num})")


if __name__ == "__main__":
    os.makedirs("sample_data", exist_ok=True)
    # run
    rag_system = BioRAGPipeline()
    rag_system.build_system()
    
    # Test execution
    test_query = "Which accession numbers are associated with the P681H mutation?"
    rag_system.query(test_query)
