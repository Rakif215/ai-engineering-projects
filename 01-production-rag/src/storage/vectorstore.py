import chromadb
from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
import os

class VectorStoreManager:
    def __init__(self, collection_name: str = "ask_my_doc"):
        # We will use a local directory for ChromaDB
        self.persist_directory = os.path.join(os.getcwd(), "chroma_db")
        os.makedirs(self.persist_directory, exist_ok=True)
        self.collection_name = collection_name
        
        # Ensure GEMINI API KEY is available for embeddings
        # Using a reliable local HuggingFace embedding model instead of Gemini for storage
        # to ensure it works without API rate limits or 404 endpoint issues.
        self.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        
        # Initialize Langchain Chroma wrapper
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory,
        )

    def add_documents(self, chunks):
        """
        Adds chunked documents to the Chromadb collection.
        """
        if not chunks:
            return []
        return self.vector_store.add_documents(documents=chunks)
        
    def get_retriever(self, k: int = 5):
        """
        Returns a basic Top-K retriever from the vector store.
        """
        return self.vector_store.as_retriever(search_kwargs={"k": k})
