from langchain_classic.retrievers.ensemble import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever

class HybridRetrieverManager:
    def __init__(self, vectorstore_retriever, documents):
        """
        Takes a base vectorstore retriever (semantic) and raw chunked documents
        to initialize the BM25 (keyword) retriever.
        """
        self.vectorstore_retriever = vectorstore_retriever
        
        # Initialize BM25 retriever
        self.bm25_retriever = BM25Retriever.from_documents(documents)
        self.bm25_retriever.k = 5 # Default top k for bm25
        
        # Combine both giving them a specific weight. 
        # Typically semantic search handles nuance better, keyword handles exact names well.
        self.ensemble_retriever = EnsembleRetriever(
            retrievers=[self.bm25_retriever, self.vectorstore_retriever],
            weights=[0.4, 0.6]
        )

    def retrieve(self, query: str):
        """
        Executes a hybrid search returning combined results.
        """
        return self.ensemble_retriever.invoke(query)
        
    def get_retriever(self):
        """
        Returns the underlying ensemble retriever object for pipeline chains.
        """
        return self.ensemble_retriever

