from langchain_classic.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from langchain_classic.retrievers.document_compressors import DocumentCompressorPipeline
import os

class RerankingManager:
    def __init__(self, base_retriever, top_n: int = 3):
        """
        Wraps a given base_retriever with a Cohere Cross-Encoder Reranker.
        Requires COHERE_API_KEY environment variable.
        """
        if "COHERE_API_KEY" not in os.environ:
            print("Warning: COHERE_API_KEY not found. Reranking might fail.")
            
        self.base_retriever = base_retriever
        self.top_n = top_n
        
        # Initialize the Cohere Reranker
        self.compressor = CohereRerank(
            model="rerank-english-v3.0",
            top_n=self.top_n
        )
        
        # Create the compression retriever
        self.compression_retriever = ContextualCompressionRetriever(
            base_compressor=self.compressor,
            base_retriever=self.base_retriever
        )

    def retrieve(self, query: str):
        """
        Executes reranked retrieval based on the base retriever.
        """
        return self.compression_retriever.invoke(query)
        
    def get_retriever(self):
        """
        Returns the wrapped contextual compression retriever.
        """
        return self.compression_retriever
