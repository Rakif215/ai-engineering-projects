class BasicRetriever:
    def __init__(self, vectorstore_manager):
        self.vectorstore_manager = vectorstore_manager
        
    def retrieve(self, query: str, top_k: int = 5):
        """
        Executes a basic Top-K semantic search in the vector database.
        Returns a list of retrieved documents.
        """
        retriever = self.vectorstore_manager.get_retriever(k=top_k)
        docs = retriever.invoke(query)
        return docs
