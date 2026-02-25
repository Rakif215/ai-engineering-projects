import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.ingestion.loaders import DocumentIngestor
from src.ingestion.chunking import DocumentChunker
from src.storage.vectorstore import VectorStoreManager
from src.retrieval.hybrid_search import HybridRetrieverManager
from src.retrieval.reranking import RerankingManager
from src.generation.llm_chain import GenerationChain

def main():
    print("--- Ask My Doc: Production-Grade RAG ---")
    
    # 1. Provide a dummy PDF path or let the user know what to do
    # For testing, let's use the project requirements PDF if it's there
    sample_pdf = "project 1.pdf" 
    
    if not os.path.exists(sample_pdf):
        print(f"Please place a '{sample_pdf}' in the root directory to test ingestion.")
        return

    # Phase 1: Ingestion & Storage
    print("\n[1/4] Ingesting Document...")
    ingestor = DocumentIngestor()
    raw_docs = ingestor.load_pdf(sample_pdf)
    print(f"Loaded {len(raw_docs)} pages from {sample_pdf}.")
    
    print("\n[2/4] Chunking Document...")
    chunker = DocumentChunker()
    chunks = chunker.chunk_documents(raw_docs)
    print(f"Split into {len(chunks)} chunks.")
    
    print("\n[3/4] Initializing Vector Store...")
    vectorstore = VectorStoreManager(collection_name="portfolio_rag")
    vectorstore.add_documents(chunks)
    base_retriever = vectorstore.get_retriever(k=10) # Get more candidates for reranking
    
    # Phase 2: Production Quality Retrieval
    print("\n[4/4] Setting up Hybrid Search & Reranking...")
    hybrid_manager = HybridRetrieverManager(base_retriever, chunks)
    hybrid_retriever = hybrid_manager.get_retriever()
    
    rerank_manager = RerankingManager(hybrid_retriever, top_n=3)
    final_retriever = rerank_manager.get_retriever()
    
    # Generation Chain implementation configures the system to Groq for speed
    # or Gemini for nuanced Generation. We will default to Groq as requested in plan.
    print("\nInitializing Generation Pipeline (Groq)...")
    generation_chain = GenerationChain(llm_provider="groq")
    
    print("\n--- Pipeline Ready ---")
    
    # Interactive loop
    while True:
        question = input("\nAsk a question about the document (or 'quit'): ")
        if question.lower() in ['quit', 'exit', 'q']:
            break
            
        print("\nRetrieving and Generating...")
        try:
            answer = generation_chain.answer(question, final_retriever)
            print("\nAnswer:\n" + answer)
        except Exception as e:
            print(f"\nError generating answer: {e}")

if __name__ == "__main__":
    main()
