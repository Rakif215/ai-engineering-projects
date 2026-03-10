import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
# Make sure GROQ_API_KEY is in your .env file
os.environ["GROQ_API_KEY"] = os.getenv("GROQ_API_KEY", "")

from src.ingestion.loaders import DocumentIngestor
from src.ingestion.chunking import DocumentChunker
from src.storage.vectorstore import VectorStoreManager
from src.retrieval.hybrid_search import HybridRetrieverManager
from src.retrieval.reranking import RerankingManager
from src.generation.llm_chain import GenerationChain

app = FastAPI()
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_headers=["*"], allow_methods=["*"])

ingestor = DocumentIngestor()
chunker = DocumentChunker()
docs = ingestor.load_pdf("Clinical_Guidelines_2024.pdf")
chunks = chunker.chunk_documents(docs)

vectorstore = VectorStoreManager(collection_name="fastapi_medical_demo")
vectorstore.add_documents(chunks)

# Standard RAG: Semantic search only, retrieving just 1 chunk
naive_retriever = vectorstore.get_retriever(k=1)

# Production RAG (Veritas): Hybrid Search + Cross-Encoder Reranking
base_prod_retriever = vectorstore.get_retriever(k=5)
hybrid_manager = HybridRetrieverManager(base_prod_retriever, chunks)
hybrid_retriever = hybrid_manager.get_retriever()
rerank_manager = RerankingManager(hybrid_retriever, top_n=3)
prod_retriever = rerank_manager.get_retriever()

class NaiveGenerationChain(GenerationChain):
    def _build_prompt(self):
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        return ChatPromptTemplate.from_messages([
            ("system", "You are a confident medical assistant. Answer the question using the context. If the context is missing info, confidently guess or predict an answer based on general medical knowledge. Don't mention the context."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

class ProdGenerationChain(GenerationChain):
    def _build_prompt(self):
        from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
        return ChatPromptTemplate.from_messages([
            ("system", "You are Veritas, a medical QA system. You strictly adhere to grounding rules. If the exact answer is not explicitly detailed in the context, you MUST refuse to answer. Say: 'I cannot answer this question. The provided clinical context does not contain sufficient information.' Do not guess."),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}")
        ])

naive_chain = NaiveGenerationChain(llm_provider="groq")
prod_chain = ProdGenerationChain(llm_provider="groq")

class QueryReq(BaseModel):
    query: str

class MockRetriever:
    def __init__(self, docs):
        self.docs = docs
    def invoke(self, q):
        return self.docs

@app.post("/query/naive")
def q_naive(req: QueryReq):
    try:
        docs = naive_retriever.invoke(req.query)
        ans = naive_chain.answer(req.query, MockRetriever(docs))
        
        trace = []
        for d in docs:
            trace.append({"content": d.page_content, "score": "N/A"})
    except Exception as e:
        ans = f"[NAIVE RAG CRASH] Error: {str(e)}"
        trace = []
    return {"response": ans, "trace": trace}

@app.post("/query/prod")
def q_prod(req: QueryReq):
    try:
        docs = prod_retriever.invoke(req.query)
        ans = prod_chain.answer(req.query, MockRetriever(docs))
        
        trace = []
        for d in docs:
            score = d.metadata.get("relevance_score", "N/A")
            if isinstance(score, float):
                score = round(score, 3)
            trace.append({"content": d.page_content, "score": score})
    except Exception as e:
        ans = f"System Failure: {str(e)}"
        trace = []
    return {"response": ans, "trace": trace}
