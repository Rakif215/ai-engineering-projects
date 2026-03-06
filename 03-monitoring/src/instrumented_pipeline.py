"""
instrumented_pipeline.py — The full RAG pipeline from Project 1,
wrapped with tracing and metrics collection.

This is the single entry point for monitored RAG requests.
Import this instead of running main.py from Project 1 directly.
"""

import os
import sys
import time
import uuid
from typing import Optional, Generator

from dotenv import load_dotenv
load_dotenv()

# Add Project 1 to path so we can reuse its modules
_PROJECT1 = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "01-production-rag"))
if _PROJECT1 not in sys.path:
    sys.path.insert(0, _PROJECT1)

from src.ingestion.loaders import DocumentIngestor
from src.ingestion.chunking import DocumentChunker
from src.storage.vectorstore import VectorStoreManager
from src.retrieval.hybrid_search import HybridRetrieverManager
from src.retrieval.reranking import RerankingManager
from src.generation.llm_chain import GenerationChain

from .tracer import RAGTracer
from .metrics import MetricsRecorder


class InstrumentedRAGPipeline:
    """
    Production RAG pipeline with full observability instrumentation.

    Every request is:
      1. Traced (Langfuse) — retrieval + reranking + generation spans
      2. Measured (SQLite) — latency P50/P95, cost, citation coverage
    """

    def __init__(self, llm_provider: str = "groq"):
        self.llm_provider = llm_provider
        self.tracer = RAGTracer()
        self.recorder = MetricsRecorder()
        self._retriever = None
        self._generation_chain = None
        self._all_chunks = []

    # ── Pipeline Setup ────────────────────────────────────────────────
    def build(self, file_paths: list[str]):
        """Ingest documents and build the retrieval pipeline."""
        ingestor = DocumentIngestor()
        chunker = DocumentChunker()
        all_docs = []

        for path in file_paths:
            suffix = os.path.splitext(path)[1].lower()
            if suffix == ".pdf":
                all_docs.extend(ingestor.load_pdf(path))
            elif suffix == ".md":
                all_docs.extend(ingestor.load_markdown(path))

        self._all_chunks = chunker.chunk_documents(all_docs)

        vectorstore = VectorStoreManager(collection_name="monitored_rag")
        vectorstore.add_documents(self._all_chunks)
        base_retriever = vectorstore.get_retriever(k=10)

        hybrid = HybridRetrieverManager(base_retriever, self._all_chunks)
        self._retriever = RerankingManager(hybrid.get_retriever(), top_n=3).get_retriever()

        self._generation_chain = GenerationChain(llm_provider=self.llm_provider)
        return self

    # ── Monitored Query ───────────────────────────────────────────────
    def query(self, question: str, session_id: Optional[str] = None) -> dict:
        """
        Run a single RAG query with full tracing and metrics.

        Returns:
          {
            "answer": str,
            "session_id": str,
            "latency_ms": float,
            "cost_usd": float,
            "is_grounded": bool,
            "chunks_used": int,
          }
        """
        session_id = session_id or str(uuid.uuid4())
        t0 = time.perf_counter()
        is_error = False
        answer = ""

        span = self.tracer.trace(session_id)

        try:
            # Step 1 — Retrieval
            raw_chunks = self._retriever.invoke(question)
            span.log_retrieval(question, raw_chunks)

            # Step 2 — Generation (with streaming collected into string)
            full_response = ""
            for token in self._generation_chain.stream(question, self._retriever):
                full_response += token
            answer = full_response

            # Step 3 — Citation check
            is_grounded = "cannot answer" not in answer.lower()
            span.log_citation_outcome(is_grounded)

            # Step 4 — Token estimation (tiktoken-based)
            prompt_tok, comp_tok = self._estimate_tokens(question, answer)
            span.log_generation(
                prompt=question,
                response=answer,
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                model=self._generation_chain.llm.model_name
                if self.llm_provider == "groq" else "gemini-2.5-flash",
                provider=self.llm_provider,
            )

        except Exception as e:
            is_error = True
            answer = f"⚠️ Error: {e}"
            is_grounded = False

        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            trace_data = span.finish()

            self.recorder.record(
                session_id=session_id,
                latency_ms=latency_ms,
                prompt_tokens=trace_data.get("token_counts", {}).get("prompt_tokens", 0),
                completion_tokens=trace_data.get("token_counts", {}).get("completion_tokens", 0),
                cost_usd=trace_data.get("cost_usd", 0.0),
                is_grounded=is_grounded,
                is_error=is_error,
                model="llama-3.1-8b-instant" if self.llm_provider == "groq" else "gemini-2.5-flash",
                provider=self.llm_provider,
                question=question,
                answer_preview=answer[:500],
            )

        return {
            "answer": answer,
            "session_id": session_id,
            "latency_ms": round(latency_ms, 1),
            "cost_usd": trace_data.get("cost_usd", 0.0),
            "is_grounded": is_grounded,
            "is_error": is_error,
        }

    def stream_query(self, question: str, session_id: Optional[str] = None) -> Generator[str, None, dict]:
        """Stream tokens while still recording metrics on completion."""
        session_id = session_id or str(uuid.uuid4())
        t0 = time.perf_counter()
        full_response = ""
        is_error = False

        try:
            for token in self._generation_chain.stream(question, self._retriever):
                full_response += token
                yield token
        except Exception as e:
            is_error = True
            full_response = f"⚠️ Error: {e}"
            yield full_response
        finally:
            latency_ms = (time.perf_counter() - t0) * 1000
            is_grounded = "cannot answer" not in full_response.lower()
            prompt_tok, comp_tok = self._estimate_tokens(question, full_response)
            cost = span_cost = self.tracer.trace(session_id)._estimate_cost(
                self.llm_provider,
                "llama-3.1-8b-instant" if self.llm_provider == "groq" else "gemini-2.5-flash",
                prompt_tok, comp_tok
            )
            self.recorder.record(
                session_id=session_id,
                latency_ms=latency_ms,
                prompt_tokens=prompt_tok,
                completion_tokens=comp_tok,
                cost_usd=cost,
                is_grounded=is_grounded,
                is_error=is_error,
                model="llama-3.1-8b-instant",
                provider=self.llm_provider,
                question=question,
                answer_preview=full_response[:500],
            )

    @staticmethod
    def _estimate_tokens(prompt: str, response: str) -> tuple[int, int]:
        """Simple whitespace-based token estimation (no tiktoken required)."""
        return len(prompt.split()), len(response.split())
