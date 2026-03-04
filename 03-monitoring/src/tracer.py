"""
tracer.py — Langfuse-based tracing for the RAG pipeline.

Instruments every stage of the pipeline:
  - Document retrieval (chunks retrieved + scores)
  - Reranker score delta (before vs. after order)
  - Final prompt sent to the LLM
  - Token counts per step
  - End-to-end latency per request

Usage:
    from src.tracer import RAGTracer
    tracer = RAGTracer()
    with tracer.trace("my-session") as span:
        span.log_retrieval(chunks)
        span.log_reranking(before, after)
        span.log_generation(prompt, response, tokens)
"""

import os
import time
import json
from datetime import datetime
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
load_dotenv()


# ─── Langfuse Client Setup ───────────────────────────────────────────
try:
    from langfuse import Langfuse
    _langfuse = Langfuse(
        public_key=os.getenv("LANGFUSE_PUBLIC_KEY", ""),
        secret_key=os.getenv("LANGFUSE_SECRET_KEY", ""),
        host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
    )
    LANGFUSE_ENABLED = bool(os.getenv("LANGFUSE_PUBLIC_KEY"))
except Exception:
    _langfuse = None
    LANGFUSE_ENABLED = False


class TraceSpan:
    """Wraps a single end-to-end RAG request trace."""

    def __init__(self, session_id: str, tracer: "RAGTracer"):
        self.session_id = session_id
        self.tracer = tracer
        self.start_time = time.perf_counter()
        self.events: List[Dict[str, Any]] = []
        self.token_counts: Dict[str, int] = {}
        self.cost_usd: float = 0.0
        self._trace = None
        self._generation_span = None

        if LANGFUSE_ENABLED and _langfuse:
            self._trace = _langfuse.trace(
                name="rag-request",
                session_id=session_id,
                tags=["production-rag"],
            )

    # ── Retrieval ────────────────────────────────────────────────────
    def log_retrieval(self, query: str, chunks: List[Any]):
        """Log what chunks were retrieved from the vector store."""
        chunk_data = []
        for i, chunk in enumerate(chunks):
            source = os.path.basename(chunk.metadata.get("source", "unknown"))
            page = chunk.metadata.get("page", "?")
            chunk_data.append({
                "rank": i + 1,
                "source": source,
                "page": page,
                "content_preview": chunk.page_content[:150] + "...",
            })

        self.events.append({"step": "retrieval", "query": query, "chunks": chunk_data})

        if self._trace:
            self._trace.span(
                name="hybrid-retrieval",
                input={"query": query},
                output={"chunks_retrieved": len(chunks), "chunks": chunk_data},
            )

    # ── Reranking ────────────────────────────────────────────────────
    def log_reranking(self, before_chunks: List[Any], after_chunks: List[Any]):
        """Log how the reranker changed ordering."""
        def chunk_id(c):
            return f"{os.path.basename(c.metadata.get('source','?'))}:p{c.metadata.get('page','?')}"

        before_ids = [chunk_id(c) for c in before_chunks]
        after_ids = [chunk_id(c) for c in after_chunks]

        reorder_delta = {
            "before": before_ids,
            "after": after_ids,
            "dropped": [x for x in before_ids if x not in after_ids],
            "promoted": [
                x for x in after_ids
                if x in before_ids and after_ids.index(x) < before_ids.index(x)
            ],
        }
        self.events.append({"step": "reranking", "delta": reorder_delta})

        if self._trace:
            self._trace.span(
                name="cohere-reranker",
                input={"candidates": len(before_chunks)},
                output={"selected": len(after_chunks), "reorder_delta": reorder_delta},
            )

    # ── Generation ───────────────────────────────────────────────────
    def log_generation(
        self,
        prompt: str,
        response: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        model: str = "llama-3.1-8b-instant",
        provider: str = "groq",
    ):
        """Log the final prompt, response, and token counts."""
        self.token_counts = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
        }

        # Cost estimation
        self.cost_usd = self._estimate_cost(provider, model, prompt_tokens, completion_tokens)

        self.events.append({
            "step": "generation",
            "model": model,
            "provider": provider,
            "tokens": self.token_counts,
            "cost_usd": self.cost_usd,
            "response_preview": response[:300],
        })

        if self._trace:
            self._generation_span = self._trace.generation(
                name="llm-generation",
                model=model,
                input=prompt,
                output=response,
                usage={
                    "prompt_tokens": prompt_tokens,
                    "completion_tokens": completion_tokens,
                    "total_tokens": prompt_tokens + completion_tokens,
                },
            )

    # ── Quality ──────────────────────────────────────────────────────
    def log_citation_outcome(self, is_grounded: bool):
        """Track whether the response was grounded (cited) or declined."""
        self.events.append({"step": "citation_check", "is_grounded": is_grounded})
        if self._trace:
            self._trace.score(
                name="is_grounded",
                value=1.0 if is_grounded else 0.0,
                comment="Citation enforcement check",
            )

    # ── Finalize ─────────────────────────────────────────────────────
    def finish(self) -> Dict[str, Any]:
        """Complete the trace and return a metrics summary."""
        elapsed_ms = (time.perf_counter() - self.start_time) * 1000

        summary = {
            "session_id": self.session_id,
            "timestamp": datetime.utcnow().isoformat(),
            "latency_ms": round(elapsed_ms, 2),
            "token_counts": self.token_counts,
            "cost_usd": self.cost_usd,
            "events": self.events,
        }

        if _langfuse:
            _langfuse.flush()

        return summary

    # ── Helpers ──────────────────────────────────────────────────────
    @staticmethod
    def _estimate_cost(provider: str, model: str, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost in USD based on published pricing."""
        # Groq pricing (per million tokens) — as of early 2025
        GROQ_PRICING = {
            "llama-3.1-8b-instant": {"prompt": 0.05, "completion": 0.08},
            "llama-3.1-70b-versatile": {"prompt": 0.59, "completion": 0.79},
            "mixtral-8x7b-32768": {"prompt": 0.24, "completion": 0.24},
        }
        # Gemini pricing (per million tokens)
        GEMINI_PRICING = {
            "gemini-2.5-flash": {"prompt": 0.075, "completion": 0.30},
            "gemini-1.5-pro": {"prompt": 1.25, "completion": 5.00},
        }

        pricing = GROQ_PRICING if provider == "groq" else GEMINI_PRICING
        rates = pricing.get(model, {"prompt": 0.0, "completion": 0.0})

        cost = (prompt_tokens * rates["prompt"] + completion_tokens * rates["completion"]) / 1_000_000
        return round(cost, 8)

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.finish()


class RAGTracer:
    """Factory for creating RAG trace spans."""

    def trace(self, session_id: Optional[str] = None) -> TraceSpan:
        sid = session_id or f"session-{int(time.time() * 1000)}"
        return TraceSpan(session_id=sid, tracer=self)
