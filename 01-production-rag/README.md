# AI Engineering Portfolio: Production-Grade RAG

## Core AI Engineering Competencies

This project is part of a broader "Production Lifecycle" portfolio, demonstrating the capacity to build reliable, scalable, and observable AI systems—moving beyond simple demos into production-ready engineering.

**Focus Area:** Architected a production-grade RAG pipeline with automated CI/CD evaluation and real-time observability.
**Message:** "I don't just build chatbots; I build reliable, monitorable systems."

## The Business Problem

Many AI applications fail in production because they hallucinate, lack domain-specific grounding, or degrade silently over time. This project solves for **reliability and trust** by ensuring every generated claim is explicitly backed by retrieved context, and introducing automated evaluation gates to catch regressions before they hit production.

## System Architecture

The "Ask My Doc" system employs a 3-phase progression to achieve production readiness:

1.  **Fundamentals (Data & Retrieval)**
    *   **Data Ingestion:** Supports PDF, Markdown, and Web pages.
    *   **Vector Storage:** Local ChromaDB instance for fast prototyping.
    *   **Chunking:** 500-800 token chunks with ~100 token overlap to maintain context across boundaries.
2.  **Production Quality (Hybrid Search & Reranking)**
    *   **Hybrid Retrieval:** Combines BM25 (keyword search) with Semantic (vector search) for higher recall.
    *   **Re-ranking:** Utilizes Cohere's Cross-Encoder to rescore and refine retrieved chunks.
    *   **LLMs:** Leverages **Google Gemini** for generation and embeddings, integrated with **Groq** for ultra-low latency specific tasks.
    *   **Citation Enforcement:** Strict prompting config to force the LLM to decline queries if context is missing, minimizing hallucinations.
3.  **Evaluation & CI/CD (RAGAS)**
    *   **Golden Dataset:** Curated Q&A pairs for continuous testing.
    *   **Metrics:** RAGAS "Faithfulness" checking ensures claims exactly match the retrieved context.
    *   **CI/CD:** GitHub Actions block deployments if evaluation metrics drop below a threshold.

## Technical Deep Dive: Trade-offs & Decisions

*   **ChromaDB over Cloud Vector DBs:** For this specific iteration, a local ChromaDB instance was chosen to reduce latency during development and avoid unnecessary cloud costs. It seamlessly transitions to a persistent client if scaled.
*   **Gemini + Groq Hybrid Generation:** Gemini provides excellent reasoning and context windows for complex document analysis, while Groq is introduced for scenarios requiring near-instantaneous token generation.
*   **The Hallucination Fix:** Initially, RAG systems often guess when context is sparse. The implementation of a Cross-Encoder (Cohere reranking) coupled with explicit Prompt Management forces the system to say "I don't know" rather than fabricating an answer, drastically cutting down false claims.

## Metrics to Watch

*(To be populated as the system is deployed and benchmarked)*
*   **Tokens/Sec Generation Rate:** Targeting sub-second latency with Groq.
*   **% Faithfulness (RAGAS):** Targeting >95% adherence to context.
*   **P95 Retrieval Latency:** Measuring vector search + reranking bottlenecks.

---

### Setup Instructions

1.  Clone the repository.
2.  Create a virtual environment: `python -m venv venv`
3.  Install dependencies: `pip install -r requirements.txt`
4.  Copy `.env.example` to `.env` and fill in your API keys:
    *   `GEMINI_API_KEY`
    *   `GROQ_API_KEY`
    *   `COHERE_API_KEY` (for reranking)
5.  Run the ingestion script to populate the vector database.
