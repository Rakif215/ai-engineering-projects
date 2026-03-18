# AI Engineering Projects

A collection of 5 production-grade AI systems focusing on reliability, observability, and evaluation. Most AI tutorials stop at building a chatbot; these projects explore the engineering needed to deploy them safely.

## Table of Contents

### 1. [Veritas: Production-Grade RAG](./01-production-rag/)
A specialized "Ask My Docs" system built to refuse answers when context is missing, preventing hallucination in high-stakes domains (like healthcare).
- **Core:** Hybrid Retrieval (BM25 + Semantic), Cross-Encoder Reranking, Strict Citation Grounding.
- **Stack:** Llama-3.1 (Groq) / Gemini, ChromaDB, HuggingFace embeddings, Cohere Rerank.
- **Frontend:** React + Vite

### 2. [Local AI Benchmarking & Evaluation](./02-local-ai-assistant/)
A privacy-first local AI benchmarking suite using Small Language Models (SLMs) optimized for an 8GB Unified Memory footprint. Includes a UI for rigorously measuring and comparing model performance.
- **Core:** Structured Output (JSON Enforcing), Latency Tracking (TTFT, TPS), Quantization Benchmarking.
- **Stack:** Ollama, Llama 3.2 3B, Qwen 2.5 1.5B, Phi-3 Mini 3.8B, React Dashboard.

### 3. AI Monitoring & Observability (Coming Soon)
A production telemetry layer for tracking token costs, latency, and quality drift in real-time.
- **Core:** Tracing, Cost Analytics, Guardrails, Output Validation.
- **Stack:** LangSmith / Phoenix, OpenTelemetry, Pydantic/Instructor.

### 4. Task-Specific Fine-Tuning (Coming Soon)
Customizing an open-source model using parameter-efficient fine-tuning (PEFT) on a proprietary dataset.
- **Core:** LoRA/QLoRA, Dataset Curation, Instruct-Tuning.
- **Stack:** HuggingFace Unsloth, PyTorch, vLLM.

### 5. Real-time Multimodal App (Coming Soon)
A robust service orchestrating interleaved text, audio, and vision endpoints.
- **Core:** Websockets, Multimodal Orchestration, Streaming Generation.
- **Stack:** FastAPI, Next.js, OpenAI Vision / Gemini 1.5 Pro.

---
*Created by Rakif Khan*
