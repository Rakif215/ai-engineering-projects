# AI Engineering Projects

A collection of production-grade AI engineering systems built with a focus on **reliability, observability, and measurable performance**. While most AI tutorials stop at building a demo chatbot, these projects explore the engineering discipline required to deploy AI systems responsibly.

---

## Projects

### 1. [Veritas: Production-Grade RAG](./01-production-rag/)
> *"A RAG system that knows when to say nothing."*

A specialized "Ask My Docs" system engineered to **refuse answers when the source context is insufficient**, preventing hallucination in high-stakes domains like healthcare and legal.

| Feature | Detail |
|---|---|
| Retrieval | Hybrid BM25 + Semantic Similarity |
| Reranking | Cohere Cross-Encoder |
| Grounding | Strict citation enforcement — no source, no answer |
| Stack | Llama-3.1 (Groq) / Gemini, ChromaDB, HuggingFace |
| Frontend | React + Vite |

---

### 2. [Local SLM Benchmarking Dashboard](./02-local-ai-assistant/)
> *"On hardware-constrained devices, smaller models aren't a compromise — they're the correct engineering decision."*

A privacy-first analytics suite that **benchmarks and compares Small Language Models (SLMs)** running fully locally against cloud API baselines, with a premium React dashboard for data storytelling.

| Feature | Detail |
|---|---|
| Metrics | TTFT, TPS, Total Latency, Std Deviation |
| Baselines | OpenRouter cloud models (Gemini 2.5 Flash, Llama 3.1 8B) |
| Analytics | Winner Podium, Cold vs. Warm run charts (log-scale), Auto-Insights |
| Models | Qwen 2.5 1.5B, Llama 3.2 3B, Phi-3 Mini 3.8B |
| Stack | Ollama, React, Vite, Recharts, Zod, TypeScript |

---

### 3. AI Monitoring & Observability *(In Development)*
A production telemetry layer for **tracking token costs, latency drift, and quality regressions** in real-time across AI pipelines.

- **Core:** Distributed Tracing, Cost Analytics, Guardrails, Output Quality Regression Tests
- **Stack:** OpenTelemetry, Pydantic/Instructor, LangSmith / Phoenix

---

### 4. Task-Specific Fine-Tuning *(Coming Soon)*
Customizing an open-source model using parameter-efficient fine-tuning (PEFT) on a proprietary dataset.

- **Core:** LoRA/QLoRA, Dataset Curation, Instruct-Tuning
- **Stack:** HuggingFace Unsloth, PyTorch, vLLM

---

### 5. Real-time Multimodal App *(Coming Soon)*
A robust service orchestrating interleaved text, audio, and vision endpoints at scale.

- **Core:** WebSockets, Multimodal Orchestration, Streaming Generation
- **Stack:** FastAPI, Next.js, OpenAI Vision / Gemini 1.5 Pro

---

*Created by Rakif Khan*
