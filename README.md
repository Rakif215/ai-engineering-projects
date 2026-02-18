# AI Engineering Portfolio

Welcome to my AI Engineering Portfolio! This repository is a collection of 5 distinct, production-grade AI systems, designed to demonstrate a comprehensive understanding of the modern AI engineering lifecycle—from prototype to production.

Each project focuses on a different core competency required in the current AI landscape, prioritizing reliability, scale, observability, and evaluation over simple chatbot tutorials.

## Projects Overview

### 1. [Production-Grade RAG System](./01-production-rag/)
A domain-specific "Ask My Docs" system built for enterprise reliability.
* **Key Skills:** Hybrid Retrieval (BM25 + Semantic), Cross-Encoder Reranking, Citation Enforcement, Anti-Hallucination Prompting.
* **Tech Stack:** LangChain, ChromaDB, HuggingFace Local Embeddings, Groq (Llama-3.1), Cohere Rerank, RAGAS (CI/CD Evaluation).
* **Highlights:** Programmatic gating of pull requests based on automated `Faithfulness` evaluation scores.

### 2. Local AI Assistant (Coming Soon)
A fully local, privacy-first AI agent utilizing specialized Small Language Models (SLMs).
* **Key Skills:** Quantization (GGUF), Local LLM Execution, On-device Agentic Workflows.
* **Tech Stack:** Ollama, Llama.cpp, Llama-3-8B-Instruct.

### 3. AI Monitoring & Observability (Coming Soon)
A production telemetry system for tracking LLM costs, latency, and quality drift.
* **Key Skills:** Tracing, Cost Analytics, Guardrails, Output Validation.
* **Tech Stack:** LangSmith / Phoenix, OpenTelemetry, Pydantic/Instructor.

### 4. Task-Specific Fine-Tuning (Coming Soon)
Customizing an open-source model using parameter-efficient fine-tuning (PEFT) on proprietary data.
* **Key Skills:** LoRA/QLoRA, Dataset Curation, Instruct-Tuning, Model Deployment.
* **Tech Stack:** HuggingFace Unsloth, PyTorch, vLLM.

### 5. Real-time Multimodal App (Coming Soon)
A robust application orchestrating interleaved text, audio, and vision endpoints.
* **Key Skills:** Websockets, Multimodal Orchestration, Streaming Generation.
* **Tech Stack:** FastAPI, Next.js, OpenAI Vision / Gemini 1.5 Pro.

---
*Developed as part of a comprehensive AI Product Lifecycle journey.*
