# Project 2: Local AI Assistant & Benchmarking

The second project in the AI Engineering Project portfolio focuses on creating a reliable, privacy-first local AI assistant using Small Language Models (SLMs) via Ollama. 

Instead of just hooking up a chat interface to a massive cloud model, this project explores the engineering rigor required to evaluate and run models under strict hardware constraints (specifically, an 8GB RAM threshold) without sacrificing structured data output capability.

## Overview
A common pitfall in local AI development is selecting models that are too large for the host machine's unified memory, leading to severe SSD swapping, crashed daemons, and tokens-per-second (TPS) rates in the single digits. 

This project tackles that constraint head-on by benchmarking three highly optimized, sub-4B parameter SLMs to identify the most performant baseline for local intelligent agents.

### Evaluated Models
1. **Llama 3.2 (3B)** - Meta's highly capable edge-optimized completion model (~2GB VRAM at Q4).
2. **Qwen 2.5 (1.5B)** - Alibaba's state-of-the-art multilingual and reasoning SLM (~1GB VRAM at Q4).
3. **Phi-3 Mini (3.8B)** - Microsoft's dense model trained on synthetic, high-quality data (~2.3GB VRAM at Q4).

## The Benchmarking Engine
To rigorously test these models, a React-based **Local SLM Benchmark Dashboard** was developed. It connects directly to the local Ollama REST API (and optionally the OpenRouter API for cloud baselining) to monitor performance in real-time.

### Key Features
- **Benchmarking Engine:** Accurately measures Time to First Token (TTFT) and Generation Speed (Tokens/sec).
- **Schema Enforcement:** Forces output into a strict JSON schema using Zod, testing the local inference structure capabilities.
- **Self-Healing Loop:** Built-in "Retry-once" logic—if a local model outputs invalid JSON, the app automatically feeds the error back for a correction attempt.
- **Batch Evaluation:** Compares multiple models sequentially across a standardized prompt dataset, producing visual analytics (Recharts).

## Setup & Running Locally

### 1. Start Ollama with CORS Enabled
The web dashboard needs to communicate with your local Ollama daemon. You must start Ollama explicitly allowing cross-origin requests.

**Mac / Linux Terminal:**
```bash
OLLAMA_ORIGINS="*" ollama serve
```
*(If Ollama is already running as a menu-bar application, you must fully quit it first.)*

### 2. Pull Required Models
In a separate terminal, pull the memory-optimized models:
```bash
ollama pull llama3.2
ollama pull qwen2.5:1.5b
ollama pull phi3
```

### 3. Start the Benchmark Dashboard
```bash
cd dashboard
npm install
npm run dev
```

Visit the dashboard URL (typically `http://localhost:3000`), ensure the Ollama URL is registered correctly (`http://localhost:11434`), and start benchmarking!
