# Project 2: Local SLM Benchmarking Dashboard

> *"On hardware-constrained devices, smaller models aren't a compromise — they're the correct engineering decision."*

The second project in the AI Engineering portfolio. Instead of just connecting a chat interface to a cloud API, this project rigorously **measures, compares, and visualizes** the performance of Small Language Models (SLMs) running locally on an Apple M2 (8GB RAM) — a real constraint that forces real engineering decisions.

---

## The Hypothesis

Most developers default to the largest available model. This project tests a different thesis:

> **For structured, constrained tasks (data extraction, classification, formatting), a locally-run 1.5B parameter model can outperform a cloud-hosted 8B model on the metrics that matter for production: latency, throughput, and operational cost.**

---

## The Benchmarking Dashboard

A fully custom **React + TypeScript analytics dashboard** that connects directly to the local Ollama API and the OpenRouter cloud API to run head-to-head model comparisons.

### Dashboard Features

| Feature | Description |
|---|---|
| **Winner Podium** | Auto-computed hero cards showing best TPS, best TTFT, and most consistent model after each benchmark run |
| **Grouped Cold/Warm Charts** | Bar charts split "Cold Start (Run 1)" vs "Warm Average" to surface memory-swap penalties |
| **Log-Scale TTFT Chart** | Logarithmic axis prevents outliers (e.g. 18s Phi-3 cold start) from making all other bars look flat |
| **Auto-Generated Insights** | 3 intelligence cards auto-populated from results: Memory Pressure Alert, Cloud vs Local winner, Consistency Winner |
| **Enhanced Results Table** | Green/red highlighting for best/worst values per column, Cold/Warm run badges, and a TPS Consistency (std. dev.) column |
| **Cloud Baseline Comparison** | OpenRouter integration brings in Gemini 2.5 Flash and Llama 3.1 8B as cloud controls |
| **Self-Healing JSON Output** | Structured output mode with Zod schema validation and automatic retry-on-parse-failure |

---

## Models Evaluated

| Model | Parameters | VRAM (Q4) | Origin |
|---|---|---|---|
| Qwen 2.5 | 1.5B | ~1GB | Local (Ollama) |
| Llama 3.2 | 3B | ~2GB | Local (Ollama) |
| Phi-3 Mini | 3.8B | ~2.3GB | Local (Ollama) |
| Llama 3.1 Instruct | 8B | Cloud | OpenRouter |
| Gemini 2.5 Flash | — | Cloud | OpenRouter |

---

## Key Findings

- 🏆 **Qwen 2.5 (1.5B)** achieved **101+ TPS** on warm runs — beating the 8B cloud model's streaming rate
- ⚠️ **Phi-3 Mini** showed an 18s cold TTFT on first load, exposing **macOS SSD swap** behaviour when weights exceed unified memory
- ✅ After warm-loading, local models consistently outperformed cloud models on **tokens-per-second**, proving zero-network-overhead is a real advantage for streaming tasks

---

## Setup & Running Locally

### Prerequisites
- [Ollama](https://ollama.com) installed
- Node.js 18+
- OpenRouter API key *(optional, for cloud baseline)*

### 1. Start Ollama with CORS Enabled
```bash
OLLAMA_ORIGINS="*" ollama serve
```
> If Ollama is already running as a menu-bar app, you must fully quit it first, then run this command.

### 2. Pull the Models
```bash
ollama pull qwen2.5:1.5b
ollama pull llama3.2
ollama pull phi3
```

### 3. Start the Dashboard
```bash
cd dashboard
npm install
npm run dev
```

Open `http://localhost:3000` → **Connection** tab → click **Refresh Models**.

*(Optionally paste your OpenRouter key to load cloud baselines.)*

---

## Project Structure

```
02-local-ai-assistant/
├── dashboard/
│   ├── src/
│   │   ├── components/
│   │   │   └── Dashboard.tsx      # Main UI — podium, charts, table, insights
│   │   └── lib/
│   │       ├── ollama.ts          # Local inference engine + streaming metrics
│   │       └── openrouter.ts      # Cloud inference engine + SSE streaming
│   └── package.json
├── results/
│   └── benchmark_report.md        # Full written analysis of benchmark findings
└── README.md
```

---

*Part of the [AI Engineering Projects](../README.md) portfolio by Rakif Khan*
