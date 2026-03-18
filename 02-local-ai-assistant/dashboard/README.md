# Local SLM Benchmark Dashboard

A privacy-focused, web-based benchmarking tool for Small Language Models (SLMs) running locally via [Ollama](https://ollama.com/). This dashboard helps you evaluate models based on engineering metrics rather than hype, testing for speed, latency, and structural reliability.

## Features

- **Local Inference:** Connects directly to your local Ollama instance. No data is sent to external APIs.
- **Single Run & Validation:** 
  - Test individual prompts against specific models.
  - **Schema Enforcement:** Force JSON output and validate it against a strict Zod schema.
  - **Self-Correction:** Includes a "retry-once" mechanism that feeds validation errors back to the model to correct invalid JSON.
- **Multi-Model Batch Benchmarking:**
  - Run the same set of prompts across multiple models simultaneously (e.g., Llama 3.2, Phi-4, Mistral).
  - Automatically locks temperature to `0` for deterministic benchmarking.
  - Visualizes **Tokens Per Second (TPS)** and **Time to First Token (TTFT)** using interactive charts.
- **Quantization Testing:** Easily compare different quantized versions (e.g., Q4 vs Q5) of the same model to document the "Quality vs. Speed" trade-off.

## Prerequisites

1. Install [Ollama](https://ollama.com/).
2. Pull the models you want to test. For example:
   ```bash
   ollama run llama3.2
   ollama run phi4
   ollama run mistral
   ```

## Getting Started: Fixing CORS Issues

Because this dashboard is a web application hosted in the browser, it needs permission to communicate with your local Ollama instance. By default, Ollama blocks these cross-origin requests.

**You must start Ollama with CORS enabled.**

1. **Stop Ollama** if it is currently running (quit the app from your Mac menu bar or Windows system tray).
2. **Start Ollama** from your terminal using the appropriate command for your OS:

**Mac / Linux:**
```bash
OLLAMA_ORIGINS="*" ollama serve
```

**Windows (Command Prompt):**
```cmd
set OLLAMA_ORIGINS="*" && ollama serve
```

**Windows (PowerShell):**
```powershell
$env:OLLAMA_ORIGINS="*" ; ollama serve
```

## Usage

1. **Connection Setup:** Open the dashboard and ensure the Base URL is set to `http://localhost:11434` (or `http://127.0.0.1:11434`). Click "Connect" to load your local models.
2. **Single Run:** Select a model, enter a prompt, adjust the temperature, and optionally toggle "Enforce JSON Schema" to test the model's ability to follow strict structural constraints.
3. **Batch Benchmark:** Select multiple models, enter a list of prompts (one per line), and click "Start Benchmark" to generate comparative performance charts.

## Tech Stack

- **Frontend:** React 19, Vite, Tailwind CSS v4
- **Validation:** Zod
- **Charts:** Recharts
- **Icons:** Lucide React
- **API:** Ollama REST API
