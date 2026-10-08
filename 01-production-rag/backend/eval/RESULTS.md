# Veritas evaluation results (2026-10-08) - PARTIAL

> **Resume note:** All 30 questions ran through the pipeline (20 answerable, 10 unanswerable). The Gemini judge hit its free-tier quota after 15 answerable answers, so A16-A20 are recorded but **not yet judged** and are excluded from the numbers below. To finish (after quotas reset):
> `cd backend && source .venv/bin/activate && python eval/run_eval.py --rejudge eval/results/2026-10-08.json`
> (judges only A16-A20 from the saved answers; no pipeline calls are repeated). Raw data: `results/2026-10-08.json` (`"partial": true`).

**This is a small sample (15 judged answerable + 10 unanswerable) on three tiny documents (7 pages, 13 chunks). Treat the numbers as a sanity check, not a benchmark.**

## Setup
- Pipeline: the `/query/prod` chain of `api.py` (PDF ingest, 3200-char chunks, MiniLM embeddings in Chroma, BM25 + semantic hybrid k=5, Cohere `rerank-english-v3.0` top 3, Veritas refusal prompt copied from `api.py`).
- Generator: Groq `openai/gpt-oss-20b`. Judge: Gemini `gemini-2.5-flash` (sees question, ground truth, answer only).
- Corpus: `Clinical_Guidelines_2024.pdf`, `Company_Employee_Handbook.pdf`, `daily_use_case.pdf` (IRS W-4), all indexed together.
- Run on a MacBook (Python 3.12), home network, free-tier API keys. Golden set: `golden_dataset.json`.
- Refusal = answer matches the refusal phrase regex; all 10 unanswerable outputs were read and are the exact refusal sentence.

## Metrics
| Metric | Result | n |
|---|---|---|
| Correctness (judge: correct) | 13/15 = 86.7% | 15 judged |
| Correct or partial | 15/15 = 100% | 15 |
| Incorrect | 0 | 15 |
| False-refusal rate (answerable) | 0/15 = 0% | 15 |
| Retrieval hit (right PDF in top 3) | 15/15 = 100% | 15 |
| **Citation rate (answer names the right PDF)** | **0/15 = 0%** | 15 |
| Correct refusals (unanswerable) | 10/10 = 100% | 10 |
| Hallucination rate (unanswerable) | 0/10 = 0% | 10 |
| Latency p50 / p95 / mean | 1.64 s / 19.5 s / 5.4 s | 25 |

Latency is retrieve + rerank + generate (judge excluded). The slow tail (8-21 s) occurred only in the last unanswerable questions; I did not isolate the cause (likely Cohere trial-key throttling, unverified). No Groq/Cohere retries were logged. RAGAS was not run.

## Failure examples
1. **No citations (all 15 answers).** The `api.py` prod prompt never asks the model to cite, so answers carry no source. The README previously claimed citation grounding; that claim is not supported by this pipeline.
2. **A01 (judged partial):** asked for the AMI aspirin dose, the answer gave "162-325 mg" but omitted the required P2Y12 inhibitor co-administration.
3. **A14 (judged partial):** asked what amount is multiplied per qualifying child; the answer was a bare "$2,200." with no context. Arguably correct; the judge was strict.

## Limitations
- Tiny, self-written golden set; questions are easy single-fact lookups from one-page documents. Unanswerable questions are mostly obviously out-of-domain, so 100% refusal says little about harder near-miss questions.
- Single run, temperature 0, no confidence intervals; one LLM judge, not human-checked; A16-A20 unjudged.
- Refusal detection is regex-based.

## Code changes made to run the pipeline
- `src/generation/llm_chain.py`: Groq model `llama-3.1-8b-instant` no longer exists on the account (404 `model_not_found`), so the existing app was broken; default changed to `openai/gpt-oss-20b`, overridable via `VERITAS_GROQ_MODEL`.
- `requirements.txt`: added `langchain-classic`, `rank_bm25`, `pyyaml` (imported but missing).
- Removed `eval/run_ragas.py` (it scored the ground truth as the "answer"); replaced by `eval/run_eval.py`. CI workflow paths fixed; the script exits 0 with a SKIP message when keys are absent.
