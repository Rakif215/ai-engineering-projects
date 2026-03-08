"""
seed_metrics.py — Generates realistic synthetic metrics data so the dashboard
has something to show immediately, without needing to run 100 real queries.

Run once after setup:
    python seed_metrics.py
"""

import sys
import os
import uuid
import random
import numpy as np
from datetime import datetime, timedelta
sys.path.insert(0, os.path.dirname(__file__))
from src.metrics import MetricsRecorder

recorder = MetricsRecorder()

QUESTIONS = [
    "What is the retrieval pipeline?",
    "How does BM25 work?",
    "What is cross-encoder reranking?",
    "Who are the authors of this document?",
    "Tell me more about evaluation metrics.",
    "What is RAGAS used for?",
    "How is cost calculated per request?",
    "What are the main components of this system?",
    "Can you explain hybrid search?",
    "What models are supported?",
]

MODELS = [
    ("llama-3.1-8b-instant", "groq"),
    ("gemini-2.5-flash", "gemini"),
]

print("Seeding 200 synthetic metric records...")
now = datetime.utcnow()

for i in range(200):
    # Spread over last 7 days with more recent having more requests
    hours_ago = random.expovariate(0.3) * 24
    hours_ago = min(hours_ago, 168)  # cap at 7 days

    latency_ms = max(300, np.random.lognormal(mean=7.5, sigma=0.5))  # realistic: ~1800ms mean
    is_error = random.random() < 0.04  # 4% base error rate
    is_grounded = True if is_error else (random.random() < 0.85)
    cost_usd = random.uniform(0.000001, 0.00008)
    model, provider = random.choice(MODELS)
    question = random.choice(QUESTIONS)

    # Override timestamp via direct insert hack
    from src.metrics import _get_conn, DB_PATH
    import sqlite3
    fake_ts = (now - timedelta(hours=hours_ago)).isoformat()

    conn = _get_conn()
    conn.execute("""
        INSERT INTO requests
          (session_id, timestamp, latency_ms, prompt_tokens, completion_tokens,
           total_tokens, cost_usd, is_grounded, is_error, model, provider,
           question, answer_preview)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        str(uuid.uuid4()), fake_ts, round(latency_ms, 1),
        random.randint(50, 300), random.randint(50, 400),
        random.randint(100, 700), cost_usd,
        1 if is_grounded else 0, 1 if is_error else 0,
        model, provider, question,
        f"Sample answer for: {question}"
    ))
    conn.commit()
    conn.close()

print(f"✅ Seeded 200 records into {DB_PATH}")
print("Now run: streamlit run dashboard.py")
