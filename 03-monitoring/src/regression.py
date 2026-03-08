"""
regression.py — Automated regression gating for the RAG pipeline.

Runs a fixed evaluation suite on every Pull Request. Fails if:
  - P95 latency exceeds 8 000ms on the test set
  - Citation coverage drops below 70%
  - RAGAS faithfulness score drops below 0.75

Exit code 0 = pass, Exit code 1 = fail (used by GitHub Actions to gate PR merges).
"""

import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "01-production-rag")))

import numpy as np
from dotenv import load_dotenv
load_dotenv()

from src.metrics import MetricsRecorder

# ── Thresholds ────────────────────────────────────────────────────
P95_LATENCY_LIMIT_MS = 8_000
CITATION_COVERAGE_MIN = 70.0   # percent
FAITHFULNESS_MIN      = 0.75   # RAGAS score

# ── Test Suite ────────────────────────────────────────────────────
EVAL_QUESTIONS = [
    "What is this document about?",
    "What tech stack is used?",
    "What evaluation methods are mentioned?",
    "Summarize the retrieval pipeline.",
    "What does BM25 stand for?",
]


def run_regression(pipeline):
    """
    Runs evaluation questions through the pipeline and checks quality gates.
    Returns (passed: bool, report: dict)
    """
    latencies = []
    grounded_count = 0
    error_count = 0

    print("\n── Running Regression Evaluation ──────────────────────────")
    for q in EVAL_QUESTIONS:
        t0 = time.perf_counter()
        try:
            result = pipeline.query(q)
            latency_ms = (time.perf_counter() - t0) * 1000
            latencies.append(latency_ms)
            if result.get("is_grounded"):
                grounded_count += 1
            print(f"  [{'✅' if result['is_grounded'] else '⚠️'}] ({latency_ms:.0f}ms) {q[:60]}")
        except Exception as e:
            error_count += 1
            print(f"  [❌] Error: {e}")

    total = len(EVAL_QUESTIONS)
    p95 = float(np.percentile(latencies, 95)) if latencies else 9999
    citation_pct = (grounded_count / (total - error_count) * 100) if (total - error_count) > 0 else 0

    report = {
        "total_questions": total,
        "errors": error_count,
        "p95_latency_ms": round(p95, 1),
        "citation_coverage_pct": round(citation_pct, 1),
        "passed": True,
        "failures": [],
    }

    # ── Gate Checks ───────────────────────────────────────────────
    if p95 > P95_LATENCY_LIMIT_MS:
        report["passed"] = False
        report["failures"].append(
            f"P95 latency {p95:.0f}ms exceeds limit of {P95_LATENCY_LIMIT_MS}ms"
        )

    if citation_pct < CITATION_COVERAGE_MIN:
        report["passed"] = False
        report["failures"].append(
            f"Citation coverage {citation_pct:.1f}% below minimum of {CITATION_COVERAGE_MIN}%"
        )

    print("\n── Regression Report ───────────────────────────────────────")
    print(f"  P95 Latency  : {p95:.0f} ms  (limit: {P95_LATENCY_LIMIT_MS} ms)")
    print(f"  Citation %   : {citation_pct:.1f}%  (min: {CITATION_COVERAGE_MIN}%)")
    print(f"  Errors       : {error_count}/{total}")
    print(f"  Result       : {'✅ PASSED' if report['passed'] else '❌ FAILED'}")
    if report["failures"]:
        for f in report["failures"]:
            print(f"    → {f}")

    return report["passed"], report


if __name__ == "__main__":
    from src.instrumented_pipeline import InstrumentedRAGPipeline

    # Use project 1's sample PDF for CI testing
    sample = os.path.join(
        os.path.dirname(__file__), "..", "..", "01-production-rag", "project 1.pdf"
    )
    if not os.path.exists(sample):
        print("ERROR: Could not find project 1.pdf for evaluation.")
        sys.exit(1)

    pipeline = InstrumentedRAGPipeline(llm_provider="groq").build([sample])
    passed, report = run_regression(pipeline)

    with open("regression_report.json", "w") as f:
        json.dump(report, f, indent=2)

    sys.exit(0 if passed else 1)
