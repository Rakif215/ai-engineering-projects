"""End-to-end evaluation of the Veritas pipeline.

Runs the REAL chain used by api.py (/query/prod):
  PDF ingest -> chunk -> Chroma (MiniLM) -> hybrid BM25+semantic (k=5 each)
  -> Cohere rerank (top_n=3) -> Groq LLM (see GEN_MODEL) with the Veritas prompt
for every question in golden_dataset.json, then scores the results.

Metrics
  answerable:   correctness (Gemini judge: correct / partial / incorrect),
                citation rate (answer text names the right source PDF),
                false-refusal rate, retrieval hit rate (right PDF in top-3)
  unanswerable: correct-refusal rate, hallucination rate (answered anyway)
  all:          latency p50 / p95 (retrieve + rerank + generate, excl. judge)

Usage (from backend/):  python eval/run_eval.py [--limit N]
Needs GROQ_API_KEY, COHERE_API_KEY, GEMINI_API_KEY (env or .env).
"""
import argparse
import datetime
import json
import os
import re
import statistics
import sys
import tempfile
import time

HERE = os.path.dirname(os.path.abspath(__file__))
BACKEND = os.path.dirname(HERE)
sys.path.insert(0, BACKEND)

from dotenv import load_dotenv

load_dotenv(os.path.join(BACKEND, ".env"))

REQUIRED = ["GROQ_API_KEY", "COHERE_API_KEY", "GEMINI_API_KEY"]
missing = [k for k in REQUIRED if not os.environ.get(k)]
if missing:
    print(f"SKIP: missing env vars {missing}; evaluation needs live API keys.")
    sys.exit(0)
os.environ.setdefault("GOOGLE_API_KEY", os.environ["GEMINI_API_KEY"])

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI

from src.generation.llm_chain import GenerationChain
from src.ingestion.chunking import DocumentChunker
from src.ingestion.loaders import DocumentIngestor
from src.retrieval.hybrid_search import HybridRetrieverManager
from src.retrieval.reranking import RerankingManager
from src.storage.vectorstore import VectorStoreManager

PDFS = ["Clinical_Guidelines_2024.pdf", "Company_Employee_Handbook.pdf", "daily_use_case.pdf"]
JUDGE_MODEL = "gemini-2.5-flash"
GEN_MODEL = os.environ.get("VERITAS_GROQ_MODEL", "openai/gpt-oss-20b") + " (Groq)"
RERANK_MODEL = "rerank-english-v3.0 (Cohere)"
EMBED_MODEL = "all-MiniLM-L6-v2 (local HF)"

# Copied verbatim from api.py (ProdGenerationChain); api.py cannot be imported
# without side effects (it ingests a hard-coded PDF at import time).
PROD_SYSTEM_PROMPT = (
    "You are Veritas, a medical QA system. You strictly adhere to grounding rules. "
    "If the exact answer is not explicitly detailed in the context, you MUST refuse to answer. "
    "Say: 'I cannot answer this question. The provided clinical context does not contain "
    "sufficient information.' Do not guess."
)


class ProdGenerationChain(GenerationChain):
    def _build_prompt(self):
        return ChatPromptTemplate.from_messages([
            ("system", PROD_SYSTEM_PROMPT),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "Context:\n{context}\n\nQuestion: {question}"),
        ])


class FixedRetriever:  # same trick api.py uses (MockRetriever) to reuse retrieved docs
    def __init__(self, docs):
        self.docs = docs

    def invoke(self, q):
        return self.docs


REFUSAL_RE = re.compile(
    r"cannot answer|can't answer|does not contain (sufficient|enough|any)|"
    r"not contain sufficient|insufficient information|unable to answer", re.I)


def is_refusal(ans):
    return bool(REFUSAL_RE.search(ans))


def with_retry(fn, tries=6, base=15):
    """Call fn, backing off on rate-limit style errors. Returns (value, n_retries)."""
    for attempt in range(tries):
        try:
            return fn(), attempt
        except Exception as e:  # noqa: BLE001
            msg = str(e).lower()
            transient = any(s in msg for s in ("429", "rate", "quota", "resource_exhausted", "503", "overloaded", "timeout"))
            if attempt == tries - 1 or not transient:
                raise
            wait = base * (attempt + 1)
            print(f"   transient error ({type(e).__name__}), sleeping {wait}s")
            time.sleep(wait)


def build_pipeline():
    """Mirror api.py: ingest -> chunk -> Chroma -> hybrid -> rerank."""
    ingestor, chunker = DocumentIngestor(), DocumentChunker()
    chunks, pages = [], 0
    for pdf in PDFS:
        docs = ingestor.load_pdf(os.path.join(BACKEND, pdf))
        pages += len(docs)
        chunks.extend(chunker.chunk_documents(docs))
    # VectorStoreManager persists under cwd/chroma_db; use a throwaway dir.
    tmp = tempfile.mkdtemp(prefix="veritas_eval_")
    os.chdir(tmp)
    vs = VectorStoreManager(collection_name="veritas_eval")
    vs.add_documents(chunks)
    hybrid = HybridRetrieverManager(vs.get_retriever(k=5), chunks).get_retriever()
    retriever = RerankingManager(hybrid, top_n=3).get_retriever()
    return retriever, ProdGenerationChain(llm_provider="groq"), len(chunks), pages


JUDGE_PROMPT = """You are grading an answer against a reference answer.
Question: {q}
Reference answer: {gt}
Candidate answer: {a}

Grade the candidate: "correct" if it conveys the same key facts as the reference
(extra detail is fine, but no contradictions), "partial" if it has some but not all
key facts, "incorrect" if it is wrong, contradicts the reference, or declines to answer.
Reply with exactly one word: correct, partial, or incorrect."""


def judge(llm, q, gt, a):
    def call():
        out = llm.invoke(JUDGE_PROMPT.format(q=q, gt=gt, a=a)).content
        out = out if isinstance(out, str) else str(out)
        m = re.search(r"\b(correct|partial|incorrect)\b", out.lower())
        if not m:
            raise ValueError(f"unparseable judge output: {out!r}")
        return m.group(1)
    return with_retry(call, base=20)[0]


def pct(x, n):
    return None if n == 0 else round(100.0 * x / n, 1)


def percentile(vals, p):
    vals = sorted(vals)
    if not vals:
        return None
    k = (len(vals) - 1) * p / 100
    lo, hi = int(k), min(int(k) + 1, len(vals) - 1)
    return round(vals[lo] + (vals[hi] - vals[lo]) * (k - lo), 2)


def compute_metrics(records):
    ok = [r for r in records if "error" not in r]
    ans_r = [r for r in ok if r["answerable"]]
    un_r = [r for r in ok if not r["answerable"]]
    lats = [r["latency_s"] for r in ok]
    verdicts = [r["verdict"] for r in ans_r]
    metrics = {
        "n_total": len(records), "n_errors": len(records) - len(ok),
        "answerable": {
            "n": len(ans_r),
            "correct": verdicts.count("correct"), "partial": verdicts.count("partial"),
            "incorrect": verdicts.count("incorrect"),
            "correct_pct": pct(verdicts.count("correct"), len(ans_r)),
            "correct_or_partial_pct": pct(verdicts.count("correct") + verdicts.count("partial"), len(ans_r)),
            "false_refusals": sum(r["refused"] for r in ans_r),
            "false_refusal_pct": pct(sum(r["refused"] for r in ans_r), len(ans_r)),
            "citation_pct": pct(sum(r["cited_correct_source"] for r in ans_r), len(ans_r)),
            "citation_count": sum(r["cited_correct_source"] for r in ans_r),
            "retrieval_hit_pct": pct(sum(r["retrieval_hit"] for r in ans_r), len(ans_r)),
            "retrieval_hit_count": sum(r["retrieval_hit"] for r in ans_r),
        },
        "unanswerable": {
            "n": len(un_r),
            "correct_refusals": sum(r["refused"] for r in un_r),
            "correct_refusal_pct": pct(sum(r["refused"] for r in un_r), len(un_r)),
            "hallucinated": sum(not r["refused"] for r in un_r),
            "hallucination_pct": pct(sum(not r["refused"] for r in un_r), len(un_r)),
        },
        "latency_s": {"p50": percentile(lats, 50), "p95": percentile(lats, 95),
                      "mean": round(statistics.mean(lats), 2) if lats else None, "n": len(lats)},
    }
    return metrics


def write_results(records, metrics, n_chunks, n_pages, date):
    out = {
        "date": date,
        "models": {"generator": GEN_MODEL, "reranker": RERANK_MODEL, "embeddings": EMBED_MODEL, "judge": JUDGE_MODEL},
        "corpus": {"pdfs": PDFS, "pages": n_pages, "chunks": n_chunks},
        "metrics": metrics,
        "records": records,
    }
    os.makedirs(os.path.join(HERE, "results"), exist_ok=True)
    path = os.path.join(HERE, "results", f"{date}.json")
    json.dump(out, open(path, "w"), indent=2)
    print(json.dumps(metrics, indent=2))
    print("Wrote", path)


def rejudge(path):
    """Re-run only the judge for records whose judging failed (rate limit); keeps answers."""
    d = json.load(open(path))
    llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0)
    for r in d["records"]:
        if r.get("answerable") and "answer" in r and "verdict" not in r:
            r.pop("error", None)
            r["retrieval_hit"] = any(x["source"] == r["source"] for x in r["retrieved"])
            r["cited_correct_source"] = r["source"] in r["cited_sources"]
            time.sleep(6.5)
            try:
                r["verdict"] = judge(llm, r["question"], r["ground_truth"], r["answer"])
            except Exception as e:  # noqa: BLE001
                r["error"] = f"judge {type(e).__name__}: {str(e)[:120]}"
            print(r["id"], r.get("verdict"), r.get("error"))
    d["metrics"] = compute_metrics(d["records"])
    json.dump(d, open(path, "w"), indent=2)
    print(json.dumps(d["metrics"], indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rejudge", metavar="RESULTS_JSON", help="only re-judge records whose judge call failed")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sleep", type=float, default=7.0, help="seconds between questions (Cohere trial = 10 rerank calls/min)")
    args = ap.parse_args()
    if args.rejudge:
        return rejudge(args.rejudge)

    golden = json.load(open(os.path.join(HERE, "golden_dataset.json")))
    if args.limit:
        golden = golden[: args.limit]

    print("Building pipeline (ingest + embed)...")
    retriever, chain, n_chunks, n_pages = build_pipeline()
    print(f"Indexed {n_pages} pages / {n_chunks} chunks from {len(PDFS)} PDFs")
    judge_llm = ChatGoogleGenerativeAI(model=JUDGE_MODEL, temperature=0)

    records = []
    for i, item in enumerate(golden, 1):
        print(f"[{i}/{len(golden)}] {item['id']} {item['question']}")
        rec = {k: item[k] for k in ("id", "answerable", "question", "ground_truth", "source", "page")}
        try:
            def run():
                t0 = time.perf_counter()
                docs = retriever.invoke(item["question"])
                chain.clear_history()  # each question is independent
                ans = chain.answer(item["question"], FixedRetriever(docs))
                return docs, ans, time.perf_counter() - t0
            (docs, ans, lat), retries = with_retry(run)
            rec.update(
                answer=ans,
                latency_s=round(lat, 3),
                retries=retries,
                refused=is_refusal(ans),
                retrieved=[
                    {"source": os.path.basename(d.metadata.get("source", "?")),
                     "page": int(d.metadata.get("page", 0)) + 1,
                     "score": round(float(d.metadata.get("relevance_score", -1)), 3),
                     "text": d.page_content}
                    for d in docs
                ],
            )
            low = ans.lower()
            rec["cited_sources"] = sorted({p for p in PDFS if p.lower() in low or p.lower().removesuffix(".pdf") in low})
            if item["answerable"]:
                rec["retrieval_hit"] = any(r["source"] == item["source"] for r in rec["retrieved"])
                rec["cited_correct_source"] = item["source"] in rec["cited_sources"]
                time.sleep(6.5)  # Gemini free tier ~10 RPM
                rec["verdict"] = judge(judge_llm, item["question"], item["ground_truth"], ans)
                print(f"   -> refused={rec['refused']} verdict={rec['verdict']} lat={lat:.2f}s")
            else:
                print(f"   -> refused={rec['refused']} lat={lat:.2f}s")
        except Exception as e:  # noqa: BLE001
            rec["error"] = f"{type(e).__name__}: {str(e)[:200]}"
            print("   ERROR", rec["error"])
        records.append(rec)
        time.sleep(args.sleep)

    metrics = compute_metrics(records)
    write_results(records, metrics, n_chunks, n_pages, datetime.date.today().isoformat())


if __name__ == "__main__":
    main()
