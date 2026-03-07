"""
dashboard.py — Streamlit Observability Dashboard for the RAG pipeline.

Shows:
  - P50 / P95 latency histogram
  - Cost-per-request timeline  
  - Citation coverage gauge
  - Failure rate panel
  - Per-request trace inspector
"""

import sys
import os
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(__file__))
from src.metrics import MetricsRecorder, DB_PATH
from src.tracer import LANGFUSE_ENABLED

# ─── Page Config ───────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Observability Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Dark Theme CSS ────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.page-title {
    background: linear-gradient(135deg, #f093fb 0%, #f5576c 50%, #fda085 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-size: 2.2rem;
    font-weight: 700;
}
.metric-big {
    background: #1e293b;
    border: 1px solid #334155;
    border-radius: 12px;
    padding: 1.2rem;
    text-align: center;
}
.metric-big .val { font-size: 2rem; font-weight: 700; color: #a5b4fc; }
.metric-big .lbl { font-size: 0.8rem; color: #64748b; margin-top: 2px; }
.metric-big .delta { font-size: 0.75rem; margin-top: 4px; }
.status-ok  { color: #34d399; }
.status-warn { color: #fbbf24; }
.status-bad { color: #f87171; }
.trace-row {
    background: #1e293b;
    border-radius: 8px;
    padding: 0.7rem 1rem;
    margin-bottom: 0.4rem;
    border-left: 3px solid #a5b4fc;
}
</style>
""", unsafe_allow_html=True)

recorder = MetricsRecorder()

# ─── Sidebar ──────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚙️ Dashboard Controls")
    hours = st.selectbox("Time Window", [1, 6, 24, 48, 168], index=2,
                         format_func=lambda h: f"Last {h}h" if h < 48 else f"Last {h//24}d")
    auto_refresh = st.toggle("Auto-refresh (30s)", value=False)
    if st.button("🔄 Refresh Now", use_container_width=True):
        st.rerun()

    st.markdown("---")
    st.markdown("### 🔗 Integrations")
    if LANGFUSE_ENABLED:
        st.success("✅ Langfuse connected")
        st.markdown("[Open Langfuse →](https://cloud.langfuse.com)")
    else:
        st.warning("⚠️ Langfuse not configured\nSet `LANGFUSE_PUBLIC_KEY` + `LANGFUSE_SECRET_KEY` in `.env`")

    st.markdown("---")
    st.markdown("### 📐 SLO Targets")
    st.markdown("- P95 Latency: **< 5 000 ms**")
    st.markdown("- Citation Coverage: **> 80%**")
    st.markdown("- Failure Rate: **< 5%**")
    st.markdown("- Cost / Request: **< $0.001**")

    if auto_refresh:
        time.sleep(30)
        st.rerun()

# ─── Load Data ────────────────────────────────────────────────────
stats = recorder.get_all_stats()
lat   = stats["latency"]
cost  = stats["cost"]
qual  = stats["quality"]
rows  = recorder.get_recent_requests(hours=hours, limit=200)

# ─── Header ───────────────────────────────────────────────────────
st.markdown('<h1 class="page-title">RAG Observability Dashboard</h1>', unsafe_allow_html=True)
st.markdown(f'<p style="color:#64748b">Production metrics for the AI Engineering Portfolio RAG · Last {hours}h · {qual["total_requests"]} requests</p>', unsafe_allow_html=True)

# ─── Top KPI Cards ───────────────────────────────────────────────
c1, c2, c3, c4, c5 = st.columns(5)

def kpi_card(col, value, label, status_cls="status-ok", delta=""):
    col.markdown(f"""
    <div class="metric-big">
        <div class="val {status_cls}">{value}</div>
        <div class="lbl">{label}</div>
        <div class="delta {status_cls}">{delta}</div>
    </div>""", unsafe_allow_html=True)

p95_cls = "status-ok" if lat["p95"] < 5000 else ("status-warn" if lat["p95"] < 8000 else "status-bad")
cit_cls = "status-ok" if qual["citation_pct"] >= 80 else ("status-warn" if qual["citation_pct"] >= 60 else "status-bad")
err_cls = "status-ok" if qual["failure_rate_pct"] < 5 else ("status-warn" if qual["failure_rate_pct"] < 15 else "status-bad")
cost_cls = "status-ok" if (cost["avg_usd"] or 0) < 0.001 else "status-warn"

kpi_card(c1, f"{lat['p50']:.0f} ms",   "P50 Latency",       "status-ok")
kpi_card(c2, f"{lat['p95']:.0f} ms",   "P95 Latency",       p95_cls,  "SLO: <5 000ms")
kpi_card(c3, f"{qual['citation_pct']}%", "Citation Coverage", cit_cls,  "SLO: >80%")
kpi_card(c4, f"{qual['failure_rate_pct']}%", "Failure Rate",  err_cls,  "SLO: <5%")
kpi_card(c5, f"${cost['avg_usd']:.5f}", "Avg Cost/Request",  cost_cls, f"Total: ${cost['total_usd']:.4f}")

st.markdown("<br>", unsafe_allow_html=True)

# ─── Charts Row 1 ─────────────────────────────────────────────────
col_lat, col_pie = st.columns([2, 1])

with col_lat:
    st.markdown("#### ⏱️ Latency Distribution (P50 / P95)")
    if rows:
        df = pd.DataFrame(rows)
        fig = px.histogram(
            df[df["is_error"] == 0], x="latency_ms",
            nbins=30, color_discrete_sequence=["#818cf8"],
            labels={"latency_ms": "Latency (ms)", "count": "Requests"},
        )
        # Add P50/P95 vertical lines
        for pct, val, color in [
            ("P50", lat["p50"], "#34d399"),
            ("P95", lat["p95"], "#f87171"),
        ]:
            fig.add_vline(x=val, line_dash="dash", line_color=color,
                          annotation_text=f" {pct}: {val:.0f}ms",
                          annotation_font_color=color)
        fig.update_layout(
            paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
            font_color="#94a3b8", height=300, margin=dict(t=20, b=20, l=20, r=20),
            xaxis=dict(gridcolor="#1e293b"), yaxis=dict(gridcolor="#1e293b"),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet. Run some queries first.")

with col_pie:
    st.markdown("#### 📌 Citation Quality")
    if qual["total_requests"] > 0:
        fig = go.Figure(go.Pie(
            labels=["Grounded ✅", "Declined ⚠️", "Errors ❌"],
            values=[qual["grounded"], qual["declined"], qual["errors"]],
            hole=0.6,
            marker_colors=["#34d399", "#fbbf24", "#f87171"],
        ))
        fig.update_traces(textposition="outside", textfont_size=12)
        fig.add_annotation(
            text=f"{qual['citation_pct']}%",
            x=0.5, y=0.5, font_size=22, font_color="#a5b4fc",
            showarrow=False
        )
        fig.update_layout(
            paper_bgcolor="#0f172a", font_color="#94a3b8",
            height=300, margin=dict(t=20, b=20, l=20, r=20),
            showlegend=True, legend=dict(font=dict(color="#94a3b8")),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No data yet.")

# ─── Charts Row 2: Cost & Latency Timeline ────────────────────────
st.markdown("#### 📈 Request Timeline")
if rows:
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=df["timestamp"], y=df["latency_ms"],
        mode="lines+markers", name="Latency (ms)",
        line=dict(color="#818cf8", width=2),
        marker=dict(size=4),
        yaxis="y1",
    ))
    fig.add_trace(go.Bar(
        x=df["timestamp"], y=df["cost_usd"] * 1e6,
        name="Cost (µ$)", opacity=0.4,
        marker_color="#f093fb",
        yaxis="y2",
    ))
    fig.update_layout(
        paper_bgcolor="#0f172a", plot_bgcolor="#0f172a",
        font_color="#94a3b8", height=280,
        margin=dict(t=20, b=20, l=50, r=50),
        xaxis=dict(gridcolor="#1e293b"),
        yaxis=dict(title="Latency (ms)", gridcolor="#1e293b", title_font_color="#818cf8"),
        yaxis2=dict(title="Cost (µ$)", overlaying="y", side="right",
                    title_font_color="#f093fb", showgrid=False),
        legend=dict(font=dict(color="#94a3b8")),
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("No timeline data yet.")

# ─── Trace Inspector ──────────────────────────────────────────────
st.markdown("---")
st.markdown("#### 🔬 Trace Inspector — Recent Requests")
if rows:
    df_display = pd.DataFrame(rows[:20])[[
        "timestamp", "latency_ms", "cost_usd", "prompt_tokens",
        "completion_tokens", "is_grounded", "is_error", "model", "question"
    ]].copy()
    df_display["latency_ms"] = df_display["latency_ms"].apply(lambda x: f"{x:.0f} ms")
    df_display["cost_usd"] = df_display["cost_usd"].apply(lambda x: f"${x:.6f}")
    df_display["is_grounded"] = df_display["is_grounded"].apply(lambda x: "✅ Yes" if x else "⚠️ No")
    df_display["is_error"] = df_display["is_error"].apply(lambda x: "❌ Error" if x else "—")
    df_display = df_display.rename(columns={
        "timestamp": "Time", "latency_ms": "Latency", "cost_usd": "Cost",
        "prompt_tokens": "Prompt Tok.", "completion_tokens": "Comp. Tok.",
        "is_grounded": "Grounded?", "is_error": "Error?",
        "model": "Model", "question": "Question"
    })
    st.dataframe(df_display, use_container_width=True, hide_index=True)
else:
    st.info("No requests recorded yet. Run the instrumented pipeline to start generating metrics.")
    st.markdown("""
    **To generate sample data, run:**
    ```bash
    cd 03-monitoring
    python seed_metrics.py
    ```
    """)
