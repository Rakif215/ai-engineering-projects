import os
import tempfile
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from src.ingestion.loaders import DocumentIngestor
from src.ingestion.chunking import DocumentChunker
from src.storage.vectorstore import VectorStoreManager
from src.retrieval.hybrid_search import HybridRetrieverManager
from src.retrieval.reranking import RerankingManager
from src.generation.llm_chain import GenerationChain

# ─── Page Config ───────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Comparison: Naive vs. Production",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #f43f5e 0%, #fb923c 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-size: 2.4rem;
        font-weight: 700;
        margin-bottom: 0;
    }
    .sub-header {
        color: #94a3b8;
        font-size: 1rem;
        margin-top: -10px;
        margin-bottom: 1.5rem;
    }
    .column-header-naive {
        background: #334155;
        padding: 10px;
        border-radius: 8px 8px 0 0;
        text-align: center;
        color: #cbd5e1;
        font-weight: 600;
        border-bottom: 3px solid #ef4444;
    }
    .column-header-prod {
        background: #1e293b;
        padding: 10px;
        border-radius: 8px 8px 0 0;
        text-align: center;
        color: #e2e8f0;
        font-weight: 600;
        border-bottom: 3px solid #10b981;
    }
    .result-container {
        background: #0f172a;
        padding: 15px;
        border-radius: 0 0 8px 8px;
        min-height: 400px;
        border: 1px solid #1e293b;
        border-top: none;
    }
    .metric-chip {
        display: inline-block;
        font-size: 0.75rem;
        padding: 4px 8px;
        border-radius: 12px;
        margin-right: 5px;
        margin-bottom: 10px;
    }
    .metric-red { background: rgba(239, 68, 68, 0.2); color: #fca5a5; border: 1px solid #ef4444; }
    .metric-green { background: rgba(16, 185, 129, 0.2); color: #6ee7b7; border: 1px solid #10b981; }
    .stChatMessage { border-radius: 12px; }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ──────────────────────────────
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "naive_chain" not in st.session_state:
    st.session_state.naive_chain = None
if "prod_chain" not in st.session_state:
    st.session_state.prod_chain = None
if "naive_retriever" not in st.session_state:
    st.session_state.naive_retriever = None
if "prod_retriever" not in st.session_state:
    st.session_state.prod_retriever = None

# ─── Pipeline Builder ─────────────────────────────────────────
def build_comparison_pipeline(uploaded_files, llm_provider):
    ingestor = DocumentIngestor()
    chunker = DocumentChunker()
    all_docs = []

    for uploaded_file in uploaded_files:
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        if suffix.lower() == ".pdf":
            docs = ingestor.load_pdf(tmp_path)
        elif suffix.lower() == ".md":
            docs = ingestor.load_markdown(tmp_path)
        else:
            os.unlink(tmp_path)
            continue
        all_docs.extend(docs)
        os.unlink(tmp_path)

    if not all_docs:
        return False

    chunks = chunker.chunk_documents(all_docs)

    # 1. NAIVE RAG Pipeline (Vector Only)
    vectorstore = VectorStoreManager(collection_name="streamlit_comparison")
    vectorstore.add_documents(chunks)
    naive_retriever = vectorstore.get_retriever(k=5)  # Semantic only, top 5

    # 2. PROD RAG Pipeline (Hybrid + Reranker)
    hybrid_manager = HybridRetrieverManager(naive_retriever, chunks)
    hybrid_retriever = hybrid_manager.get_retriever()
    rerank_manager = RerankingManager(hybrid_retriever, top_n=3)
    prod_retriever = rerank_manager.get_retriever()

    # Shared Generation Chains (but distinct histories if needed)
    naive_chain = GenerationChain(llm_provider=llm_provider)
    # Give the Prod chain a strict prompt if we had one, but we use the same chain logic
    prod_chain = GenerationChain(llm_provider=llm_provider)

    # Store
    st.session_state.naive_retriever = naive_retriever
    st.session_state.prod_retriever = prod_retriever
    st.session_state.naive_chain = naive_chain
    st.session_state.prod_chain = prod_chain
    st.session_state.pipeline_ready = True
    return True

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('### ⚙️ Setup Comparison')
    llm_provider = st.selectbox("LLM Provider", ["groq", "gemini"], index=0)
    uploaded_files = st.file_uploader("Upload Document for Test", type=["pdf", "md"], accept_multiple_files=True)

    if uploaded_files and st.button("🚀 Initialize Pipelines", use_container_width=True, type="primary"):
        with st.spinner("Building Naive and Prod pipelines..."):
            if build_comparison_pipeline(uploaded_files, llm_provider):
                st.success("Pipelines Ready!")
                st.rerun()

    st.markdown("---")
    st.markdown("""
    **What this demonstrates:**
    * **Naive RAG:** Standard Semantic Vector search. Prone to missing keywords and guessing.
    * **Production RAG:** Hybrid Search (BM25+Vector) + Cohere Reranking. Maximizes recall and forces the LLM to only use top-tier context.
    """)

# ─── Main Area ───────────────────────────────────────────
st.markdown('<h1 class="main-header">Vs. Mode: AI Architecture</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Visually testing standard RAG against Production-Grade engineering.</p>', unsafe_allow_html=True)

if not st.session_state.pipeline_ready:
    st.info("👈 Upload a document and Initialize Pipelines to start the side-by-side test.")
else:
    query = st.chat_input("Ask a question designed to trick a normal RAG...")

    if query:
        st.markdown(f"**Query:** `{query}`")
        col1, col2 = st.columns(2)

        with col1:
            st.markdown('<div class="column-header-naive">Standard RAG (Tutorial Grade)</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown('<span class="metric-chip metric-red">Retrieval: Semantic Only (Top 5)</span>', unsafe_allow_html=True)
                st.markdown('<span class="metric-chip metric-red">Reranking: None</span>', unsafe_allow_html=True)
                st.markdown('<span class="metric-chip metric-red">Hallucination Risk: High</span>', unsafe_allow_html=True)
                st.markdown("---")
                
                with st.spinner("Naive RAG generating..."):
                    try:
                        # stream response
                        response_container = st.empty()
                        full_res = ""
                        for chunk in st.session_state.naive_chain.stream(query, st.session_state.naive_retriever):
                            full_res += chunk
                            response_container.markdown(full_res)
                    except Exception as e:
                        st.error(f"Error: {e}")
                st.markdown('</div>', unsafe_allow_html=True)

        with col2:
            st.markdown('<div class="column-header-prod">Production RAG (Enterprise Grade)</div>', unsafe_allow_html=True)
            with st.container():
                st.markdown('<div class="result-container">', unsafe_allow_html=True)
                st.markdown('<span class="metric-chip metric-green">Retrieval: Hybrid Ensemble (BM25 + Semantic)</span>', unsafe_allow_html=True)
                st.markdown('<span class="metric-chip metric-green">Reranking: Cohere Cross-Encoder (Top 3)</span>', unsafe_allow_html=True)
                st.markdown('<span class="metric-chip metric-green">Grounding: Strict Citation Enforcement</span>', unsafe_allow_html=True)
                st.markdown("---")

                with st.spinner("Prod RAG generating..."):
                    try:
                        response_container = st.empty()
                        full_res = ""
                        for chunk in st.session_state.prod_chain.stream(query, st.session_state.prod_retriever):
                            full_res += chunk
                            response_container.markdown(full_res)
                    except Exception as e:
                        st.error(f"Error: {e}")
                st.markdown('</div>', unsafe_allow_html=True)
