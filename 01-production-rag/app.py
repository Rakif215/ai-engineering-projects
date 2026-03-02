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
    page_title="Ask My Docs — Production RAG",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─── Custom CSS ────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
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
    .stChatMessage { border-radius: 12px; }
    .sidebar-info {
        background: linear-gradient(135deg, #1e1b4b 0%, #312e81 100%);
        border-radius: 12px;
        padding: 1rem;
        color: white;
        margin-bottom: 1rem;
    }
    .sidebar-info h4 { color: #a5b4fc; margin-bottom: 0.5rem; }
    .sidebar-info p { color: #c7d2fe; font-size: 0.85rem; }
    .metric-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 10px;
        padding: 0.8rem;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .metric-card h3 { color: #a5b4fc; font-size: 1.5rem; margin: 0; }
    .metric-card p { color: #94a3b8; font-size: 0.75rem; margin: 0; }
    .doc-chip {
        display: inline-block;
        background: #312e81;
        color: #c7d2fe;
        padding: 4px 10px;
        border-radius: 20px;
        font-size: 0.75rem;
        margin: 2px;
    }
</style>
""", unsafe_allow_html=True)

# ─── Session State Initialization ──────────────────────────────
if "pipeline_ready" not in st.session_state:
    st.session_state.pipeline_ready = False
if "messages" not in st.session_state:
    st.session_state.messages = []
if "generation_chain" not in st.session_state:
    st.session_state.generation_chain = None
if "final_retriever" not in st.session_state:
    st.session_state.final_retriever = None
if "doc_count" not in st.session_state:
    st.session_state.doc_count = 0
if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0
if "doc_names" not in st.session_state:
    st.session_state.doc_names = []

# ─── Pipeline Builder ─────────────────────────────────────────
def build_pipeline(uploaded_files, llm_provider):
    """Ingests uploaded documents, chunks them, builds the full retrieval pipeline."""
    ingestor = DocumentIngestor()
    chunker = DocumentChunker()
    all_docs = []
    doc_names = []

    for uploaded_file in uploaded_files:
        # Save uploaded file to a temporary location
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.getbuffer())
            tmp_path = tmp.name

        # Load document based on type
        if suffix.lower() == ".pdf":
            docs = ingestor.load_pdf(tmp_path)
        elif suffix.lower() == ".md":
            docs = ingestor.load_markdown(tmp_path)
        else:
            st.warning(f"Skipping unsupported file: {uploaded_file.name}")
            os.unlink(tmp_path)
            continue

        all_docs.extend(docs)
        doc_names.append(uploaded_file.name)
        os.unlink(tmp_path)  # Clean up temp file

    if not all_docs:
        st.error("No documents were successfully loaded.")
        return False

    # Chunk
    chunks = chunker.chunk_documents(all_docs)

    # Vector Store
    vectorstore = VectorStoreManager(collection_name="streamlit_rag")
    vectorstore.add_documents(chunks)
    base_retriever = vectorstore.get_retriever(k=10)

    # Hybrid Search + Reranking
    hybrid_manager = HybridRetrieverManager(base_retriever, chunks)
    hybrid_retriever = hybrid_manager.get_retriever()

    rerank_manager = RerankingManager(hybrid_retriever, top_n=3)
    final_retriever = rerank_manager.get_retriever()

    # Generation Chain
    generation_chain = GenerationChain(llm_provider=llm_provider)

    # Store in session
    st.session_state.final_retriever = final_retriever
    st.session_state.generation_chain = generation_chain
    st.session_state.pipeline_ready = True
    st.session_state.doc_count = len(doc_names)
    st.session_state.chunk_count = len(chunks)
    st.session_state.doc_names = doc_names
    st.session_state.messages = []  # Reset chat on new docs

    return True

# ─── Sidebar ───────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-info"><h4>📄 Ask My Docs</h4><p>Production-Grade RAG with Hybrid Retrieval, Cross-Encoder Reranking, Citation Enforcement, and Streaming Generation.</p></div>', unsafe_allow_html=True)

    st.markdown("### ⚙️ Configuration")

    llm_provider = st.selectbox(
        "LLM Provider",
        ["groq", "gemini"],
        index=0,
        help="Groq = fast (Llama 3.1), Gemini = deep reasoning"
    )

    st.markdown("### 📁 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload PDFs or Markdown files",
        type=["pdf", "md"],
        accept_multiple_files=True,
        help="Upload one or more documents to query against"
    )

    if uploaded_files:
        if st.button("🚀 Build Pipeline", use_container_width=True, type="primary"):
            with st.spinner("Ingesting, chunking, embedding, and indexing..."):
                success = build_pipeline(uploaded_files, llm_provider)
                if success:
                    st.success(f"✅ Pipeline ready! {st.session_state.doc_count} doc(s), {st.session_state.chunk_count} chunks.")
                    st.rerun()

    # Show pipeline status
    if st.session_state.pipeline_ready:
        st.markdown("---")
        st.markdown("### 📊 Pipeline Status")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f'<div class="metric-card"><h3>{st.session_state.doc_count}</h3><p>Documents</p></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><h3>{st.session_state.chunk_count}</h3><p>Chunks</p></div>', unsafe_allow_html=True)

        st.markdown("**Loaded Files:**")
        for name in st.session_state.doc_names:
            st.markdown(f'<span class="doc-chip">📄 {name}</span>', unsafe_allow_html=True)

        st.markdown("---")
        if st.button("🗑️ Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            if st.session_state.generation_chain:
                st.session_state.generation_chain.clear_history()
            st.rerun()

    # Architecture info
    st.markdown("---")
    st.markdown("### 🏗️ Architecture")
    st.markdown("""
    ```
    PDF/MD Upload
        ↓
    Chunking (800 tokens)
        ↓
    ChromaDB + BM25
        ↓
    Ensemble Retriever
        ↓
    Cohere Reranker (Top 3)
        ↓
    Groq/Gemini Generation
        ↓
    Cited Answer (Streamed)
    ```
    """)

# ─── Main Chat Area ───────────────────────────────────────────
st.markdown('<h1 class="main-header">Ask My Docs</h1>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Production-Grade RAG with Hybrid Retrieval • Cross-Encoder Reranking • Citation Enforcement</p>', unsafe_allow_html=True)

if not st.session_state.pipeline_ready:
    st.info("👈 Upload one or more documents in the sidebar and click **Build Pipeline** to get started.")
else:
    # Display previous messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):
        # Display user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Stream the assistant response
        with st.chat_message("assistant"):
            try:
                response_stream = st.session_state.generation_chain.stream(
                    prompt, st.session_state.final_retriever
                )
                response = st.write_stream(response_stream)
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"⚠️ Error: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
