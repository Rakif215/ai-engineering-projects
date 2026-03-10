import { useState } from 'react'
import { Send, CheckCircle2, AlertCircle, RefreshCw, ChevronDown, ChevronUp, Shield, Search, Brain } from 'lucide-react'
import './App.css'

interface TraceItem {
  content: string;
  score: string | number;
}

function App() {
  const [query, setQuery] = useState('')
  const [isGenerating, setIsGenerating] = useState(false)
  const [hasSearched, setHasSearched] = useState(false)
  
  const [naiveResponse, setNaiveResponse] = useState('')
  const [naiveTrace, setNaiveTrace] = useState<TraceItem[]>([])
  const [showNaiveTrace, setShowNaiveTrace] = useState(false)
  
  const [prodResponse, setProdResponse] = useState('')
  const [prodTrace, setProdTrace] = useState<TraceItem[]>([])
  const [showProdTrace, setShowProdTrace] = useState(false)

  const handleSearch = async (overrideQuery?: string) => {
    const q = overrideQuery || query
    if (!q.trim()) return

    setQuery(q)
    setHasSearched(true)
    setIsGenerating(true)
    setNaiveResponse('')
    setNaiveTrace([])
    setProdResponse('')
    setProdTrace([])

    try {
      const [nRes, pRes] = await Promise.all([
        fetch('http://localhost:8000/query/naive', {
          method: 'POST',
          headers:{'Content-Type': 'application/json'},
          body: JSON.stringify({query: q})
        }).then(r => r.json()),
        fetch('http://localhost:8000/query/prod', {
          method: 'POST',
          headers:{'Content-Type': 'application/json'},
          body: JSON.stringify({query: q})
        }).then(r => r.json())
      ])
      
      setNaiveResponse(nRes.response)
      setNaiveTrace(nRes.trace || [])
      setProdResponse(pRes.response)
      setProdTrace(pRes.trace || [])
    } catch {
      setNaiveResponse("Error connecting to backend.")
      setProdResponse("Error connecting to backend.")
    }

    setIsGenerating(false)
  }

  const isRefusal = (text: string) => text.includes("I cannot answer") || text.includes("unsafe")
  
  const renderTrace = (traces: TraceItem[], isProd: boolean) => {
    if (!traces || traces.length === 0) return <div className="trace-item empty">No chunks retrieved</div>
    return traces.map((t, idx) => (
      <div key={idx} className="trace-item">
        <div className="trace-header">
          <span className="trace-label">Chunk {idx + 1}</span>
          <span className="trace-score">
            {isProd && typeof t.score === 'number' ? (
              <span className={`score-pill ${t.score > 0.8 ? 'high' : t.score > 0.5 ? 'med' : 'low'}`}>
                Relevance: {t.score}
              </span>
            ) : (
              <span className="score-pill neutral">No Reranking</span>
            )}
          </span>
        </div>
        <p className="trace-content">{t.content}</p>
      </div>
    ))
  }

  return (
    <div className="app-container">
      <header className="header">
        <div className="logo">
          <Shield className="logo-icon" size={32} />
          <h1>Veritas</h1>
        </div>
        <p className="tagline">Medical Protocol QA — built for safety, not speed.</p>
      </header>

      {/* QUERY BAR */}
      <div className="query-section">
        <form onSubmit={(e) => { e.preventDefault(); handleSearch(); }} className="search-form">
          <Search size={20} className="search-icon" />
          <input 
            type="text" 
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            placeholder="Ask a clinical question..."
            className="search-input"
          />
          <button type="submit" className="search-button" disabled={!query.trim() || isGenerating}>
            <Send size={18} />
          </button>
        </form>
        <div className="try-section">
          <span className="try-label">Try:</span>
          <button className="try-btn" onClick={() => handleSearch("What is the target blood pressure for most adults according to the protocol?")}>
            Blood pressure targets
          </button>
          <button className="try-btn" onClick={() => handleSearch("What is the recommended dosage for off-label use of Drug X in pediatric patients?")}>
            Pediatric Drug X dosage
          </button>
          <button className="try-btn" onClick={() => handleSearch("What are the first-line medications for hypertension management?")}>
            Hypertension medications
          </button>
        </div>
      </div>

      {/* RESULTS */}
      {!hasSearched ? (
        <div className="empty-state">
          <Brain size={56} className="empty-icon" />
          <h2>Clinical guidelines loaded</h2>
          <p>Ask a question above, or try one of the suggested queries to see how Veritas handles missing context.</p>
        </div>
      ) : (
        <div className="comparison-grid">
          {/* STANDARD RAG */}
          <div className="rag-column naive-column">
            <div className="column-header">
              <div className="column-title">
                <h2>Standard RAG</h2>
                <span className="column-sub">Semantic search → LLM</span>
              </div>
            </div>
            
            <div className="pipeline-steps">
              <div className="step bad"><AlertCircle size={14} /> Semantic search only</div>
              <div className="step bad"><AlertCircle size={14} /> No reranking</div>
              <div className="step bad"><AlertCircle size={14} /> No grounding check</div>
            </div>

            {/* Trace */}
            <button className="trace-toggle" onClick={() => setShowNaiveTrace(!showNaiveTrace)}>
              Retrieved Chunks ({naiveTrace.length}) {showNaiveTrace ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {showNaiveTrace && (
              <div className="trace-container">
                {renderTrace(naiveTrace, false)}
              </div>
            )}

            <div className="response-area">
              {isGenerating && !naiveResponse ? (
                <div className="loading"><RefreshCw className="spinner" size={16} /> Generating...</div>
              ) : naiveResponse && (
                <>
                  <div className="verdict bad-verdict">
                    <AlertCircle size={14} /> Unverified
                  </div>
                  <div className="response-text">{naiveResponse}</div>
                </>
              )}
            </div>
          </div>

          {/* VERITAS */}
          <div className="rag-column prod-column">
            <div className="column-header">
              <div className="column-title">
                <h2>Veritas</h2>
                <span className="column-sub">Hybrid search → Reranker → Grounding gate</span>
              </div>
            </div>
            
            <div className="pipeline-steps">
              <div className="step good"><CheckCircle2 size={14} /> BM25 + Semantic hybrid</div>
              <div className="step good"><CheckCircle2 size={14} /> Cross-encoder reranking</div>
              <div className="step good"><CheckCircle2 size={14} /> Citation enforcement</div>
            </div>

            {/* Trace */}
            <button className="trace-toggle" onClick={() => setShowProdTrace(!showProdTrace)}>
              Retrieved Chunks ({prodTrace.length}) {showProdTrace ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
            </button>
            {showProdTrace && (
              <div className="trace-container">
                {renderTrace(prodTrace, true)}
              </div>
            )}

            <div className="response-area">
              {isGenerating && !prodResponse ? (
                <div className="loading"><RefreshCw className="spinner" size={16} /> Analyzing context...</div>
              ) : prodResponse && (
                <>
                  {isRefusal(prodResponse) ? (
                    <div className="verdict refusal-verdict">
                      <Shield size={14} /> Refused — insufficient evidence
                    </div>
                  ) : (
                    <div className="verdict good-verdict">
                      <CheckCircle2 size={14} /> Grounded
                    </div>
                  )}
                  <div className="response-text">{prodResponse}</div>
                </>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App
