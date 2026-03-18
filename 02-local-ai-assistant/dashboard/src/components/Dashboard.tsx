import React, { useState, useEffect } from 'react';
import { 
  Settings, 
  Activity, 
  Database, 
  BarChart3, 
  Play, 
  RefreshCw,
  CheckCircle2,
  XCircle,
  AlertTriangle
} from 'lucide-react';
import { z } from 'zod';
import { getModels, generate, generateWithRetry, OllamaModel, OllamaMetrics } from '../lib/ollama';
import { generateOpenRouter, generateWithRetryOpenRouter } from '../lib/openrouter';
import { cn } from '../lib/utils';
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer
} from 'recharts';

// Pre-defined JSON Schema for testing
const SentimentSchema = z.object({
  summary: z.string(),
  sentiment: z.enum(['positive', 'neutral', 'negative']),
  confidence: z.number().min(0).max(1)
});

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'setup' | 'single' | 'batch'>('setup');
  const [baseUrl, setBaseUrl] = useState('http://localhost:11434');
  const [openRouterKey, setOpenRouterKey] = useState('');
  const [models, setModels] = useState<OllamaModel[]>([]);
  const [loadingModels, setLoadingModels] = useState(false);
  const [connectionError, setConnectionError] = useState('');

  // Single Run State
  const [selectedModel, setSelectedModel] = useState('');
  const [prompt, setPrompt] = useState('Write a short summary of the movie Inception and analyze its sentiment.');
  const [temperature, setTemperature] = useState(0.7);
  const [enforceJson, setEnforceJson] = useState(false);
  const [singleRunning, setSingleRunning] = useState(false);
  const [singleResult, setSingleResult] = useState<{
    text: string;
    metrics: OllamaMetrics;
    parsedData?: any;
    retries?: number;
    error?: string;
  } | null>(null);

  // Batch Run State
  const [batchPrompts, setBatchPrompts] = useState(
    'EXTRACT: "Anthropic raised $4B from Amazon." Return JSON: {company: string, amount: string}\nCLASSIFY SENTIMENT: "The battery life is okay, but the screen cracked after a week." Reply with only the word POSITIVE or NEGATIVE.\nFORMAT: "milk eggs bread" as an HTML unordered list. No markdown blocks.'
  );
  const [selectedBatchModels, setSelectedBatchModels] = useState<string[]>([]);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchResults, setBatchResults] = useState<any[]>([]);

  const fetchModels = async () => {
    setLoadingModels(true);
    setConnectionError('');
    try {
      const m = await getModels(baseUrl);
      
      // Inject OpenRouter baseline models
      if (openRouterKey) {
        m.push({
          name: 'openrouter/google/gemini-2.5-flash',
          size: 0,
          digest: '',
          details: { parameter_size: 'Cloud (Google)', quantization_level: 'FP16' }
        });
        m.push({
          name: 'openrouter/meta-llama/llama-3.1-8b-instruct',
          size: 0,
          digest: '',
          details: { parameter_size: 'Cloud (Meta 8B)', quantization_level: 'FP16' }
        });
      }

      setModels(m);
      if (m.length > 0 && !selectedModel) setSelectedModel(m[0].name);
    } catch (e: any) {
      setConnectionError(e.message || 'Failed to connect to Ollama. Make sure it is running with OLLAMA_ORIGINS="*"');
    } finally {
      setLoadingModels(false);
    }
  };

  useEffect(() => {
    fetchModels();
  }, []);

  const runSingle = async () => {
    if (!selectedModel) return;
    setSingleRunning(true);
    setSingleResult(null);
    try {
      const isOpenRouter = selectedModel.startsWith('openrouter/');
      const actualModel = isOpenRouter ? selectedModel.split('openrouter/')[1] : selectedModel;

      if (enforceJson) {
        const res = isOpenRouter 
          ? await generateWithRetryOpenRouter(openRouterKey, actualModel, prompt, SentimentSchema, { temperature })
          : await generateWithRetry(baseUrl, actualModel, prompt, SentimentSchema, { temperature });
          
        setSingleResult({
          text: res.raw,
          metrics: res.metrics as OllamaMetrics,
          parsedData: res.data,
          retries: res.retries,
          error: res.error
        });
      } else {
        const res = isOpenRouter
          ? await generateOpenRouter(openRouterKey, actualModel, prompt, { temperature })
          : await generate(baseUrl, actualModel, prompt, { temperature });
          
        setSingleResult({
          text: res.text,
          metrics: res.metrics as OllamaMetrics,
          error: res.error
        });
      }
    } catch (e: any) {
      setSingleResult({ text: '', metrics: { ttft: 0, totalLatency: 0, tps: 0, evalCount: 0 }, error: e.message });
    } finally {
      setSingleRunning(false);
    }
  };

  const runBatch = async () => {
    if (selectedBatchModels.length === 0) return;
    const prompts = batchPrompts.split('\n').filter(p => p.trim());
    if (prompts.length === 0) return;

    setBatchRunning(true);
    setBatchResults([]);

    const results = [];
    for (const model of selectedBatchModels) {
      const isOpenRouter = model.startsWith('openrouter/');
      const actualModel = isOpenRouter ? model.split('openrouter/')[1] : model;

      for (let i = 0; i < prompts.length; i++) {
        const p = prompts[i];
        try {
          const res = isOpenRouter
            ? await generateOpenRouter(openRouterKey, actualModel, p, { temperature: 0 })
            : await generate(baseUrl, actualModel, p, { temperature: 0 }); // Use 0 for deterministic benchmarking
            
          const displayName = isOpenRouter 
            ? model.split('/').slice(-1)[0].replace('-instruct', '') + ' (Cloud)'
            : model.replace(':latest', '');

          results.push({
            model,
            displayName,
            promptIndex: i + 1,
            ttft: res.metrics.ttft,
            tps: res.metrics.tps,
            latency: res.metrics.totalLatency,
            success: !res.error
          });
          setBatchResults([...results]);
        } catch (e) {
          const displayName = isOpenRouter 
            ? model.split('/').slice(-1)[0].replace('-instruct', '') + ' (Cloud)'
            : model.replace(':latest', '');

          results.push({
            model,
            displayName,
            promptIndex: i + 1,
            ttft: 0,
            tps: 0,
            latency: 0,
            success: false
          });
          setBatchResults([...results]);
        }
      }
    }
    setBatchRunning(false);
  };

  const toggleBatchModel = (model: string) => {
    setSelectedBatchModels(prev => 
      prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]
    );
  };

  // Aggregate batch results for charts
  const aggregatedResults = selectedBatchModels.map(model => {
    const modelResults = batchResults.filter(r => r.model === model && r.success);
    // Shorten model name for chart display
    const displayName = model.startsWith('openrouter/') 
      ? model.split('/').slice(-1)[0].replace('-instruct', '') 
      : model.replace(':latest', '');
      
    if (modelResults.length === 0) return { model, displayName: modelResults[0]?.displayName || model, avgTtft: 0, avgTps: 0, avgLatency: 0 };
    return {
      model,
      displayName: modelResults[0].displayName,
      avgTtft: modelResults.reduce((sum, r) => sum + r.ttft, 0) / modelResults.length,
      avgTps: modelResults.reduce((sum, r) => sum + r.tps, 0) / modelResults.length,
      avgLatency: modelResults.reduce((sum, r) => sum + r.latency, 0) / modelResults.length,
    };
  });

  return (
    <div className="flex h-screen bg-[#141414] text-[#E4E3E0] font-sans selection:bg-indigo-500/30">
      {/* Sidebar */}
      <div className="w-64 border-r border-[#2A2A2A] bg-[#0F0F0F] flex flex-col">
        <div className="p-6 border-b border-[#2A2A2A]">
          <h1 className="text-xl font-bold tracking-tight flex items-center gap-2">
            <Activity className="w-5 h-5 text-indigo-400" />
            SLM Bench
          </h1>
          <p className="text-xs text-gray-500 mt-1 font-mono">Local Inference Tester</p>
        </div>
        <nav className="flex-1 p-4 space-y-2">
          <button 
            onClick={() => setActiveTab('setup')}
            className={cn("w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors", activeTab === 'setup' ? "bg-[#2A2A2A] text-white" : "text-gray-400 hover:bg-[#1A1A1A] hover:text-gray-200")}
          >
            <Settings className="w-4 h-4" /> Connection
          </button>
          <button 
            onClick={() => setActiveTab('single')}
            className={cn("w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors", activeTab === 'single' ? "bg-[#2A2A2A] text-white" : "text-gray-400 hover:bg-[#1A1A1A] hover:text-gray-200")}
          >
            <Play className="w-4 h-4" /> Single Run
          </button>
          <button 
            onClick={() => setActiveTab('batch')}
            className={cn("w-full flex items-center gap-3 px-4 py-3 rounded-lg text-sm font-medium transition-colors", activeTab === 'batch' ? "bg-[#2A2A2A] text-white" : "text-gray-400 hover:bg-[#1A1A1A] hover:text-gray-200")}
          >
            <Database className="w-4 h-4" /> Batch Benchmark
          </button>
        </nav>
        <div className="p-4 border-t border-[#2A2A2A]">
          <div className="flex items-center gap-2 text-xs font-mono text-gray-500">
            <div className={cn("w-2 h-2 rounded-full", models.length > 0 ? "bg-emerald-500" : "bg-red-500")} />
            {models.length > 0 ? `Connected (${models.length} models)` : 'Disconnected'}
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 overflow-auto">
        {activeTab === 'setup' && (
          <div className="p-8 max-w-3xl mx-auto">
            <h2 className="text-2xl font-semibold mb-6">Ollama Connection Setup</h2>
            <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-xl p-6 mb-6">
              <label className="block text-sm font-medium text-gray-400 mb-2">Ollama Base URL</label>
              <div className="flex gap-3 mb-4">
                <input 
                  type="text" 
                  value={baseUrl}
                  onChange={(e) => setBaseUrl(e.target.value)}
                  className="flex-1 bg-[#0F0F0F] border border-[#333] rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-indigo-500 font-mono"
                />
              </div>

              <label className="block text-sm font-medium text-gray-400 mb-2 mt-4">OpenRouter API Key (Optional Baseline)</label>
              <div className="flex gap-3">
                <input 
                  type="password" 
                  value={openRouterKey}
                  onChange={(e) => setOpenRouterKey(e.target.value)}
                  placeholder="sk-or-..."
                  className="flex-1 bg-[#0F0F0F] border border-[#333] rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-indigo-500 font-mono"
                />
                <button 
                  onClick={fetchModels}
                  disabled={loadingModels}
                  className="bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-2 rounded-lg text-sm font-medium flex items-center gap-2 disabled:opacity-50"
                >
                  <RefreshCw className={cn("w-4 h-4", loadingModels && "animate-spin")} />
                  Connect
                </button>
              </div>
              {connectionError && (
                <div className="mt-4 p-4 bg-red-500/10 border border-red-500/20 rounded-lg flex items-start gap-3 text-red-400 text-sm">
                  <AlertTriangle className="w-5 h-5 shrink-0" />
                  <div>
                    <p className="font-semibold mb-1">Connection Failed: {connectionError}</p>
                    <p className="mb-2">Your browser blocked the request to localhost, or Ollama is not running. You <strong>must</strong> enable CORS in Ollama for this web app to connect to your local machine.</p>
                    <div className="mt-3 text-xs font-mono bg-black/50 p-3 rounded space-y-2">
                      <p className="text-gray-400 uppercase tracking-wider text-[10px]">Mac / Linux</p>
                      <p>OLLAMA_ORIGINS="*" ollama serve</p>
                      <div className="h-px bg-[#333] my-2" />
                      <p className="text-gray-400 uppercase tracking-wider text-[10px]">Windows (Command Prompt)</p>
                      <p>set OLLAMA_ORIGINS="*" && ollama serve</p>
                      <div className="h-px bg-[#333] my-2" />
                      <p className="text-gray-400 uppercase tracking-wider text-[10px]">Windows (PowerShell)</p>
                      <p>$env:OLLAMA_ORIGINS="*" ; ollama serve</p>
                    </div>
                    <p className="mt-3 text-xs text-gray-400">
                      Note: If Ollama is already running as a background service, you must stop it first before running the command above. Also try changing localhost to 127.0.0.1 in the URL.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {models.length > 0 && (
              <div>
                <h3 className="text-lg font-medium mb-4">Available Models</h3>
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {models.map(m => (
                    <div key={m.name} className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-lg p-4 flex items-center justify-between">
                      <div>
                        <div className="font-medium">{m.name}</div>
                        <div className="text-xs text-gray-500 font-mono mt-1">
                          {m.details.parameter_size} • {m.details.quantization_level}
                        </div>
                      </div>
                      <Database className="w-4 h-4 text-gray-600" />
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {activeTab === 'single' && (
          <div className="p-8 max-w-5xl mx-auto grid grid-cols-1 lg:grid-cols-2 gap-8">
            <div className="space-y-6">
              <h2 className="text-2xl font-semibold">Single Run & Validation</h2>
              
              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">Model</label>
                <select 
                  value={selectedModel}
                  onChange={(e) => setSelectedModel(e.target.value)}
                  className="w-full bg-[#1A1A1A] border border-[#333] rounded-lg px-4 py-2 text-sm focus:outline-none focus:border-indigo-500 appearance-none"
                >
                  <option value="">Select a model...</option>
                  {models.map(m => <option key={m.name} value={m.name}>{m.name}</option>)}
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-400 mb-2">Prompt</label>
                <textarea 
                  value={prompt}
                  onChange={(e) => setPrompt(e.target.value)}
                  rows={4}
                  className="w-full bg-[#1A1A1A] border border-[#333] rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 resize-none font-mono"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-400 mb-2">Temperature: {temperature}</label>
                  <input 
                    type="range" 
                    min="0" max="1" step="0.1" 
                    value={temperature}
                    onChange={(e) => setTemperature(parseFloat(e.target.value))}
                    className="w-full accent-indigo-500"
                  />
                </div>
                <div className="flex items-center justify-end">
                  <label className="flex items-center gap-3 cursor-pointer">
                    <span className="text-sm font-medium text-gray-400">Enforce JSON Schema</span>
                    <div className={cn("w-10 h-6 rounded-full transition-colors relative", enforceJson ? "bg-indigo-600" : "bg-[#333]")}>
                      <input type="checkbox" className="sr-only" checked={enforceJson} onChange={(e) => setEnforceJson(e.target.checked)} />
                      <div className={cn("absolute top-1 left-1 bg-white w-4 h-4 rounded-full transition-transform", enforceJson ? "translate-x-4" : "")} />
                    </div>
                  </label>
                </div>
              </div>

              {enforceJson && (
                <div className="bg-[#1A1A1A] border border-[#333] rounded-lg p-4">
                  <div className="text-xs font-medium text-gray-500 mb-2 uppercase tracking-wider">Target Schema (Zod)</div>
                  <pre className="text-xs font-mono text-indigo-300 overflow-x-auto">
{`{
  "summary": "string",
  "sentiment": "positive|neutral|negative",
  "confidence": "number (0-1)"
}`}
                  </pre>
                </div>
              )}

              <button 
                onClick={runSingle}
                disabled={singleRunning || !selectedModel}
                className="w-full bg-indigo-600 hover:bg-indigo-700 text-white px-4 py-3 rounded-lg text-sm font-medium flex items-center justify-center gap-2 disabled:opacity-50"
              >
                {singleRunning ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {singleRunning ? 'Generating...' : 'Run Inference'}
              </button>
            </div>

            <div className="bg-[#0A0A0A] border border-[#2A2A2A] rounded-xl flex flex-col overflow-hidden">
              <div className="p-4 border-b border-[#2A2A2A] bg-[#141414] flex items-center justify-between">
                <h3 className="font-medium flex items-center gap-2">
                  <BarChart3 className="w-4 h-4 text-gray-400" /> Results
                </h3>
                {singleResult && (
                  <div className="flex items-center gap-4 text-xs font-mono">
                    <span className="text-gray-400">TTFT: <span className="text-white">{(singleResult.metrics.ttft / 1000).toFixed(2)}s</span></span>
                    <span className="text-gray-400">TPS: <span className="text-white">{singleResult.metrics.tps.toFixed(1)}</span></span>
                    <span className="text-gray-400">Total: <span className="text-white">{(singleResult.metrics.totalLatency / 1000).toFixed(2)}s</span></span>
                  </div>
                )}
              </div>
              
              <div className="flex-1 p-6 overflow-auto">
                {!singleResult && !singleRunning && (
                  <div className="h-full flex items-center justify-center text-gray-600 text-sm">
                    Run inference to see results and metrics.
                  </div>
                )}
                
                {singleRunning && (
                  <div className="h-full flex items-center justify-center text-indigo-400 text-sm gap-3">
                    <RefreshCw className="w-5 h-5 animate-spin" /> Processing...
                  </div>
                )}

                {singleResult && (
                  <div className="space-y-6">
                    {singleResult.error ? (
                      <div className="p-4 bg-red-500/10 border border-red-500/20 rounded-lg text-red-400 text-sm">
                        {singleResult.error}
                      </div>
                    ) : (
                      <>
                        {enforceJson && (
                          <div className={cn("p-4 border rounded-lg flex items-start gap-3 text-sm", singleResult.parsedData ? "bg-emerald-500/10 border-emerald-500/20 text-emerald-400" : "bg-red-500/10 border-red-500/20 text-red-400")}>
                            {singleResult.parsedData ? <CheckCircle2 className="w-5 h-5 shrink-0" /> : <XCircle className="w-5 h-5 shrink-0" />}
                            <div>
                              <p className="font-semibold mb-1">
                                {singleResult.parsedData ? 'Schema Validation Passed' : 'Schema Validation Failed'}
                                {singleResult.retries ? ` (after ${singleResult.retries} retry)` : ''}
                              </p>
                              {singleResult.parsedData && (
                                <pre className="mt-2 text-xs font-mono bg-black/30 p-3 rounded overflow-x-auto">
                                  {JSON.stringify(singleResult.parsedData, null, 2)}
                                </pre>
                              )}
                            </div>
                          </div>
                        )}
                        
                        <div>
                          <div className="text-xs font-medium text-gray-500 mb-2 uppercase tracking-wider">Raw Output</div>
                          <div className="text-sm font-mono text-gray-300 whitespace-pre-wrap bg-[#1A1A1A] p-4 rounded-lg border border-[#333]">
                            {singleResult.text}
                          </div>
                        </div>
                      </>
                    )}
                  </div>
                )}
              </div>
            </div>
          </div>
        )}

        {activeTab === 'batch' && (
          <div className="p-8 max-w-7xl mx-auto space-y-8">
            <div className="flex items-center justify-between">
              <h2 className="text-2xl font-semibold">Multi-Model Comparison Study</h2>
              <button 
                onClick={runBatch}
                disabled={batchRunning || selectedBatchModels.length === 0}
                className="bg-indigo-600 hover:bg-indigo-700 text-white px-6 py-2 rounded-lg text-sm font-medium flex items-center gap-2 disabled:opacity-50"
              >
                {batchRunning ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4" />}
                {batchRunning ? 'Running Benchmark...' : 'Start Benchmark'}
              </button>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-3 gap-8">
              <div className="lg:col-span-1 space-y-6">
                <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-xl p-6">
                  <h3 className="font-medium mb-4">Select Models to Compare</h3>
                  <div className="space-y-2 max-h-64 overflow-y-auto pr-2">
                    {models.map(m => (
                      <label key={m.name} className="flex items-center gap-3 p-3 rounded-lg border border-[#333] hover:bg-[#222] cursor-pointer transition-colors">
                        <input 
                          type="checkbox" 
                          checked={selectedBatchModels.includes(m.name)}
                          onChange={() => toggleBatchModel(m.name)}
                          className="accent-indigo-500 w-4 h-4"
                        />
                        <div className="flex-1 min-w-0">
                          <div className="text-sm font-medium truncate">{m.name}</div>
                          <div className="text-xs text-gray-500 font-mono">{m.details.parameter_size}</div>
                        </div>
                      </label>
                    ))}
                    {models.length === 0 && <div className="text-sm text-gray-500">No models available. Connect first.</div>}
                  </div>
                </div>

                <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-xl p-6">
                  <h3 className="font-medium mb-4">Test Prompts (One per line)</h3>
                  <textarea 
                    value={batchPrompts}
                    onChange={(e) => setBatchPrompts(e.target.value)}
                    rows={8}
                    className="w-full bg-[#0F0F0F] border border-[#333] rounded-lg px-4 py-3 text-sm focus:outline-none focus:border-indigo-500 resize-none font-mono whitespace-pre"
                  />
                  <p className="text-xs text-gray-500 mt-3">Temperature is locked to 0 for deterministic benchmarking.</p>
                </div>
              </div>

              <div className="lg:col-span-2 space-y-6">
                {aggregatedResults.length > 0 && !batchRunning && (
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                    <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-xl p-6 h-96">
                      <h3 className="text-sm font-medium text-gray-400 mb-6 text-center">Tokens Per Second (Higher is better)</h3>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart layout="vertical" data={aggregatedResults} margin={{ top: 0, right: 20, left: 100, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={false} />
                          <XAxis type="number" stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
                          <YAxis type="category" dataKey="displayName" stroke="#888" fontSize={11} width={120} tickLine={false} axisLine={false} />
                          <Tooltip cursor={{fill: '#222'}} contentStyle={{backgroundColor: '#111', borderColor: '#333', borderRadius: '8px'}} />
                          <Bar dataKey="avgTps" fill="#6366f1" radius={[0, 4, 4, 0]} barSize={24} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                    <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-xl p-6 h-96">
                      <h3 className="text-sm font-medium text-gray-400 mb-6 text-center">Time to First Token (Lower is better)</h3>
                      <ResponsiveContainer width="100%" height="100%">
                        <BarChart layout="vertical" data={aggregatedResults} margin={{ top: 0, right: 20, left: 100, bottom: 0 }}>
                          <CartesianGrid strokeDasharray="3 3" stroke="#333" horizontal={false} />
                          <XAxis type="number" stroke="#888" fontSize={12} tickLine={false} axisLine={false} />
                          <YAxis type="category" dataKey="displayName" stroke="#888" fontSize={11} width={120} tickLine={false} axisLine={false} />
                          <Tooltip cursor={{fill: '#222'}} contentStyle={{backgroundColor: '#111', borderColor: '#333', borderRadius: '8px'}} formatter={(value: number) => `${(value / 1000).toFixed(2)}s`} />
                          <Bar dataKey="avgTtft" fill="#10b981" radius={[0, 4, 4, 0]} barSize={24} />
                        </BarChart>
                      </ResponsiveContainer>
                    </div>
                  </div>
                )}

                {batchResults.length > 0 && (
                  <div className="bg-[#1A1A1A] border border-[#2A2A2A] rounded-xl overflow-hidden">
                    <div className="overflow-x-auto">
                      <table className="w-full text-sm text-left">
                        <thead className="text-xs text-gray-400 uppercase bg-[#222] border-b border-[#333]">
                          <tr>
                            <th className="px-6 py-4 font-medium">Model</th>
                            <th className="px-6 py-4 font-medium">Prompt #</th>
                            <th className="px-6 py-4 font-medium">TTFT (s)</th>
                            <th className="px-6 py-4 font-medium">TPS</th>
                            <th className="px-6 py-4 font-medium">Latency (s)</th>
                            <th className="px-6 py-4 font-medium">Status</th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-[#2A2A2A]">
                          {batchResults.map((r, i) => (
                            <tr key={i} className="hover:bg-[#222]/50 transition-colors">
                              <td className="px-6 py-4 font-medium" title={r.model}>{r.displayName}</td>
                              <td className="px-6 py-4 text-gray-400">#{r.promptIndex}</td>
                              <td className="px-6 py-4 font-mono">{(r.ttft / 1000).toFixed(2)}</td>
                              <td className="px-6 py-4 font-mono">{r.tps.toFixed(1)}</td>
                              <td className="px-6 py-4 font-mono">{(r.latency / 1000).toFixed(2)}</td>
                              <td className="px-6 py-4">
                                {r.success ? (
                                  <span className="inline-flex items-center gap-1.5 py-1 px-2 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-400">
                                    <CheckCircle2 className="w-3.5 h-3.5" /> Success
                                  </span>
                                ) : (
                                  <span className="inline-flex items-center gap-1.5 py-1 px-2 rounded-md text-xs font-medium bg-red-500/10 text-red-400">
                                    <XCircle className="w-3.5 h-3.5" /> Failed
                                  </span>
                                )}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  </div>
                )}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
