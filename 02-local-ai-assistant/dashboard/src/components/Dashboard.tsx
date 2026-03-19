import React, { useState, useEffect, useMemo } from 'react';
import { 
  Activity, Database, Play, RefreshCw,
  Zap, Clock, Target, Server, Cpu, ShieldCheck
} from 'lucide-react';
import { z } from 'zod';
import { getModels, generate, generateWithRetry, OllamaModel, OllamaMetrics } from '../lib/ollama';
import { generateOpenRouter, generateWithRetryOpenRouter } from '../lib/openrouter';
import { cn } from '../lib/utils';
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Legend
} from 'recharts';

// Pre-defined JSON Schema for testing
const SentimentSchema = z.object({
  summary: z.string(),
  sentiment: z.enum(['positive', 'neutral', 'negative']),
  confidence: z.number().min(0).max(1)
});

// Helper for standard deviation
function getStandardDeviation(array: number[]) {
  if (array.length <= 1) return 0;
  const n = array.length;
  const mean = array.reduce((a, b) => a + b) / n;
  return Math.sqrt(array.map(x => Math.pow(x - mean, 2)).reduce((a, b) => a + b) / (n - 1));
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState<'setup' | 'single' | 'batch'>('batch');
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
  const [singleResult, setSingleResult] = useState<any>(null);

  // Batch Run State
  const [batchPrompts, setBatchPrompts] = useState(
    'EXTRACT: "Anthropic raised $4B from Amazon." Return JSON: {company: string, amount: string}\nCLASSIFY SENTIMENT: "The battery life is okay, but the screen cracked after a week." Reply with only the word POSITIVE or NEGATIVE.\nFORMAT: "milk eggs bread" as an HTML unordered list. No markdown blocks.'
  );
  const [selectedBatchModels, setSelectedBatchModels] = useState<string[]>([]);
  const [batchRunning, setBatchRunning] = useState(false);
  const [batchResults, setBatchResults] = useState<any[]>([]);

  useEffect(() => {
    fetchModels();
  }, []);

  const fetchModels = async () => {
    setLoadingModels(true);
    setConnectionError('');
    try {
      const m = await getModels(baseUrl);
      
      // Inject OpenRouter baseline models
      if (openRouterKey) {
        m.push({ name: 'openrouter/google/gemini-2.5-flash', size: 0, digest: '', details: { parameter_size: 'Cloud', quantization_level: 'FP16' } });
        m.push({ name: 'openrouter/meta-llama/llama-3.1-8b-instruct', size: 0, digest: '', details: { parameter_size: 'Cloud', quantization_level: 'FP16' } });
      }
      setModels(m);
      if (m.length > 0 && !selectedModel) setSelectedModel(m[0].name);
    } catch (e: any) {
      setConnectionError(e.message || 'Failed to connect to Ollama.');
    } finally {
      setLoadingModels(false);
    }
  };

  const runSingle = async () => { /* Kept simple for brevity, main focus is batch */ };

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
      const displayName = isOpenRouter 
        ? model.split('/').slice(-1)[0].replace('-instruct', '') + ' (Cloud)'
        : model.replace(':latest', '');

      for (let i = 0; i < prompts.length; i++) {
        const p = prompts[i];
        try {
          const res = isOpenRouter
            ? await generateOpenRouter(openRouterKey, actualModel, p, { temperature: 0 })
            : await generate(baseUrl, actualModel, p, { temperature: 0 });
            
          results.push({
            model, displayName, isCloud: isOpenRouter, promptIndex: i + 1,
            ttft: res.metrics.ttft, tps: res.metrics.tps, latency: res.metrics.totalLatency, success: !res.error
          });
          setBatchResults([...results]);
        } catch (e) {
          results.push({
            model, displayName, isCloud: isOpenRouter, promptIndex: i + 1,
            ttft: 0, tps: 0, latency: 0, success: false
          });
          setBatchResults([...results]);
        }
      }
    }
    setBatchRunning(false);
  };

  const toggleBatchModel = (model: string) => {
    setSelectedBatchModels(prev => prev.includes(model) ? prev.filter(m => m !== model) : [...prev, model]);
  };

  // ---------------------------------------------------------------------------
  // DATA PROCESSING & STORYTELLING LOGIC
  // ---------------------------------------------------------------------------
  const { aggregatedData, podium, insights, tableStats } = useMemo(() => {
    if (batchResults.length === 0) return { aggregatedData: [], podium: null, insights: [], tableStats: null };

    const aggs = selectedBatchModels.map(model => {
      const runs = batchResults.filter(r => r.model === model && r.success);
      if (runs.length === 0) return null;

      const coldRun = runs.find(r => r.promptIndex === 1);
      const warmRuns = runs.filter(r => r.promptIndex > 1);
      
      const coldTtft = coldRun ? coldRun.ttft : 0;
      const coldTps = coldRun ? coldRun.tps : 0;
      
      const warmTtft = warmRuns.length ? warmRuns.reduce((sum, r) => sum + r.ttft, 0) / warmRuns.length : coldTtft;
      const warmTps = warmRuns.length ? warmRuns.reduce((sum, r) => sum + r.tps, 0) / warmRuns.length : coldTps;
      
      const allTps = runs.map(r => r.tps);
      const stdDevTps = getStandardDeviation(allTps);

      return {
        model,
        displayName: runs[0].displayName,
        isCloud: runs[0].isCloud,
        coldTtft, warmTtft,
        coldTps, warmTps,
        stdDevTps,
        avgLatency: runs.reduce((s, r) => s + r.latency, 0) / runs.length
      };
    }).filter(Boolean) as any[];

    // Table Stats (Min/Max for Highlighting)
    const validTtft = batchResults.filter(r => r.success).map(r => r.ttft);
    const validTps = batchResults.filter(r => r.success).map(r => r.tps);
    const validLatency = batchResults.filter(r => r.success).map(r => r.latency);
    
    const tableStats = {
      minTtft: Math.min(...validTtft), maxTtft: Math.max(...validTtft),
      minTps: Math.min(...validTps), maxTps: Math.max(...validTps),
      minLatency: Math.min(...validLatency), maxLatency: Math.max(...validLatency),
    };

    if (aggs.length === 0) return { aggregatedData: [], podium: null, insights: [], tableStats };

    // Podium Winners
    const fastestTps = [...aggs].sort((a, b) => b.warmTps - a.warmTps)[0];
    const lowestTtft = [...aggs].sort((a, b) => a.warmTtft - b.warmTtft)[0];
    const mostConsistent = [...aggs].sort((a, b) => a.stdDevTps - b.stdDevTps)[0];

    // Insights Generation
    const newInsights = [];
    
    // Insight 1: Memory Pressure Alert
    const memoryHog = aggs.find(a => !a.isCloud && a.coldTtft > a.warmTtft * 10);
    if (memoryHog) {
      newInsights.push({
        type: 'danger', icon: <Server className="w-5 h-5" />, title: 'Memory Pressure / SSD Swap Detected',
        desc: memoryHog.displayName + ' had a cold TTFT of ' + (memoryHog.coldTtft/1000).toFixed(1) + 's, which is >10x its warm speed. This indicates the model weights exceeded unified memory, forcing macOS to swap to the SSD on first load.'
      });
    }

    // Insight 2: Cloud vs Local Peak
    const bestLocal = aggs.filter(a => !a.isCloud).sort((a, b) => b.warmTps - a.warmTps)[0];
    const bestCloud = aggs.filter(a => a.isCloud).sort((a, b) => b.warmTps - a.warmTps)[0];
    
    if (bestLocal && bestCloud) {
      if (bestLocal.warmTps > bestCloud.warmTps) {
        newInsights.push({
          type: 'success', icon: <Zap className="w-5 h-5" />, title: 'Local Outperforms Cloud Streaming',
          desc: bestLocal.displayName + ' running locally outpaced ' + bestCloud.displayName + ' over the network (' + bestLocal.warmTps.toFixed(1) + ' TPS vs ' + bestCloud.warmTps.toFixed(1) + ' TPS). Zero network overhead pays off.'
        });
      } else {
        newInsights.push({
          type: 'info', icon: <Activity className="w-5 h-5" />, title: 'Cloud Dominates Throughput',
          desc: bestCloud.displayName + ' (' + bestCloud.warmTps.toFixed(1) + ' TPS) beat the fastest local model, ' + bestLocal.displayName + ' (' + bestLocal.warmTps.toFixed(1) + ' TPS). The cloud cluster\'s dedicated GPU infrastructure wins at raw throughput.'
        });
      }
    }

    // Insight 3: Consistency
    newInsights.push({
      type: 'warning', icon: <Target className="w-5 h-5" />, title: 'Most Consistent Predictability',
      desc: mostConsistent.displayName + ' had the lowest variance across all runs (\u00b1' + mostConsistent.stdDevTps.toFixed(1) + ' TPS). Highly valuable for deterministic infrastructure where predictable SLAs matter.'
    });

    return { aggregatedData: aggs, podium: { fastestTps, lowestTtft, mostConsistent }, insights: newInsights, tableStats };
  }, [batchResults, selectedBatchModels]);

  // ---------------------------------------------------------------------------
  // RENDER HELPERS
  // ---------------------------------------------------------------------------
  const cardOuterClass = "bg-[#111118] border border-[#1e1e2e] shadow-lg shadow-black/20 rounded-xl transition-all hover:shadow-[#3b82f6]/5";

  return (
    <div className="flex h-screen bg-[#0a0a0f] text-gray-200 font-sans selection:bg-[#3b82f6]/30">
      
      {/* ---------------- SIDEBAR ---------------- */}
      <div className="w-72 border-r border-[#1e1e2e] bg-[#0a0a0f] flex flex-col z-10 shrink-0">
        <div className="p-6 border-b border-[#1e1e2e]">
          <h1 className="text-xl font-bold tracking-tight flex items-center gap-2 text-white">
            <Cpu className="w-5 h-5 text-[#3b82f6]" /> SLM Bench
          </h1>
          <p className="text-xs text-gray-500 mt-1 font-mono">Performance Analytics</p>
        </div>
        
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {/* Connection Block */}
          <div className="space-y-4">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Environment</h3>
            <div>
              <input 
                type="text" value={baseUrl} onChange={(e) => setBaseUrl(e.target.value)}
                placeholder="Ollama URL" className="w-full bg-[#111118] border border-[#1e1e2e] rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-[#3b82f6] font-mono"
              />
            </div>
            <div>
              <input 
                type="password" value={openRouterKey} onChange={(e) => setOpenRouterKey(e.target.value)}
                placeholder="OpenRouter Key (sk-or...)" className="w-full bg-[#111118] border border-[#1e1e2e] rounded-lg px-3 py-2 text-xs focus:outline-none focus:border-[#3b82f6] font-mono mb-2"
              />
              <button 
                onClick={fetchModels} disabled={loadingModels}
                className="w-full bg-[#1e1e2e] hover:bg-[#2a2a3a] text-gray-300 px-3 py-2 rounded-lg text-xs font-medium flex items-center justify-center gap-2 transition-colors"
              >
                <RefreshCw className={cn("w-3.5 h-3.5", loadingModels && "animate-spin")} /> Refresh Models
              </button>
            </div>
            {connectionError && <div className="text-[10px] text-[#ef4444] bg-[#ef4444]/10 p-2 rounded">{connectionError}</div>}
          </div>

          <hr className="border-[#1e1e2e]" />

          {/* Model Selection */}
          <div className="space-y-3">
             <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider flex justify-between">
               Compare ({selectedBatchModels.length})
             </h3>
             <div className="space-y-1.5 max-h-48 overflow-y-auto pr-1">
               {models.map(m => (
                 <label key={m.name} className={cn("flex items-start gap-3 p-2.5 rounded-lg border cursor-pointer transition-colors text-xs", selectedBatchModels.includes(m.name) ? "bg-[#3b82f6]/10 border-[#3b82f6]/30" : "bg-[#111118] border-[#1e1e2e] hover:border-[#333]")}>
                   <input type="checkbox" checked={selectedBatchModels.includes(m.name)} onChange={() => toggleBatchModel(m.name)} className="mt-0.5 accent-[#3b82f6] w-3.5 h-3.5" />
                   <div className="flex-1 min-w-0">
                     <div className="font-medium text-gray-300 truncate">{m.name.replace('openrouter/', '').replace(':latest', '')}</div>
                     <div className="text-[10px] text-gray-500 font-mono mt-0.5">{m.details.parameter_size}</div>
                   </div>
                 </label>
               ))}
               {models.length === 0 && <div className="text-xs text-gray-600 italic">No models found. Check connection.</div>}
             </div>
          </div>

          <hr className="border-[#1e1e2e]" />
          
          {/* Prompts */}
          <div className="space-y-3">
            <h3 className="text-xs font-semibold text-gray-500 uppercase tracking-wider">Test Suite</h3>
            <textarea 
              value={batchPrompts} onChange={(e) => setBatchPrompts(e.target.value)} rows={5}
              className="w-full bg-[#111118] border border-[#1e1e2e] rounded-lg px-3 py-2 text-[11px] focus:outline-none focus:border-[#3b82f6] resize-none font-mono whitespace-pre leading-relaxed text-gray-400"
            />
          </div>

        </div>

        {/* Global Action */}
        <div className="p-4 border-t border-[#1e1e2e] bg-[#0a0a0f]">
          <button 
            onClick={runBatch} disabled={batchRunning || selectedBatchModels.length === 0}
            className="w-full bg-[#3b82f6] hover:bg-blue-500 text-white shadow-lg shadow-[#3b82f6]/20 px-4 py-3 rounded-xl text-sm font-semibold flex items-center justify-center gap-2 disabled:opacity-50 disabled:shadow-none transition-all"
          >
            {batchRunning ? <RefreshCw className="w-4 h-4 animate-spin" /> : <Play className="w-4 h-4 ml-1" />}
            {batchRunning ? 'Executing Test Suite...' : 'Start Benchmark'}
          </button>
        </div>
      </div>

      {/* ---------------- MAIN CONTENT ---------------- */}
      <div className="flex-1 overflow-auto p-8 relative">
        <div className="max-w-7xl mx-auto space-y-8">
          
          <header className="mb-8">
            <h2 className="text-3xl font-bold tracking-tight text-white">Analysis & Benchmarks</h2>
            <p className="text-gray-400 mt-2">Empirical evaluation of local SLMs vs cloud baseline controls.</p>
          </header>

          {batchResults.length === 0 && !batchRunning && (
            <div className="h-64 border border-dashed border-[#1e1e2e] rounded-2xl flex flex-col items-center justify-center text-gray-600">
              <Database className="w-12 h-12 mb-4 opacity-20" />
              <p>Configure parameters on the left and run benchmark to view analytics.</p>
            </div>
          )}

          {aggregatedData.length > 0 && !batchRunning && podium && (
            <>
              {/* --- HERO PODIUM --- */}
              <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
                <div className={cn(cardOuterClass, "p-5 relative overflow-hidden group")}>
                  <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity"><Zap className="w-16 h-16 text-[#3b82f6]" /></div>
                  <div className="text-xs font-semibold tracking-wider text-[#3b82f6] mb-1 uppercase">Highest Throughput</div>
                  <div className="text-2xl font-bold text-white mb-4 truncate pr-12">{podium.fastestTps.displayName}</div>
                  <div className="flex items-end gap-2">
                    <span className="text-4xl font-black text-white">{podium.fastestTps.warmTps.toFixed(1)}</span>
                    <span className="text-sm text-gray-500 mb-1 font-mono">TPS (Avg)</span>
                  </div>
                </div>

                <div className={cn(cardOuterClass, "p-5 relative overflow-hidden group")}>
                  <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity"><Clock className="w-16 h-16 text-[#22c55e]" /></div>
                  <div className="text-xs font-semibold tracking-wider text-[#22c55e] mb-1 uppercase">Lowest Latency</div>
                  <div className="text-2xl font-bold text-white mb-4 truncate pr-12">{podium.lowestTtft.displayName}</div>
                  <div className="flex items-end gap-2">
                    <span className="text-4xl font-black text-white">{(podium.lowestTtft.warmTtft/1000).toFixed(2)}</span>
                    <span className="text-sm text-gray-500 mb-1 font-mono">Sec (TTFT)</span>
                  </div>
                </div>

                <div className={cn(cardOuterClass, "p-5 relative overflow-hidden group")}>
                  <div className="absolute top-0 right-0 p-4 opacity-10 group-hover:opacity-20 transition-opacity"><ShieldCheck className="w-16 h-16 text-[#f59e0b]" /></div>
                  <div className="text-xs font-semibold tracking-wider text-[#f59e0b] mb-1 uppercase">Most Predictable</div>
                  <div className="text-2xl font-bold text-white mb-4 truncate pr-12">{podium.mostConsistent.displayName}</div>
                  <div className="flex items-end gap-2">
                    <span className="text-4xl font-black text-white">±{podium.mostConsistent.stdDevTps.toFixed(1)}</span>
                    <span className="text-sm text-gray-500 mb-1 font-mono">TPS Var</span>
                  </div>
                </div>
              </div>

              {/* --- GROUPED CHARTS --- */}
              <div className="grid grid-cols-1 xl:grid-cols-2 gap-6">
                <div className={cn(cardOuterClass, "p-6 h-[400px] flex flex-col")}>
                   <div className="mb-6">
                     <h3 className="font-semibold text-white flex items-center gap-2"><Clock className="w-4 h-4 text-gray-400" /> Time to First Token (TTFT)</h3>
                     <p className="text-xs text-gray-500 mt-1">Logarithmic Scale • Lower is better</p>
                   </div>
                   <ResponsiveContainer width="100%" height="100%">
                     <BarChart data={aggregatedData} layout="vertical" margin={{ top: 0, right: 30, left: 60, bottom: 0 }}>
                       <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" horizontal={false} />
                       <XAxis type="number" scale="log" domain={['auto', 'auto']} stroke="#666" fontSize={11} tickLine={false} axisLine={false} tickFormatter={(v) => (v/1000).toFixed(1) + 's'} />
                       <YAxis type="category" dataKey="displayName" stroke="#888" fontSize={11} width={120} tickLine={false} axisLine={false} />
                       <Tooltip cursor={{fill: '#1e1e2e'}} contentStyle={{backgroundColor: '#111118', borderColor: '#2e2e3e', borderRadius: '8px'}} formatter={(value: any) => [(value / 1000).toFixed(2) + 's']} />
                       <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                       <Bar dataKey="coldTtft" name="Cold Start (Run 1)" fill="#ef4444" radius={[0, 4, 4, 0]} barSize={12} opacity={0.6} >
                       </Bar>
                       <Bar dataKey="warmTtft" name="Warm Average" fill="#22c55e" radius={[0, 4, 4, 0]} barSize={12} />
                     </BarChart>
                   </ResponsiveContainer>
                </div>

                <div className={cn(cardOuterClass, "p-6 h-[400px] flex flex-col")}>
                   <div className="mb-6">
                     <h3 className="font-semibold text-white flex items-center gap-2"><Zap className="w-4 h-4 text-gray-400" /> Tokens Per Second (TPS)</h3>
                     <p className="text-xs text-gray-500 mt-1">Linear Scale • Higher is better</p>
                   </div>
                   <ResponsiveContainer width="100%" height="100%">
                     <BarChart data={aggregatedData} layout="vertical" margin={{ top: 0, right: 30, left: 60, bottom: 0 }}>
                       <CartesianGrid strokeDasharray="3 3" stroke="#1e1e2e" horizontal={false} />
                       <XAxis type="number" stroke="#666" fontSize={11} tickLine={false} axisLine={false} />
                       <YAxis type="category" dataKey="displayName" stroke="#888" fontSize={11} width={120} tickLine={false} axisLine={false} />
                       <Tooltip cursor={{fill: '#1e1e2e'}} contentStyle={{backgroundColor: '#111118', borderColor: '#2e2e3e', borderRadius: '8px'}} formatter={(value: any) => [value.toFixed(1) + ' t/s']} />
                       <Legend wrapperStyle={{ fontSize: '12px', paddingTop: '10px' }} />
                       <Bar dataKey="coldTps" name="Cold Start (Run 1)" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={12} opacity={0.6} />
                       <Bar dataKey="warmTps" name="Warm Average" fill="#3b82f6" radius={[0, 4, 4, 0]} barSize={12} />
                     </BarChart>
                   </ResponsiveContainer>
                </div>
              </div>

              {/* --- GRANULAR RESULTS TABLE --- */}
              {tableStats && (
                <div className={cn(cardOuterClass, "overflow-hidden")}>
                  <div className="p-5 border-b border-[#1e1e2e] bg-[#0d0d14]">
                    <h3 className="font-semibold text-white">Execution Logs</h3>
                  </div>
                  <div className="overflow-x-auto">
                    <table className="w-full text-sm text-left">
                      <thead className="text-[10px] text-gray-500 uppercase tracking-wider bg-[#111118] border-b border-[#1e1e2e]">
                        <tr>
                          <th className="px-6 py-4 font-semibold">Model</th>
                          <th className="px-6 py-4 font-semibold">Run Type</th>
                          <th className="px-6 py-4 font-semibold">Prompt #</th>
                          <th className="px-6 py-4 font-semibold text-right">TTFT (s)</th>
                          <th className="px-6 py-4 font-semibold text-right">TPS</th>
                          <th className="px-6 py-4 font-semibold text-right">Latency (s)</th>
                          <th className="px-6 py-4 font-semibold text-center">Consistency</th>
                        </tr>
                      </thead>
                      <tbody className="divide-y divide-[#1e1e2e]">
                        {batchResults.map((r, i) => {
                          const isCold = r.promptIndex === 1;
                          const agg = aggregatedData.find(a => a.model === r.model);
                          const isBestTtft = r.ttft === tableStats.minTtft;
                          const isWorstTtft = r.ttft === tableStats.maxTtft;
                          const isBestTps = r.tps === tableStats.maxTps;
                          const isWorstTps = r.tps === tableStats.minTps;

                          return (
                            <tr key={i} className="hover:bg-[#1e1e2e]/30 transition-colors group">
                              <td className="px-6 py-4 font-medium text-gray-200">{r.displayName}</td>
                              <td className="px-6 py-3">
                                <span className={cn("inline-flex items-center px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border", 
                                  isCold ? "bg-[#f59e0b]/10 text-[#f59e0b] border-[#f59e0b]/20" : "bg-[#22c55e]/10 text-[#22c55e] border-[#22c55e]/20"
                                )}>
                                  {isCold ? 'Cold' : 'Warm'}
                                </span>
                              </td>
                              <td className="px-6 py-4 text-gray-500 font-mono text-xs">#{r.promptIndex}</td>
                              
                              <td className={cn("px-6 py-4 font-mono text-right text-xs", 
                                isBestTtft ? "text-[#22c55e] font-bold" : isWorstTtft ? "text-[#ef4444]" : "text-gray-400"
                              )}>
                                {(r.ttft / 1000).toFixed(2)}
                              </td>
                              
                              <td className={cn("px-6 py-4 font-mono text-right text-xs", 
                                isBestTps ? "text-[#3b82f6] font-bold" : isWorstTps ? "text-[#ef4444]" : "text-gray-400"
                              )}>
                                {r.tps.toFixed(1)}
                              </td>
                              
                              <td className="px-6 py-4 font-mono text-right text-gray-400 text-xs">{(r.latency / 1000).toFixed(2)}</td>
                              
                              <td className="px-6 py-4 text-center">
                                {r.promptIndex === 1 && agg ? (
                                  <span className={cn("text-xs font-mono px-2 py-1 rounded bg-black/40", 
                                    agg.stdDevTps < 2 ? "text-[#22c55e]" : agg.stdDevTps > 10 ? "text-[#ef4444]" : "text-gray-400"
                                  )}>
                                    ±{agg.stdDevTps.toFixed(1)}
                                  </span>
                                ) : <span className="text-gray-600">-</span>}
                              </td>
                            </tr>
                          );
                        })}
                      </tbody>
                    </table>
                  </div>
                </div>
              )}

              {/* --- AUTO INSIGHTS --- */}
              <div>
                <h3 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
                  <Activity className="w-5 h-5 text-[#3b82f6]" /> Key Engineering Insights
                </h3>
                <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
                  {insights.map((insight, idx) => {
                    const colorClasses = {
                      danger: "border-[#ef4444]/30 bg-[#ef4444]/5 text-[#ef4444]",
                      success: "border-[#22c55e]/30 bg-[#22c55e]/5 text-[#22c55e]",
                      warning: "border-[#f59e0b]/30 bg-[#f59e0b]/5 text-[#f59e0b]",
                      info: "border-[#3b82f6]/30 bg-[#3b82f6]/5 text-[#3b82f6]"
                    }[insight.type];

                    return (
                      <div key={idx} className={cn(cardOuterClass, "p-5 border", colorClasses)}>
                        <h4 className="font-semibold text-sm flex items-center gap-2 mb-3">
                          {insight.icon} {insight.title}
                        </h4>
                        <p className="text-sm leading-relaxed text-gray-300">
                          {insight.desc}
                        </p>
                      </div>
                    );
                  })}
                </div>
              </div>

            </>
          )}

        </div>
      </div>
    </div>
  );
}
