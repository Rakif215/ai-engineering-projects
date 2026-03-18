import { z } from 'zod';

export interface OllamaModel {
  name: string;
  size: number;
  digest: string;
  details: {
    parameter_size: string;
    quantization_level: string;
  };
}

export interface OllamaMetrics {
  ttft: number; // Time to first token (ms)
  totalLatency: number; // Total time (ms)
  tps: number; // Tokens per second
  evalCount: number; // Number of tokens generated
}

export interface GenerateResponse {
  text: string;
  metrics: OllamaMetrics;
  error?: string;
}

export async function getModels(baseUrl: string): Promise<OllamaModel[]> {
  try {
    const res = await fetch(`${baseUrl}/api/tags`);
    if (!res.ok) throw new Error('Failed to fetch models');
    const data = await res.json();
    return data.models || [];
  } catch (e) {
    console.error('Error fetching models:', e);
    throw e;
  }
}

export async function generate(
  baseUrl: string,
  model: string,
  prompt: string,
  options: { temperature?: number; format?: 'json' } = {}
): Promise<GenerateResponse> {
  const startTime = performance.now();
  let firstTokenTime = 0;

  try {
    const res = await fetch(`${baseUrl}/api/generate`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        model,
        prompt,
        stream: true,
        options: {
          temperature: options.temperature ?? 0.7,
        },
        format: options.format,
      }),
    });

    if (!res.ok) {
      throw new Error(`Ollama API error: ${res.statusText}`);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let fullText = '';
    let metricsData: any = {};
    let buffer = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      if (firstTokenTime === 0) {
        firstTokenTime = performance.now();
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      
      // Keep the last partial line in the buffer
      buffer = lines.pop() || '';
      
      for (const line of lines) {
        if (!line.trim()) continue;
        try {
          const data = JSON.parse(line);
          if (data.response) fullText += data.response;
          if (data.done) {
            metricsData = data;
          }
        } catch (e) {
          console.error('Failed to parse line:', line, e);
        }
      }
    }

    const endTime = performance.now();
    const ttft = firstTokenTime > 0 ? firstTokenTime - startTime : 0;
    const totalLatency = endTime - startTime;
    
    // eval_duration is in nanoseconds
    const evalDurationSec = (metricsData.eval_duration || 0) / 1e9;
    const evalCount = metricsData.eval_count || 0;
    const tps = evalDurationSec > 0 ? evalCount / evalDurationSec : 0;

    return {
      text: fullText,
      metrics: {
        ttft,
        totalLatency,
        tps,
        evalCount,
      }
    };
  } catch (e: any) {
    return {
      text: '',
      metrics: { ttft: 0, totalLatency: 0, tps: 0, evalCount: 0 },
      error: e.message || 'Unknown error occurred',
    };
  }
}

// Phase 2: Structure & Determinism
// Retry-once mechanism for invalid JSON
export async function generateWithRetry(
  baseUrl: string,
  model: string,
  prompt: string,
  schema: z.ZodTypeAny,
  options: { temperature?: number } = {}
): Promise<{ data: any; raw: string; metrics: OllamaMetrics; retries: number; error?: string }> {
  
  let retries = 0;
  let response = await generate(baseUrl, model, prompt, { ...options, format: 'json' });
  
  if (response.error) {
    return { data: null, raw: response.text, metrics: response.metrics, retries, error: response.error };
  }

  try {
    const parsed = JSON.parse(response.text);
    const validated = schema.parse(parsed);
    return { data: validated, raw: response.text, metrics: response.metrics, retries };
  } catch (e: any) {
    // Retry once
    retries++;
    const retryPrompt = `${prompt}\n\nREMINDER: You MUST output valid JSON matching the schema. Previous attempt failed with error: ${e.message}. Do not include markdown formatting, only raw JSON.`;
    
    const retryResponse = await generate(baseUrl, model, retryPrompt, { ...options, format: 'json' });
    
    if (retryResponse.error) {
      return { data: null, raw: retryResponse.text, metrics: retryResponse.metrics, retries, error: retryResponse.error };
    }

    try {
      const parsed = JSON.parse(retryResponse.text);
      const validated = schema.parse(parsed);
      return { data: validated, raw: retryResponse.text, metrics: retryResponse.metrics, retries };
    } catch (retryError: any) {
      return { 
        data: null, 
        raw: retryResponse.text, 
        metrics: retryResponse.metrics, 
        retries, 
        error: `Validation failed after retry: ${retryError.message}` 
      };
    }
  }
}
