import { z } from 'zod';

export interface OpenRouterMetrics {
  ttft: number;
  totalLatency: number;
  tps: number;
  evalCount: number;
}

export interface OpenRouterResponse {
  text: string;
  metrics: OpenRouterMetrics;
  error?: string;
}

export async function generateOpenRouter(
  apiKey: string,
  model: string,
  prompt: string,
  options: { temperature?: number; format?: 'json' } = {}
): Promise<OpenRouterResponse> {
  const startTime = performance.now();
  let firstTokenTime = 0;

  try {
    const fetchOptions: any = {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${apiKey}`,
        'HTTP-Referer': 'http://localhost:3000',
        'X-Title': 'SLM Benchmark',
      },
      body: JSON.stringify({
        model: model,
        messages: [{ role: 'user', content: prompt }],
        stream: true,
        temperature: options.temperature ?? 0.7,
        response_format: options.format === 'json' ? { type: 'json_object' } : undefined,
      }),
    };

    const res = await fetch('https://openrouter.ai/api/v1/chat/completions', fetchOptions);

    if (!res.ok) {
      throw new Error(`OpenRouter API error: ${res.statusText}`);
    }

    const reader = res.body?.getReader();
    if (!reader) throw new Error('No response body');

    const decoder = new TextDecoder();
    let fullText = '';
    let buffer = '';
    let evalCount = 0;

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      if (firstTokenTime === 0) {
        firstTokenTime = performance.now();
      }

      buffer += decoder.decode(value, { stream: true });
      const lines = buffer.split('\n');
      buffer = lines.pop() || '';

      for (const line of lines) {
        const trimmed = line.trim();
        if (!trimmed || trimmed === 'data: [DONE]') continue;
        if (trimmed.startsWith('data: ')) {
          try {
            const data = JSON.parse(trimmed.slice(6));
            if (data.choices && data.choices[0].delta && data.choices[0].delta.content) {
              fullText += data.choices[0].delta.content;
              evalCount++; // Rough estimation of tokens since OpenRouter stream doesn't give precise count until end (sometimes)
            }
          } catch (e) {
            console.error('Failed to parse line:', line, e);
          }
        }
      }
    }

    const endTime = performance.now();
    const ttft = firstTokenTime > 0 ? firstTokenTime - startTime : 0;
    const totalLatency = endTime - startTime;
    const durationAfterFirstTokenSec = (totalLatency - ttft) / 1000;
    const tps = durationAfterFirstTokenSec > 0 ? evalCount / durationAfterFirstTokenSec : 0;

    return {
      text: fullText,
      metrics: {
        ttft,
        totalLatency,
        tps,
        evalCount,
      },
    };
  } catch (e: any) {
    return {
      text: '',
      metrics: { ttft: 0, totalLatency: 0, tps: 0, evalCount: 0 },
      error: e.message || 'Unknown error occurred',
    };
  }
}

// Retry-once mechanism for invalid JSON
export async function generateWithRetryOpenRouter(
  apiKey: string,
  model: string,
  prompt: string,
  schema: z.ZodTypeAny,
  options: { temperature?: number } = {}
): Promise<{ data: any; raw: string; metrics: OpenRouterMetrics; retries: number; error?: string }> {
  
  let retries = 0;
  let response = await generateOpenRouter(apiKey, model, prompt, { ...options, format: 'json' });
  
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
    
    const retryResponse = await generateOpenRouter(apiKey, model, retryPrompt, { ...options, format: 'json' });
    
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
