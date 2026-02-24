/**
 * llm-reporter.ts
 *
 * Sends parameter analysis to GLM 5 API for intelligent recommendations.
 * Uses the same LLM that's running the trading system.
 *
 * Rate limited to max 1 call per hour to avoid costs/latency.
 * NEVER auto-applies changes - logs suggestions only.
 */

import { AnalysisResult, ParameterSuggestion } from './parameter-analyzer.js';

// ─────────────────────────────────────────────────────────────────
// INTERFACES
// ─────────────────────────────────────────────────────────────────

export interface LLMConfig {
  apiUrl: string;              // GLM 5 API endpoint
  apiKey: string;              // API key (from env)
  model: string;               // Model name
  maxTokens: number;           // Max response tokens
  temperature: number;         // Creativity (0 = deterministic)
  maxHourlyCalls: number;      // Rate limit
}

export interface LLMRecommendation {
  category: string;
  currentValue: string;
  recommendedValue: string;
  reasoning: string;
  confidence: 'HIGH' | 'MEDIUM' | 'LOW';
  codeChange?: string;         // Exact CONFIG change if applicable
}

export interface LLMReport {
  timestamp: string;
  analysisSummary: string;
  recommendations: LLMRecommendation[];
  rawResponse?: string;        // For debugging
  error?: string;
}

// ─────────────────────────────────────────────────────────────────
// DEFAULT CONFIG
// ─────────────────────────────────────────────────────────────────

export const DEFAULT_LLM_CONFIG: LLMConfig = {
  apiUrl: process.env.LLM_API_URL || 'https://open.bigmodel.cn/api/paas/v4/chat/completions',
  apiKey: process.env.LLM_API_KEY || process.env.GLM_API_KEY || '',
  model: 'glm-4-flash',  // Fast, cheap model for parameter suggestions
  maxTokens: 1000,
  temperature: 0.3,       // Low temp for consistent recommendations
  maxHourlyCalls: 1,      // Max 1 call per hour
};

// ─────────────────────────────────────────────────────────────────
// LLM REPORTER CLASS
// ─────────────────────────────────────────────────────────────────

export class LLMReporter {
  private config: LLMConfig;
  private lastCallTime: number = 0;
  private callCount: number = 0;
  private enabled: boolean;

  constructor(config: Partial<LLMConfig> = {}) {
    this.config = { ...DEFAULT_LLM_CONFIG, ...config };
    this.enabled = !!this.config.apiKey;
  }

  /**
   * Check if we should make an LLM call (rate limited)
   */
  shouldCall(): boolean {
    if (!this.enabled) return false;

    const now = Date.now();
    const hourMs = 60 * 60 * 1000;
    const timeSinceLastCall = now - this.lastCallTime;

    // Only allow if at least 1 hour has passed
    return timeSinceLastCall >= (hourMs / this.config.maxHourlyCalls);
  }

  /**
   * Get time until next allowed call
   */
  getTimeUntilNextCall(): number {
    if (!this.enabled) return Infinity;
    const hourMs = 60 * 60 * 1000;
    const minInterval = hourMs / this.config.maxHourlyCalls;
    const elapsed = Date.now() - this.lastCallTime;
    return Math.max(0, minInterval - elapsed);
  }

  /**
   * Check if LLM is enabled and configured
   */
  isEnabled(): boolean {
    return this.enabled;
  }

  /**
   * Send analysis to LLM and get recommendations
   */
  async getRecommendations(analysis: AnalysisResult): Promise<LLMReport> {
    const timestamp = new Date().toISOString();

    if (!this.enabled) {
      return {
        timestamp,
        analysisSummary: analysis.summary,
        recommendations: [],
        error: 'LLM API key not configured (set LLM_API_KEY or GLM_API_KEY env var)',
      };
    }

    if (!this.shouldCall()) {
      const waitMinutes = Math.ceil(this.getTimeUntilNextCall() / 60000);
      return {
        timestamp,
        analysisSummary: analysis.summary,
        recommendations: [],
        error: `Rate limited - next call in ${waitMinutes} minutes`,
      };
    }

    try {
      const prompt = this.buildPrompt(analysis);
      const response = await this.callLLM(prompt);
      const recommendations = this.parseResponse(response);

      this.lastCallTime = Date.now();
      this.callCount++;

      return {
        timestamp,
        analysisSummary: analysis.summary,
        recommendations,
        rawResponse: response,
      };
    } catch (error: any) {
      return {
        timestamp,
        analysisSummary: analysis.summary,
        recommendations: [],
        error: error.message || 'Unknown error',
      };
    }
  }

  // ───────────────────────────────────────────────────────────────
  // PRIVATE METHODS
  // ───────────────────────────────────────────────────────────────

  private buildPrompt(analysis: AnalysisResult): string {
    const suggestionsText = analysis.suggestions
      .slice(0, 5)
      .map((s, i) => `${i + 1}. [${s.category}] ${s.current} → ${s.suggested}
   Reason: ${s.reason}
   Impact: ${s.impact}, Sample: ${s.sampleSize} trades, Win rate: ${(s.winRate * 100).toFixed(0)}%`)
      .join('\n\n');

    return `You are a quantitative trading parameter optimizer. Analyze the following trading performance data and suggest parameter changes.

## Current Performance
- Total trades: ${analysis.totalTrades}
- Overall win rate: ${(analysis.overallWinRate * 100).toFixed(1)}%
- Summary: ${analysis.summary}

## Detected Underperforming Buckets
${suggestionsText}

## Task
For each underperforming bucket, provide a specific parameter change recommendation in this JSON format:
{
  "recommendations": [
    {
      "category": "BB_THRESHOLD",
      "currentValue": "15%",
      "recommendedValue": "10%",
      "reasoning": "More extreme entries have better edge",
      "confidence": "HIGH",
      "codeChange": "CONFIG.regime.bbExtremeLong: 0.15 → 0.10"
    }
  ]
}

Guidelines:
1. Be conservative - only suggest changes with clear statistical evidence (sample >= 10 trades)
2. Consider interactions between parameters
3. Prefer tightening criteria over loosening
4. If unsure, set confidence to LOW
5. Provide exact CONFIG field changes when possible

Respond with ONLY the JSON, no other text.`;
  }

  private async callLLM(prompt: string): Promise<string> {
    const response = await fetch(this.config.apiUrl, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        'Authorization': `Bearer ${this.config.apiKey}`,
      },
      body: JSON.stringify({
        model: this.config.model,
        messages: [
          { role: 'system', content: 'You are a quantitative trading parameter optimizer. Respond only with valid JSON.' },
          { role: 'user', content: prompt },
        ],
        max_tokens: this.config.maxTokens,
        temperature: this.config.temperature,
      }),
    });

    if (!response.ok) {
      const errorText = await response.text().catch(() => 'Unknown error');
      throw new Error(`LLM API error ${response.status}: ${errorText.slice(0, 200)}`);
    }

    const data = await response.json() as any;
    return data.choices?.[0]?.message?.content || '';
  }

  private parseResponse(response: string): LLMRecommendation[] {
    const recommendations: LLMRecommendation[] = [];

    try {
      // Try to extract JSON from response
      let jsonStr = response.trim();

      // Remove markdown code blocks if present
      if (jsonStr.startsWith('```')) {
        jsonStr = jsonStr.replace(/^```json?\s*/i, '').replace(/\s*```$/, '');
      }

      const parsed = JSON.parse(jsonStr);

      if (parsed.recommendations && Array.isArray(parsed.recommendations)) {
        for (const r of parsed.recommendations) {
          recommendations.push({
            category: r.category || 'GENERAL',
            currentValue: r.currentValue || '',
            recommendedValue: r.recommendedValue || '',
            reasoning: r.reasoning || '',
            confidence: r.confidence || 'LOW',
            codeChange: r.codeChange,
          });
        }
      }
    } catch (e) {
      // If JSON parsing fails, try to extract recommendations from text
      const lines = response.split('\n');
      for (const line of lines) {
        if (line.includes('→') || line.includes('change to')) {
          recommendations.push({
            category: 'GENERAL',
            currentValue: '',
            recommendedValue: line.trim(),
            reasoning: 'Extracted from LLM response',
            confidence: 'LOW',
          });
        }
      }
    }

    return recommendations;
  }
}

// ─────────────────────────────────────────────────────────────────
// HELPER: Format recommendations for display
// ─────────────────────────────────────────────────────────────────

export function formatRecommendations(report: LLMReport): string {
  if (report.error) {
    return `⚠️ LLM Report: ${report.error}`;
  }

  if (report.recommendations.length === 0) {
    return `📊 LLM Report: No recommendations (all buckets performing adequately)`;
  }

  const lines: string[] = [`📊 LLM Report (${report.recommendations.length} recommendations):`];

  for (const r of report.recommendations.slice(0, 3)) {
    const confIcon = r.confidence === 'HIGH' ? '🟢' : r.confidence === 'MEDIUM' ? '🟡' : '🔴';
    lines.push(`${confIcon} [${r.category}] ${r.currentValue} → ${r.recommendedValue}`);
    if (r.codeChange) {
      lines.push(`   Code: ${r.codeChange}`);
    }
    lines.push(`   Reason: ${r.reasoning}`);
  }

  return lines.join('\n');
}
