/**
 * parameter-analyzer.ts
 *
 * Bucket analysis of closed trades to identify parameter weaknesses.
 * Groups trades by features (regime, BB position, volume, hold time)
 * and calculates win rate per bucket to find underperforming configs.
 *
 * Runs every N trades (default 50) and generates actionable suggestions.
 */

// ─────────────────────────────────────────────────────────────────
// INTERFACES
// ─────────────────────────────────────────────────────────────────

export interface ClosedTrade {
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  exitPrice: number;
  pnl: number;
  status: 'CLOSED';
  entryTime: number;
  exitTime: number;
  // Features for bucketing
  regime?: 'MOMENTUM' | 'RANGE' | 'NONE';
  bbPosition?: number;        // 0-1 where 0=lower band, 1=upper band
  volumeRatio?: number;       // Volume / avg volume
  rsi?: number;
  holdCandles?: number;       // How many candles held
  atrPercent?: number;        // ATR as % of price
  liquidityScore?: number;    // 0-100 from liquidity detector
  signalStrength?: number;    // 0-1
}

export interface Bucket {
  name: string;
  trades: ClosedTrade[];
  wins: number;
  losses: number;
  winRate: number;
  avgPnl: number;
  totalPnl: number;
}

export interface ParameterSuggestion {
  category: 'REGIME' | 'BB_THRESHOLD' | 'VOLUME' | 'HOLD_TIME' | 'LIQUIDITY' | 'RSI' | 'GENERAL';
  current: string;
  suggested: string;
  reason: string;
  impact: 'HIGH' | 'MEDIUM' | 'LOW';
  bucketName: string;
  winRate: number;
  sampleSize: number;
}

export interface AnalysisResult {
  timestamp: string;
  totalTrades: number;
  overallWinRate: number;
  buckets: Bucket[];
  suggestions: ParameterSuggestion[];
  summary: string;
}

export interface AnalyzerConfig {
  minTradesPerBucket: number;     // Minimum trades to consider bucket valid
  underperformThreshold: number;  // Win rate below this triggers suggestion (e.g., 0.35)
  overperformThreshold: number;   // Win rate above this for positive feedback
  triggerEveryNTrades: number;    // Analyze every N trades
}

export const DEFAULT_CONFIG: AnalyzerConfig = {
  minTradesPerBucket: 5,
  underperformThreshold: 0.40,
  overperformThreshold: 0.55,
  triggerEveryNTrades: 50,
};

// ─────────────────────────────────────────────────────────────────
// PARAMETER ANALYZER CLASS
// ─────────────────────────────────────────────────────────────────

export class ParameterAnalyzer {
  private config: AnalyzerConfig;
  private lastAnalysisTrades: number = 0;

  constructor(config: AnalyzerConfig = DEFAULT_CONFIG) {
    this.config = config;
  }

  /**
   * Analyze closed trades and generate suggestions
   */
  analyze(trades: ClosedTrade[]): AnalysisResult {
    const closedTrades = trades.filter(t => t.status === 'CLOSED');
    const totalTrades = closedTrades.length;
    const wins = closedTrades.filter(t => (t.pnl || 0) > 0).length;
    const overallWinRate = totalTrades > 0 ? wins / totalTrades : 0;

    // Build buckets
    const buckets = this.buildBuckets(closedTrades);

    // Generate suggestions from underperforming buckets
    const suggestions = this.generateSuggestions(buckets, overallWinRate);

    // Build summary
    const summary = this.buildSummary(buckets, suggestions, overallWinRate);

    return {
      timestamp: new Date().toISOString(),
      totalTrades,
      overallWinRate,
      buckets: buckets.filter(b => b.trades.length >= this.config.minTradesPerBucket),
      suggestions,
      summary,
    };
  }

  /**
   * Check if analysis should run based on trade count
   */
  shouldAnalyze(currentTradeCount: number): boolean {
    if (currentTradeCount < this.config.triggerEveryNTrades) return false;
    const tradesSinceLast = currentTradeCount - this.lastAnalysisTrades;
    return tradesSinceLast >= this.config.triggerEveryNTrades;
  }

  /**
   * Mark that analysis was performed
   */
  markAnalyzed(currentTradeCount: number): void {
    this.lastAnalysisTrades = currentTradeCount;
  }

  // ───────────────────────────────────────────────────────────────
  // PRIVATE: BUCKET BUILDERS
  // ───────────────────────────────────────────────────────────────

  private buildBuckets(trades: ClosedTrade[]): Bucket[] {
    const buckets: Bucket[] = [];

    // 1. Regime buckets
    buckets.push(...this.bucketByRegime(trades));

    // 2. BB position buckets
    buckets.push(...this.bucketByBBPosition(trades));

    // 3. Volume ratio buckets
    buckets.push(...this.bucketByVolume(trades));

    // 4. Hold time buckets
    buckets.push(...this.bucketByHoldTime(trades));

    // 5. Liquidity score buckets
    buckets.push(...this.bucketByLiquidity(trades));

    // 6. Direction buckets
    buckets.push(...this.bucketByDirection(trades));

    return buckets;
  }

  private bucketByRegime(trades: ClosedTrade[]): Bucket[] {
    const regimes = ['MOMENTUM', 'RANGE', 'NONE'] as const;
    return regimes.map(regime => {
      const bucketTrades = trades.filter(t => t.regime === regime);
      return this.createBucket(`Regime: ${regime}`, bucketTrades);
    });
  }

  private bucketByBBPosition(trades: ClosedTrade[]): Bucket[] {
    // LONG: BB < 20%, 20-40%, 40-60%, 60-80%, > 80%
    // SHORT: BB > 80%, 60-80%, 40-60%, 20-40%, < 20%
    const zones = [
      { name: 'BB: Extreme Low (<15%)', filter: (t: ClosedTrade) => (t.bbPosition ?? 0.5) < 0.15 },
      { name: 'BB: Low (15-30%)', filter: (t: ClosedTrade) => {
        const bb = t.bbPosition ?? 0.5;
        return bb >= 0.15 && bb < 0.30;
      }},
      { name: 'BB: Mid (30-70%)', filter: (t: ClosedTrade) => {
        const bb = t.bbPosition ?? 0.5;
        return bb >= 0.30 && bb < 0.70;
      }},
      { name: 'BB: High (70-85%)', filter: (t: ClosedTrade) => {
        const bb = t.bbPosition ?? 0.5;
        return bb >= 0.70 && bb < 0.85;
      }},
      { name: 'BB: Extreme High (>85%)', filter: (t: ClosedTrade) => (t.bbPosition ?? 0.5) >= 0.85 },
    ];

    return zones.map(zone => this.createBucket(zone.name, trades.filter(zone.filter)));
  }

  private bucketByVolume(trades: ClosedTrade[]): Bucket[] {
    const volumeZones = [
      { name: 'Volume: Low (<1.0x)', filter: (t: ClosedTrade) => (t.volumeRatio ?? 1) < 1.0 },
      { name: 'Volume: Normal (1.0-1.5x)', filter: (t: ClosedTrade) => {
        const v = t.volumeRatio ?? 1;
        return v >= 1.0 && v < 1.5;
      }},
      { name: 'Volume: High (1.5-2.0x)', filter: (t: ClosedTrade) => {
        const v = t.volumeRatio ?? 1;
        return v >= 1.5 && v < 2.0;
      }},
      { name: 'Volume: Spike (>2.0x)', filter: (t: ClosedTrade) => (t.volumeRatio ?? 1) >= 2.0 },
    ];

    return volumeZones.map(zone => this.createBucket(zone.name, trades.filter(zone.filter)));
  }

  private bucketByHoldTime(trades: ClosedTrade[]): Bucket[] {
    const holdZones = [
      { name: 'Hold: Quick (<3 candles)', filter: (t: ClosedTrade) => (t.holdCandles ?? 5) < 3 },
      { name: 'Hold: Short (3-5 candles)', filter: (t: ClosedTrade) => {
        const h = t.holdCandles ?? 5;
        return h >= 3 && h < 5;
      }},
      { name: 'Hold: Medium (5-10 candles)', filter: (t: ClosedTrade) => {
        const h = t.holdCandles ?? 5;
        return h >= 5 && h < 10;
      }},
      { name: 'Hold: Long (>10 candles)', filter: (t: ClosedTrade) => (t.holdCandles ?? 5) >= 10 },
    ];

    return holdZones.map(zone => this.createBucket(zone.name, trades.filter(zone.filter)));
  }

  private bucketByLiquidity(trades: ClosedTrade[]): Bucket[] {
    const liqZones = [
      { name: 'Liquidity: None (0)', filter: (t: ClosedTrade) => (t.liquidityScore ?? 0) === 0 },
      { name: 'Liquidity: Low (1-25)', filter: (t: ClosedTrade) => {
        const s = t.liquidityScore ?? 0;
        return s > 0 && s < 25;
      }},
      { name: 'Liquidity: Medium (25-50)', filter: (t: ClosedTrade) => {
        const s = t.liquidityScore ?? 0;
        return s >= 25 && s < 50;
      }},
      { name: 'Liquidity: High (50-75)', filter: (t: ClosedTrade) => {
        const s = t.liquidityScore ?? 0;
        return s >= 50 && s < 75;
      }},
      { name: 'Liquidity: Very High (75+)', filter: (t: ClosedTrade) => (t.liquidityScore ?? 0) >= 75 },
    ];

    return liqZones.map(zone => this.createBucket(zone.name, trades.filter(zone.filter)));
  }

  private bucketByDirection(trades: ClosedTrade[]): Bucket[] {
    return [
      this.createBucket('Direction: LONG', trades.filter(t => t.direction === 'LONG')),
      this.createBucket('Direction: SHORT', trades.filter(t => t.direction === 'SHORT')),
    ];
  }

  private createBucket(name: string, trades: ClosedTrade[]): Bucket {
    const wins = trades.filter(t => (t.pnl || 0) > 0).length;
    const losses = trades.filter(t => (t.pnl || 0) <= 0).length;
    const totalPnl = trades.reduce((sum, t) => sum + (t.pnl || 0), 0);
    const avgPnl = trades.length > 0 ? totalPnl / trades.length : 0;
    const winRate = trades.length > 0 ? wins / trades.length : 0;

    return {
      name,
      trades,
      wins,
      losses,
      winRate,
      avgPnl,
      totalPnl,
    };
  }

  // ───────────────────────────────────────────────────────────────
  // PRIVATE: SUGGESTION GENERATORS
  // ───────────────────────────────────────────────────────────────

  private generateSuggestions(buckets: Bucket[], overallWinRate: number): ParameterSuggestion[] {
    const suggestions: ParameterSuggestion[] = [];
    const threshold = this.config.underperformThreshold;
    const minTrades = this.config.minTradesPerBucket;

    for (const bucket of buckets) {
      if (bucket.trades.length < minTrades) continue;

      // Skip if win rate is acceptable
      if (bucket.winRate >= threshold) continue;

      // Generate suggestion based on bucket type
      const suggestion = this.bucketToSuggestion(bucket, overallWinRate);
      if (suggestion) {
        suggestions.push(suggestion);
      }
    }

    // Sort by impact (HIGH first) then by sample size
    return suggestions.sort((a, b) => {
      const impactOrder = { HIGH: 0, MEDIUM: 1, LOW: 2 };
      if (impactOrder[a.impact] !== impactOrder[b.impact]) {
        return impactOrder[a.impact] - impactOrder[b.impact];
      }
      return b.sampleSize - a.sampleSize;
    });
  }

  private bucketToSuggestion(bucket: Bucket, overallWinRate: number): ParameterSuggestion | null {
    const { name, winRate, trades } = bucket;

    // Determine impact based on how bad the underperformance is
    const impact: 'HIGH' | 'MEDIUM' | 'LOW' =
      winRate < 0.25 ? 'HIGH' :
      winRate < 0.35 ? 'MEDIUM' : 'LOW';

    // Regime suggestions
    if (name.includes('Regime:')) {
      const regime = name.split(': ')[1];
      return {
        category: 'REGIME',
        current: `Trading in ${regime} mode`,
        suggested: regime === 'NONE'
          ? 'Skip trades when regime=NONE (no edge in transition)'
          : `Tighten ${regime} mode entry criteria or reduce position size`,
        reason: `${regime} trades: ${(winRate * 100).toFixed(0)}% win rate vs ${(overallWinRate * 100).toFixed(0)}% overall`,
        impact,
        bucketName: name,
        winRate,
        sampleSize: trades.length,
      };
    }

    // BB position suggestions
    if (name.includes('BB:')) {
      const bbZone = name.split(': ')[1];
      if (name.includes('Extreme Low')) {
        return {
          category: 'BB_THRESHOLD',
          current: 'LONG at BB < 15%',
          suggested: 'Require BB < 10% for LONG entries (more extreme)',
          reason: `Extreme low BB entries: ${(winRate * 100).toFixed(0)}% win rate`,
          impact,
          bucketName: name,
          winRate,
          sampleSize: trades.length,
        };
      }
      if (name.includes('Extreme High')) {
        return {
          category: 'BB_THRESHOLD',
          current: 'SHORT at BB > 85%',
          suggested: 'Require BB > 90% for SHORT entries (more extreme)',
          reason: `Extreme high BB entries: ${(winRate * 100).toFixed(0)}% win rate`,
          impact,
          bucketName: name,
          winRate,
          sampleSize: trades.length,
        };
      }
      if (name.includes('Mid')) {
        return {
          category: 'BB_THRESHOLD',
          current: 'Allowing entries at BB 30-70%',
          suggested: 'Skip entries when BB is 30-70% (no edge in middle)',
          reason: `Mid-BB entries: ${(winRate * 100).toFixed(0)}% win rate`,
          impact,
          bucketName: name,
          winRate,
          sampleSize: trades.length,
        };
      }
    }

    // Volume suggestions
    if (name.includes('Volume:')) {
      if (name.includes('Low')) {
        return {
          category: 'VOLUME',
          current: 'Allowing entries with volume < 1.0x',
          suggested: 'Require volume >= 1.0x for all entries',
          reason: `Low volume entries: ${(winRate * 100).toFixed(0)}% win rate`,
          impact,
          bucketName: name,
          winRate,
          sampleSize: trades.length,
        };
      }
    }

    // Hold time suggestions
    if (name.includes('Hold:')) {
      if (name.includes('Long')) {
        return {
          category: 'HOLD_TIME',
          current: 'Allowing holds > 10 candles',
          suggested: 'Add time stop at 10 candles (trades losing edge)',
          reason: `Long holds: ${(winRate * 100).toFixed(0)}% win rate`,
          impact,
          bucketName: name,
          winRate,
          sampleSize: trades.length,
        };
      }
    }

    // Liquidity suggestions
    if (name.includes('Liquidity:')) {
      if (name.includes('None')) {
        return {
          category: 'LIQUIDITY',
          current: 'No liquidity confirmation required',
          suggested: 'Require minimum liquidity score >= 25 for entries',
          reason: `No liquidity trades: ${(winRate * 100).toFixed(0)}% win rate`,
          impact,
          bucketName: name,
          winRate,
          sampleSize: trades.length,
        };
      }
    }

    // Direction suggestions
    if (name.includes('Direction:')) {
      const dir = name.includes('LONG') ? 'LONG' : 'SHORT';
      if (winRate < 0.35) {
        return {
          category: 'GENERAL',
          current: `Taking ${dir} trades`,
          suggested: `${dir} signals underperforming - review entry criteria`,
          reason: `${dir} trades: ${(winRate * 100).toFixed(0)}% win rate`,
          impact,
          bucketName: name,
          winRate,
          sampleSize: trades.length,
        };
      }
    }

    return null;
  }

  // ───────────────────────────────────────────────────────────────
  // PRIVATE: SUMMARY BUILDER
  // ───────────────────────────────────────────────────────────────

  private buildSummary(buckets: Bucket[], suggestions: ParameterSuggestion[], overallWinRate: number): string {
    const lines: string[] = [];
    lines.push(`Overall: ${(overallWinRate * 100).toFixed(1)}% win rate`);

    // Find worst buckets
    const worstBuckets = buckets
      .filter(b => b.trades.length >= this.config.minTradesPerBucket)
      .sort((a, b) => a.winRate - b.winRate)
      .slice(0, 3);

    if (worstBuckets.length > 0) {
      lines.push(`Worst: ${worstBuckets.map(b => `${b.name} (${(b.winRate * 100).toFixed(0)}%)`).join(', ')}`);
    }

    // Find best buckets
    const bestBuckets = buckets
      .filter(b => b.trades.length >= this.config.minTradesPerBucket)
      .sort((a, b) => b.winRate - a.winRate)
      .slice(0, 3);

    if (bestBuckets.length > 0) {
      lines.push(`Best: ${bestBuckets.map(b => `${b.name} (${(b.winRate * 100).toFixed(0)}%)`).join(', ')}`);
    }

    if (suggestions.length > 0) {
      lines.push(`${suggestions.length} suggestions generated`);
    }

    return lines.join(' | ');
  }
}

// ─────────────────────────────────────────────────────────────────
// HELPER: Format suggestions for display
// ─────────────────────────────────────────────────────────────────

export function formatSuggestions(suggestions: ParameterSuggestion[]): string {
  if (suggestions.length === 0) return 'No suggestions (all buckets performing adequately)';

  const lines: string[] = [];
  for (const s of suggestions.slice(0, 5)) {  // Top 5 only
    const impactIcon = s.impact === 'HIGH' ? '🔴' : s.impact === 'MEDIUM' ? '🟡' : '🟢';
    lines.push(`${impactIcon} [${s.category}] ${s.current} → ${s.suggested}`);
    lines.push(`   Reason: ${s.reason} (n=${s.sampleSize})`);
  }

  return lines.join('\n');
}
