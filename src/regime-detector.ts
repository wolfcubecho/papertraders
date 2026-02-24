/**
 * regime-detector.ts
 *
 * Portfolio-level regime classification.
 * Aggregates signals across ALL coins to determine:
 *   - RISK_ON: Strong trends, high ADX, expanding BBs -> full size
 *   - RISK_OFF: Weak/volatile, low ADX, contracting BBs -> reduce size
 *   - TRANSITION: Mixed signals -> moderate size
 *
 * This gates position sizing based on overall market health,
 * NOT individual coin conditions.
 */

// ─────────────────────────────────────────────────────────────────
// INTERFACES
// ─────────────────────────────────────────────────────────────────

export interface CoinRegimeSignals {
  symbol: string;
  adx: number;                    // 0-100 trend strength
  bbWidth: number;                // Current BB width (upper-lower)/middle
  bbWidthAvg: number;             // Average BB width
  volumeRatio: number;            // Current volume / avg volume
  atrPercent: number;             // ATR as % of price
  priceChange24h?: number;        // 24h price change % (optional)
}

export interface PortfolioRegime {
  regime: 'RISK_ON' | 'RISK_OFF' | 'TRANSITION';
  confidence: number;             // 0-100 how confident in classification
  sizeMultiplier: number;         // 0.85 - 1.0 position sizing (less punitive)
  metrics: {
    avgAdx: number;
    avgVolumeRatio: number;
    bbExpansionRatio: number;     // avg bbWidth / avg bbWidthAvg
    trendingCoinCount: number;    // Coins with ADX > 25
    totalCoinCount: number;
  };
  reasons: string[];              // Why this regime was chosen
}

export interface RegimeConfig {
  // ADX thresholds
  adxRiskOnMin: number;           // Avg ADX >= this = RISK_ON
  adxRiskOffMax: number;          // Avg ADX <= this = RISK_OFF

  // BB expansion
  bbExpansionRiskOn: number;      // BB width/avg >= this = expanding
  bbContractionRiskOff: number;   // BB width/avg <= this = contracting

  // Volume
  volumeRiskOnMin: number;        // Avg volume ratio for RISK_ON
  volumeRiskOffMax: number;       // Avg volume ratio for RISK_OFF

  // Trending coins
  trendingCoinPctRiskOn: number;  // % of coins that must be trending

  // Size multipliers (less punitive - we gate instead of shrink)
  riskOnMultiplier: number;
  riskOffMultiplier: number;      // Should be 0.85, not 0.5
  transitionMultiplier: number;

  // Gating config (skip weak setups in weak markets)
  minCoinAdxForRiskOff: number;   // Coin must have ADX >= this in RISK_OFF to trade
  minCoinAdxForTransition: number; // Coin must have ADX >= this in TRANSITION to trade
}

// ─────────────────────────────────────────────────────────────────
// DEFAULT CONFIGS
// ─────────────────────────────────────────────────────────────────

export const SCALP_REGIME_CONFIG: RegimeConfig = {
  adxRiskOnMin: 28,
  adxRiskOffMax: 18,
  bbExpansionRiskOn: 1.2,
  bbContractionRiskOff: 0.8,
  volumeRiskOnMin: 1.3,
  volumeRiskOffMax: 0.7,
  trendingCoinPctRiskOn: 0.45,
  // Less punitive sizing - gate weak setups instead
  riskOnMultiplier: 1.0,
  riskOffMultiplier: 0.85,      // Was 0.5, now 0.85
  transitionMultiplier: 0.9,   // Was 0.75, now 0.9
  // Gating thresholds
  minCoinAdxForRiskOff: 25,     // In RISK_OFF, skip if coin ADX < 25
  minCoinAdxForTransition: 28, // In TRANSITION, skip if coin ADX < 28
};

export const SWING_REGIME_CONFIG: RegimeConfig = {
  adxRiskOnMin: 30,
  adxRiskOffMax: 20,
  bbExpansionRiskOn: 1.3,
  bbContractionRiskOff: 0.75,
  volumeRiskOnMin: 1.2,
  volumeRiskOffMax: 0.6,
  trendingCoinPctRiskOn: 0.50,
  // Less punitive sizing - gate weak setups instead
  riskOnMultiplier: 1.0,
  riskOffMultiplier: 0.85,      // Was 0.5, now 0.85
  transitionMultiplier: 0.9,   // Was 0.75, now 0.9
  // Gating thresholds
  minCoinAdxForRiskOff: 25,     // In RISK_OFF, skip if coin ADX < 25
  minCoinAdxForTransition: 28, // In TRANSITION, skip if coin ADX < 28
};

// ─────────────────────────────────────────────────────────────────
// REGIME DETECTOR CLASS
// ─────────────────────────────────────────────────────────────────

export class RegimeDetector {
  private config: RegimeConfig;
  private lastRegime: PortfolioRegime | null = null;
  private regimeHistory: PortfolioRegime[] = [];
  private maxHistoryLength = 100;

  constructor(config: RegimeConfig) {
    this.config = config;
  }

  /**
   * Classify portfolio regime from all coin signals
   */
  classify(signals: CoinRegimeSignals[]): PortfolioRegime {
    if (signals.length === 0) {
      return this.createDefaultRegime('No signals available');
    }

    // Calculate aggregate metrics
    const metrics = this.calculateMetrics(signals);

    // Score each dimension
    const scores = this.scoreDimensions(metrics);

    // Determine regime from scores
    const regime = this.determineRegime(scores, metrics);

    // Store in history
    this.lastRegime = regime;
    this.regimeHistory.push(regime);
    if (this.regimeHistory.length > this.maxHistoryLength) {
      this.regimeHistory.shift();
    }

    return regime;
  }

  /**
   * Get last classified regime (without re-computing)
   */
  getLastRegime(): PortfolioRegime | null {
    return this.lastRegime;
  }

  /**
   * Get regime history for trend analysis
   */
  getHistory(): PortfolioRegime[] {
    return [...this.regimeHistory];
  }

  /**
   * Check if regime has recently changed
   */
  hasRegimeChanged(lastN: number = 3): boolean {
    if (this.regimeHistory.length < 2) return false;

    const recent = this.regimeHistory.slice(-lastN);
    const regimes = new Set(recent.map(r => r.regime));
    return regimes.size > 1;
  }

  /**
   * Check if a specific coin should be GATED (skipped) based on portfolio regime
   * In RISK_OFF/TRANSITION, we skip weak setups, don't just reduce size
   */
  shouldGateCoin(coinAdx: number, coinRegime: 'MOMENTUM' | 'RANGE' | 'NONE'): boolean {
    if (!this.lastRegime) return false;

    const { regime } = this.lastRegime;

    if (regime === 'RISK_ON') {
      // Full risk-on: don't gate anything
      return false;
    }

    if (regime === 'RISK_OFF') {
      // Gate weak setups: skip if coin ADX too low OR coin in NONE regime
      if (coinAdx < (this.config as any).minCoinAdxForRiskOff) return true;
      if (coinRegime === 'NONE') return true;
      return false;
    }

    if (regime === 'TRANSITION') {
      // Transition: more selective
      if (coinAdx < (this.config as any).minCoinAdxForTransition) return true;
      if (coinRegime === 'NONE') return true;
      return false;
    }

    return false;
  }

  // ───────────────────────────────────────────────────────────────
  // PRIVATE METHODS
  // ───────────────────────────────────────────────────────────────

  private calculateMetrics(signals: CoinRegimeSignals[]) {
    const sum = signals.reduce((acc, s) => ({
      adx: acc.adx + s.adx,
      volumeRatio: acc.volumeRatio + s.volumeRatio,
      bbExpansion: acc.bbExpansion + (s.bbWidth / Math.max(s.bbWidthAvg, 0.001)),
      trendingCount: acc.trendingCount + (s.adx > 25 ? 1 : 0),
    }), { adx: 0, volumeRatio: 0, bbExpansion: 0, trendingCount: 0 });

    const n = signals.length;

    return {
      avgAdx: sum.adx / n,
      avgVolumeRatio: sum.volumeRatio / n,
      bbExpansionRatio: sum.bbExpansion / n,
      trendingCoinCount: sum.trendingCount,
      totalCoinCount: n,
    };
  }

  private scoreDimensions(metrics: ReturnType<typeof this.calculateMetrics>) {
    const { config } = this;

    // ADX score: 0-100 (higher = more trending)
    let adxScore = 0;
    if (metrics.avgAdx >= config.adxRiskOnMin) {
      adxScore = 100;
    } else if (metrics.avgAdx <= config.adxRiskOffMax) {
      adxScore = 0;
    } else {
      const range = config.adxRiskOnMin - config.adxRiskOffMax;
      adxScore = ((metrics.avgAdx - config.adxRiskOffMax) / range) * 100;
    }

    // BB expansion score: 0-100 (higher = expanding)
    let bbScore = 0;
    if (metrics.bbExpansionRatio >= config.bbExpansionRiskOn) {
      bbScore = 100;
    } else if (metrics.bbExpansionRatio <= config.bbContractionRiskOff) {
      bbScore = 0;
    } else {
      const range = config.bbExpansionRiskOn - config.bbContractionRiskOff;
      bbScore = ((metrics.bbExpansionRatio - config.bbContractionRiskOff) / range) * 100;
    }

    // Volume score: 0-100 (higher = more volume)
    let volumeScore = 0;
    if (metrics.avgVolumeRatio >= config.volumeRiskOnMin) {
      volumeScore = 100;
    } else if (metrics.avgVolumeRatio <= config.volumeRiskOffMax) {
      volumeScore = 0;
    } else {
      const range = config.volumeRiskOnMin - config.volumeRiskOffMax;
      volumeScore = ((metrics.avgVolumeRatio - config.volumeRiskOffMax) / range) * 100;
    }

    // Trending coins score
    const trendingPct = metrics.trendingCoinCount / metrics.totalCoinCount;
    let trendScore = trendingPct >= config.trendingCoinPctRiskOn ? 100 : trendingPct * 100;

    return {
      adx: adxScore,
      bb: bbScore,
      volume: volumeScore,
      trend: trendScore,
      average: (adxScore + bbScore + volumeScore + trendScore) / 4,
    };
  }

  private determineRegime(
    scores: ReturnType<typeof this.scoreDimensions>,
    metrics: ReturnType<typeof this.calculateMetrics>
  ): PortfolioRegime {
    const { config } = this;
    const reasons: string[] = [];

    // Decision logic: need majority of dimensions to agree
    const riskOnVotes = [
      scores.adx >= 60,
      scores.bb >= 60,
      scores.volume >= 60,
      scores.trend >= 60,
    ].filter(Boolean).length;

    const riskOffVotes = [
      scores.adx <= 30,
      scores.bb <= 30,
      scores.volume <= 30,
      scores.trend <= 30,
    ].filter(Boolean).length;

    let regime: 'RISK_ON' | 'RISK_OFF' | 'TRANSITION';
    let confidence: number;
    let sizeMultiplier: number;

    if (riskOnVotes >= 3) {
      regime = 'RISK_ON';
      confidence = Math.min(100, 60 + riskOnVotes * 10);
      sizeMultiplier = config.riskOnMultiplier;
      reasons.push(`Strong trends: ${riskOnVotes}/4 dimensions bullish`);
      reasons.push(`ADX avg: ${metrics.avgAdx.toFixed(1)}`);
      reasons.push(`Trending coins: ${metrics.trendingCoinCount}/${metrics.totalCoinCount}`);
    } else if (riskOffVotes >= 3) {
      regime = 'RISK_OFF';
      confidence = Math.min(100, 60 + riskOffVotes * 10);
      sizeMultiplier = config.riskOffMultiplier;
      reasons.push(`Weak market: ${riskOffVotes}/4 dimensions bearish`);
      reasons.push(`ADX avg: ${metrics.avgAdx.toFixed(1)}`);
      reasons.push(`Low volume ratio: ${metrics.avgVolumeRatio.toFixed(2)}x`);
    } else {
      regime = 'TRANSITION';
      confidence = 50;
      sizeMultiplier = config.transitionMultiplier;
      reasons.push(`Mixed signals: ${riskOnVotes} bullish, ${riskOffVotes} bearish`);
      reasons.push(`ADX avg: ${metrics.avgAdx.toFixed(1)}`);
    }

    // Adjust confidence based on score consistency
    const scoreVariance = this.calculateVariance([
      scores.adx, scores.bb, scores.volume, scores.trend
    ]);
    if (scoreVariance > 500) {
      confidence = Math.max(30, confidence - 15);
      reasons.push('High signal variance - lower confidence');
    }

    return {
      regime,
      confidence,
      sizeMultiplier,
      metrics: {
        avgAdx: metrics.avgAdx,
        avgVolumeRatio: metrics.avgVolumeRatio,
        bbExpansionRatio: metrics.bbExpansionRatio,
        trendingCoinCount: metrics.trendingCoinCount,
        totalCoinCount: metrics.totalCoinCount,
      },
      reasons,
    };
  }

  private calculateVariance(values: number[]): number {
    const mean = values.reduce((a, b) => a + b, 0) / values.length;
    return values.reduce((acc, v) => acc + Math.pow(v - mean, 2), 0) / values.length;
  }

  private createDefaultRegime(reason: string): PortfolioRegime {
    return {
      regime: 'TRANSITION',
      confidence: 30,
      sizeMultiplier: this.config.transitionMultiplier,
      metrics: {
        avgAdx: 0,
        avgVolumeRatio: 0,
        bbExpansionRatio: 0,
        trendingCoinCount: 0,
        totalCoinCount: 0,
      },
      reasons: [reason],
    };
  }
}

// ─────────────────────────────────────────────────────────────────
// HELPER: Extract signals from trader's timeframe data
// ─────────────────────────────────────────────────────────────────

export function extractRegimeSignals(
  symbol: string,
  momentum: {
    adx: number;
    bbWidth: number;
    bbWidthAvg: number;
    volumeRatio: number;
    atrPercent: number;
  }
): CoinRegimeSignals {
  return {
    symbol,
    adx: momentum.adx || 0,
    bbWidth: momentum.bbWidth || 0,
    bbWidthAvg: momentum.bbWidthAvg || 1,
    volumeRatio: momentum.volumeRatio || 1,
    atrPercent: momentum.atrPercent || 0,
  };
}
