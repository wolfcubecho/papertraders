/**
 * Ensemble Strategy Voting System
 * Combines multiple signal sources with weighted consensus
 *
 * Signal Sources:
 * - SMC Score (35%) - Our existing order block/FVG/liquidity analysis
 * - ML Win Probability (30%) - H2O model predictions
 * - Momentum (20%) - RSI + Rate of Change
 * - Mean Reversion (15%) - Bollinger Bands deviation
 *
 * Requires 60% consensus + 50% confidence to pass
 */

export type SignalDirection = 'LONG' | 'SHORT' | 'NEUTRAL';

export interface StrategySignal {
  name: string;
  direction: SignalDirection;
  confidence: number;  // 0-1
  weight: number;      // 0-1, should sum to 1.0
  reasoning: string;
}

export interface EnsembleResult {
  direction: SignalDirection;
  consensus: number;       // 0-1, how much agreement
  confidence: number;      // 0-1, weighted confidence
  approved: boolean;
  signals: StrategySignal[];
  vetoReason?: string;
}

export interface EnsembleConfig {
  minConsensus: number;      // Default 0.6 (60%)
  minConfidence: number;     // Default 0.5 (50%)
  weights: {
    smc: number;
    ml: number;
    momentum: number;
    meanReversion: number;
  };
}

const DEFAULT_CONFIG: EnsembleConfig = {
  minConsensus: 0.60,
  minConfidence: 0.50,
  weights: {
    smc: 0.35,
    ml: 0.30,
    momentum: 0.20,
    meanReversion: 0.15,
  }
};

export interface MarketData {
  // Price data
  close: number;
  high: number;
  low: number;
  open: number;
  volume: number;

  // Pre-calculated indicators (from our existing system)
  smcScore?: number;          // 0-100
  smcDirection?: SignalDirection;
  mlWinProbability?: number;  // 0-1
  mlDirection?: SignalDirection;

  // Raw data for momentum/mean reversion calculation
  closes: number[];           // Last N closes for indicator calc
  highs: number[];
  lows: number[];
  volumes: number[];
}

export class EnsembleVoting {
  private config: EnsembleConfig;

  constructor(config: Partial<EnsembleConfig> = {}) {
    this.config = {
      ...DEFAULT_CONFIG,
      ...config,
      weights: { ...DEFAULT_CONFIG.weights, ...config.weights }
    };

    // Normalize weights
    const totalWeight = Object.values(this.config.weights).reduce((a, b) => a + b, 0);
    if (Math.abs(totalWeight - 1.0) > 0.01) {
      console.warn(`[Ensemble] Weights sum to ${totalWeight}, normalizing...`);
      for (const key of Object.keys(this.config.weights) as Array<keyof typeof this.config.weights>) {
        this.config.weights[key] /= totalWeight;
      }
    }
  }

  /**
   * Get ensemble vote on a trading opportunity
   */
  vote(data: MarketData): EnsembleResult {
    const signals: StrategySignal[] = [];

    // 1. SMC Signal (from existing analysis)
    signals.push(this.getSMCSignal(data));

    // 2. ML Signal (from H2O predictions)
    signals.push(this.getMLSignal(data));

    // 3. Momentum Signal (RSI + ROC)
    signals.push(this.getMomentumSignal(data));

    // 4. Mean Reversion Signal (Bollinger Bands)
    signals.push(this.getMeanReversionSignal(data));

    // Calculate weighted consensus
    const result = this.calculateConsensus(signals);

    // Log the vote
    console.log(`\n[Ensemble] Vote Results:`);
    for (const sig of signals) {
      const arrow = sig.direction === 'LONG' ? '↑' : sig.direction === 'SHORT' ? '↓' : '→';
      console.log(`  ${sig.name.padEnd(15)} ${arrow} ${sig.direction.padEnd(7)} ${(sig.confidence * 100).toFixed(0)}% conf (${(sig.weight * 100).toFixed(0)}% weight)`);
      console.log(`    └─ ${sig.reasoning}`);
    }
    console.log(`  ${'─'.repeat(50)}`);
    console.log(`  CONSENSUS: ${(result.consensus * 100).toFixed(0)}% | CONFIDENCE: ${(result.confidence * 100).toFixed(0)}%`);
    console.log(`  DIRECTION: ${result.direction} | APPROVED: ${result.approved ? '✅ YES' : '❌ NO'}`);
    if (result.vetoReason) {
      console.log(`  VETO: ${result.vetoReason}`);
    }

    return result;
  }

  /**
   * SMC Signal - Order Blocks, FVGs, Liquidity
   */
  private getSMCSignal(data: MarketData): StrategySignal {
    const score = data.smcScore ?? 50;
    const direction = data.smcDirection ?? 'NEUTRAL';

    // Convert score to confidence (0-100 → 0-1)
    // Scores below 40 = low confidence, above 70 = high confidence
    let confidence = 0;
    if (score >= 70) {
      confidence = 0.7 + ((score - 70) / 30) * 0.3;  // 70-100 → 0.7-1.0
    } else if (score >= 40) {
      confidence = 0.3 + ((score - 40) / 30) * 0.4;  // 40-70 → 0.3-0.7
    } else {
      confidence = score / 40 * 0.3;  // 0-40 → 0-0.3
    }

    let reasoning = '';
    if (score >= 80) reasoning = 'Strong SMC setup - multiple confluences';
    else if (score >= 60) reasoning = 'Decent SMC setup - some confluence';
    else if (score >= 40) reasoning = 'Weak SMC setup - limited confluence';
    else reasoning = 'No valid SMC setup';

    return {
      name: 'SMC',
      direction,
      confidence,
      weight: this.config.weights.smc,
      reasoning
    };
  }

  /**
   * ML Signal - H2O Model Prediction
   */
  private getMLSignal(data: MarketData): StrategySignal {
    const winProb = data.mlWinProbability ?? 0.5;
    const direction = data.mlDirection ?? (winProb > 0.5 ? 'LONG' : winProb < 0.5 ? 'SHORT' : 'NEUTRAL');

    // Win probability directly maps to confidence
    // But we want to penalize near-50% predictions
    const confidence = Math.abs(winProb - 0.5) * 2;  // 0.5 → 0, 0.75 → 0.5, 1.0 → 1.0

    let reasoning = '';
    if (winProb >= 0.7) reasoning = `High win probability (${(winProb * 100).toFixed(0)}%)`;
    else if (winProb >= 0.55) reasoning = `Moderate win probability (${(winProb * 100).toFixed(0)}%)`;
    else if (winProb <= 0.3) reasoning = `High loss probability (${((1 - winProb) * 100).toFixed(0)}% loss)`;
    else if (winProb <= 0.45) reasoning = `Moderate loss probability (${((1 - winProb) * 100).toFixed(0)}% loss)`;
    else reasoning = `Coin flip territory (${(winProb * 100).toFixed(0)}%)`;

    return {
      name: 'ML Model',
      direction,
      confidence,
      weight: this.config.weights.ml,
      reasoning
    };
  }

  /**
   * Momentum Signal - RSI + Rate of Change
   */
  private getMomentumSignal(data: MarketData): StrategySignal {
    if (!data.closes || data.closes.length < 20) {
      return {
        name: 'Momentum',
        direction: 'NEUTRAL',
        confidence: 0,
        weight: this.config.weights.momentum,
        reasoning: 'Insufficient data for momentum calculation'
      };
    }

    // Calculate RSI (14-period)
    const rsi = this.calculateRSI(data.closes, 14);

    // Calculate Rate of Change (10-period)
    const roc = this.calculateROC(data.closes, 10);

    // Determine direction and confidence
    let direction: SignalDirection = 'NEUTRAL';
    let confidence = 0;
    let reasoning = '';

    // RSI signals
    const rsiSignal = rsi > 70 ? -1 : rsi < 30 ? 1 : 0;  // Overbought/oversold
    const rsiMomentum = rsi > 50 ? 1 : rsi < 50 ? -1 : 0; // Momentum direction

    // ROC signals
    const rocSignal = roc > 5 ? 1 : roc < -5 ? -1 : 0;   // Strong momentum
    const rocTrend = roc > 0 ? 1 : roc < 0 ? -1 : 0;     // Direction

    // Combine signals
    const combined = rsiMomentum + rocTrend;

    if (combined >= 2) {
      direction = 'LONG';
      confidence = 0.7 + Math.min(0.3, (roc / 20));
      reasoning = `Strong bullish momentum - RSI ${rsi.toFixed(0)}, ROC ${roc.toFixed(1)}%`;
    } else if (combined <= -2) {
      direction = 'SHORT';
      confidence = 0.7 + Math.min(0.3, Math.abs(roc / 20));
      reasoning = `Strong bearish momentum - RSI ${rsi.toFixed(0)}, ROC ${roc.toFixed(1)}%`;
    } else if (combined === 1) {
      direction = 'LONG';
      confidence = 0.4 + Math.min(0.2, (roc / 30));
      reasoning = `Mild bullish momentum - RSI ${rsi.toFixed(0)}, ROC ${roc.toFixed(1)}%`;
    } else if (combined === -1) {
      direction = 'SHORT';
      confidence = 0.4 + Math.min(0.2, Math.abs(roc / 30));
      reasoning = `Mild bearish momentum - RSI ${rsi.toFixed(0)}, ROC ${roc.toFixed(1)}%`;
    } else {
      direction = 'NEUTRAL';
      confidence = 0.2;
      reasoning = `No clear momentum - RSI ${rsi.toFixed(0)}, ROC ${roc.toFixed(1)}%`;
    }

    // Overbought/oversold override for mean reversion potential
    if (rsi > 80) {
      reasoning += ' [OVERBOUGHT WARNING]';
      if (direction === 'LONG') confidence *= 0.5;
    } else if (rsi < 20) {
      reasoning += ' [OVERSOLD WARNING]';
      if (direction === 'SHORT') confidence *= 0.5;
    }

    return {
      name: 'Momentum',
      direction,
      confidence: Math.min(1, Math.max(0, confidence)),
      weight: this.config.weights.momentum,
      reasoning
    };
  }

  /**
   * Mean Reversion Signal - Bollinger Bands
   */
  private getMeanReversionSignal(data: MarketData): StrategySignal {
    if (!data.closes || data.closes.length < 20) {
      return {
        name: 'Mean Reversion',
        direction: 'NEUTRAL',
        confidence: 0,
        weight: this.config.weights.meanReversion,
        reasoning: 'Insufficient data for Bollinger Bands'
      };
    }

    // Calculate Bollinger Bands (20-period, 2 std dev)
    const { upper, middle, lower, percentB } = this.calculateBollingerBands(data.closes, 20, 2);
    const currentPrice = data.close;

    let direction: SignalDirection = 'NEUTRAL';
    let confidence = 0;
    let reasoning = '';

    // Percent B: 0 = at lower band, 1 = at upper band, 0.5 = at middle
    if (percentB <= 0) {
      // At or below lower band - potential long
      direction = 'LONG';
      confidence = 0.6 + Math.min(0.4, Math.abs(percentB) * 0.4);
      reasoning = `Price at lower BB (${percentB.toFixed(2)}) - mean reversion long`;
    } else if (percentB >= 1) {
      // At or above upper band - potential short
      direction = 'SHORT';
      confidence = 0.6 + Math.min(0.4, (percentB - 1) * 0.4);
      reasoning = `Price at upper BB (${percentB.toFixed(2)}) - mean reversion short`;
    } else if (percentB <= 0.2) {
      direction = 'LONG';
      confidence = 0.4 + (0.2 - percentB) * 2;
      reasoning = `Price near lower BB (${percentB.toFixed(2)}) - weak long`;
    } else if (percentB >= 0.8) {
      direction = 'SHORT';
      confidence = 0.4 + (percentB - 0.8) * 2;
      reasoning = `Price near upper BB (${percentB.toFixed(2)}) - weak short`;
    } else {
      direction = 'NEUTRAL';
      confidence = 0.2;
      reasoning = `Price in middle of BB (${percentB.toFixed(2)}) - no signal`;
    }

    // Add band width context
    const bandWidth = (upper - lower) / middle;
    if (bandWidth < 0.02) {
      reasoning += ' [SQUEEZE - expect breakout]';
      confidence *= 0.7;  // Less reliable during squeeze
    } else if (bandWidth > 0.08) {
      reasoning += ' [WIDE BANDS - high volatility]';
    }

    return {
      name: 'Mean Reversion',
      direction,
      confidence: Math.min(1, Math.max(0, confidence)),
      weight: this.config.weights.meanReversion,
      reasoning
    };
  }

  /**
   * Calculate weighted consensus from all signals
   */
  private calculateConsensus(signals: StrategySignal[]): EnsembleResult {
    let longScore = 0;
    let shortScore = 0;
    let totalWeight = 0;
    let weightedConfidence = 0;

    for (const signal of signals) {
      if (signal.direction === 'LONG') {
        longScore += signal.weight * signal.confidence;
      } else if (signal.direction === 'SHORT') {
        shortScore += signal.weight * signal.confidence;
      }
      totalWeight += signal.weight;
      weightedConfidence += signal.weight * signal.confidence;
    }

    // Normalize
    longScore /= totalWeight;
    shortScore /= totalWeight;
    weightedConfidence /= totalWeight;

    // Determine direction
    let direction: SignalDirection;
    let consensus: number;

    if (longScore > shortScore && longScore > 0.1) {
      direction = 'LONG';
      consensus = longScore / (longScore + shortScore + 0.001);
    } else if (shortScore > longScore && shortScore > 0.1) {
      direction = 'SHORT';
      consensus = shortScore / (longScore + shortScore + 0.001);
    } else {
      direction = 'NEUTRAL';
      consensus = 0;
    }

    // Check approval
    let approved = true;
    let vetoReason: string | undefined;

    if (direction === 'NEUTRAL') {
      approved = false;
      vetoReason = 'No clear direction consensus';
    } else if (consensus < this.config.minConsensus) {
      approved = false;
      vetoReason = `Consensus ${(consensus * 100).toFixed(0)}% < required ${(this.config.minConsensus * 100).toFixed(0)}%`;
    } else if (weightedConfidence < this.config.minConfidence) {
      approved = false;
      vetoReason = `Confidence ${(weightedConfidence * 100).toFixed(0)}% < required ${(this.config.minConfidence * 100).toFixed(0)}%`;
    }

    return {
      direction,
      consensus,
      confidence: weightedConfidence,
      approved,
      signals,
      vetoReason
    };
  }

  // ============ Technical Indicator Calculations ============

  private calculateRSI(closes: number[], period: number = 14): number {
    if (closes.length < period + 1) return 50;

    const recent = closes.slice(-period - 1);
    let gains = 0;
    let losses = 0;

    for (let i = 1; i < recent.length; i++) {
      const change = recent[i] - recent[i - 1];
      if (change > 0) gains += change;
      else losses -= change;
    }

    const avgGain = gains / period;
    const avgLoss = losses / period;

    if (avgLoss === 0) return 100;
    const rs = avgGain / avgLoss;
    return 100 - (100 / (1 + rs));
  }

  private calculateROC(closes: number[], period: number = 10): number {
    if (closes.length < period + 1) return 0;
    const current = closes[closes.length - 1];
    const past = closes[closes.length - period - 1];
    return ((current - past) / past) * 100;
  }

  private calculateBollingerBands(closes: number[], period: number = 20, stdDev: number = 2): {
    upper: number;
    middle: number;
    lower: number;
    percentB: number;
  } {
    const recent = closes.slice(-period);
    const middle = recent.reduce((a, b) => a + b, 0) / period;

    const variance = recent.reduce((sum, val) => sum + Math.pow(val - middle, 2), 0) / period;
    const std = Math.sqrt(variance);

    const upper = middle + stdDev * std;
    const lower = middle - stdDev * std;

    const current = closes[closes.length - 1];
    const percentB = (current - lower) / (upper - lower);

    return { upper, middle, lower, percentB };
  }
}

// Export singleton
export const ensembleVoter = new EnsembleVoting();
