/**
 * Trade Decision Pipeline - Multi-Layer Veto System
 *
 * Pipeline:
 * SMC Signal → ML Filter → Ensemble Vote → Sentiment Check → Correlation Check → RL Sizing → Execute
 *      ↓            ↓            ↓               ↓                ↓               ↓
 *   (veto)      (veto)       (veto)          (veto)           (veto)          (size)
 *
 * Each layer can VETO the trade before it reaches execution.
 * Only trades that pass ALL layers get sized by RL and executed.
 */

import { EnsembleVoting, EnsembleResult, MarketData, SignalDirection } from './ensemble-voting.js';
import { RLPositionSizer, MarketState, PositionAction } from './rl-position-sizer.js';

export interface TradeSetup {
  symbol: string;
  direction: SignalDirection;
  entryPrice: number;
  stopLoss: number;
  takeProfit: number;

  // Data for analysis
  marketData: MarketData;

  // Pre-calculated scores
  smcScore: number;
  smcDirection: SignalDirection;
  mlWinProbability: number;
  mlDirection: SignalDirection;

  // Optional context
  sentimentScore?: number;      // -1 to 1
  correlatedPositions?: string[]; // Symbols of correlated open positions
  currentDrawdown?: number;     // Current portfolio drawdown
  openPositionCount?: number;   // Number of open positions
  recentWinRate?: number;       // Recent trade win rate
}

export interface PipelineResult {
  approved: boolean;
  vetoedAt?: string;
  vetoReason?: string;

  // Results from each layer
  smcResult?: { passed: boolean; score: number; reason: string };
  mlResult?: { passed: boolean; winProb: number; reason: string };
  ensembleResult?: EnsembleResult;
  sentimentResult?: { passed: boolean; score: number; reason: string };
  correlationResult?: { passed: boolean; correlations: string[]; reason: string };

  // Final sizing (only if approved)
  positionSize?: PositionAction;

  // Summary
  passedLayers: string[];
  totalLayers: number;
}

export interface PipelineConfig {
  // SMC thresholds
  minSMCScore: number;           // Default 40

  // ML thresholds
  minMLWinProbability: number;   // Default 0.45 (allow slightly below 50%)
  maxMLLossProbability: number;  // Default 0.65 (veto if >65% loss probability)

  // Sentiment thresholds
  useSentiment: boolean;
  minSentimentScore: number;     // Default -0.3 (allow slightly negative)
  sentimentVetoThreshold: number; // Default -0.6 (strong negative = veto)

  // Correlation thresholds
  useCorrelation: boolean;
  maxCorrelatedPositions: number; // Default 2

  // General
  maxOpenPositions: number;      // Default 5
  maxDrawdownToTrade: number;    // Default 0.20 (20%)
}

const DEFAULT_CONFIG: PipelineConfig = {
  minSMCScore: 40,
  minMLWinProbability: 0.45,
  maxMLLossProbability: 0.65,
  useSentiment: true,
  minSentimentScore: -0.3,
  sentimentVetoThreshold: -0.6,
  useCorrelation: true,
  maxCorrelatedPositions: 2,
  maxOpenPositions: 5,
  maxDrawdownToTrade: 0.20,
};

export class TradeDecisionPipeline {
  private config: PipelineConfig;
  private ensemble: EnsembleVoting;
  private rlSizer: RLPositionSizer;

  constructor(config: Partial<PipelineConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.ensemble = new EnsembleVoting();
    this.rlSizer = new RLPositionSizer();
  }

  /**
   * Run a trade setup through the full pipeline
   */
  evaluate(setup: TradeSetup): PipelineResult {
    console.log('\n' + '═'.repeat(60));
    console.log(`TRADE DECISION PIPELINE - ${setup.symbol} ${setup.direction}`);
    console.log('═'.repeat(60));

    const passedLayers: string[] = [];
    const totalLayers = 5;

    // ============ LAYER 1: SMC Filter ============
    console.log('\n[Layer 1/5] SMC Filter');
    const smcResult = this.checkSMC(setup);
    if (!smcResult.passed) {
      return this.createVetoResult('SMC Filter', smcResult.reason, { smcResult }, passedLayers, totalLayers);
    }
    passedLayers.push('SMC');
    console.log(`  ✅ PASSED - Score: ${smcResult.score}`);

    // ============ LAYER 2: ML Filter ============
    console.log('\n[Layer 2/5] ML Filter');
    const mlResult = this.checkML(setup);
    if (!mlResult.passed) {
      return this.createVetoResult('ML Filter', mlResult.reason, { smcResult, mlResult }, passedLayers, totalLayers);
    }
    passedLayers.push('ML');
    console.log(`  ✅ PASSED - Win Prob: ${(mlResult.winProb * 100).toFixed(1)}%`);

    // ============ LAYER 3: Ensemble Vote ============
    console.log('\n[Layer 3/5] Ensemble Vote');

    // Prepare market data for ensemble
    const marketData: MarketData = {
      ...setup.marketData,
      smcScore: setup.smcScore,
      smcDirection: setup.smcDirection,
      mlWinProbability: setup.mlWinProbability,
      mlDirection: setup.mlDirection,
    };

    const ensembleResult = this.ensemble.vote(marketData);
    if (!ensembleResult.approved) {
      return this.createVetoResult('Ensemble Vote', ensembleResult.vetoReason || 'No consensus', { smcResult, mlResult, ensembleResult }, passedLayers, totalLayers);
    }

    // Check direction alignment
    if (ensembleResult.direction !== setup.direction) {
      const reason = `Ensemble direction (${ensembleResult.direction}) conflicts with setup (${setup.direction})`;
      return this.createVetoResult('Ensemble Vote', reason, { smcResult, mlResult, ensembleResult }, passedLayers, totalLayers);
    }
    passedLayers.push('Ensemble');
    console.log(`  ✅ PASSED - Consensus: ${(ensembleResult.consensus * 100).toFixed(0)}%, Confidence: ${(ensembleResult.confidence * 100).toFixed(0)}%`);

    // ============ LAYER 4: Sentiment Check ============
    console.log('\n[Layer 4/5] Sentiment Check');
    const sentimentResult = this.checkSentiment(setup);
    if (!sentimentResult.passed) {
      return this.createVetoResult('Sentiment Check', sentimentResult.reason, { smcResult, mlResult, ensembleResult, sentimentResult }, passedLayers, totalLayers);
    }
    passedLayers.push('Sentiment');
    console.log(`  ✅ PASSED - ${sentimentResult.reason}`);

    // ============ LAYER 5: Correlation Check ============
    console.log('\n[Layer 5/5] Correlation Check');
    const correlationResult = this.checkCorrelation(setup);
    if (!correlationResult.passed) {
      return this.createVetoResult('Correlation Check', correlationResult.reason, { smcResult, mlResult, ensembleResult, sentimentResult, correlationResult }, passedLayers, totalLayers);
    }
    passedLayers.push('Correlation');
    console.log(`  ✅ PASSED - ${correlationResult.reason}`);

    // ============ ALL LAYERS PASSED - SIZE WITH RL ============
    console.log('\n[Sizing] RL Position Sizer');

    const rlState: MarketState = {
      volatility: setup.marketData.closes ? this.calculateVolatility(setup.marketData.closes) : 0.03,
      trendStrength: ensembleResult.confidence,
      recentWinRate: setup.recentWinRate ?? 0.5,
      currentDrawdown: setup.currentDrawdown ?? 0,
      openPositions: setup.openPositionCount ?? 0,
      mlConfidence: setup.mlWinProbability,
      smcScore: setup.smcScore,
    };

    const positionSize = this.rlSizer.selectAction(rlState);

    // ============ FINAL RESULT ============
    console.log('\n' + '═'.repeat(60));
    console.log('PIPELINE RESULT: ✅ APPROVED');
    console.log('═'.repeat(60));
    console.log(`  Direction: ${setup.direction}`);
    console.log(`  Position: ${positionSize.allocationPct}% × ${positionSize.leverage}x = ${positionSize.effectiveSize}% effective`);
    console.log(`  Layers Passed: ${passedLayers.join(' → ')}`);
    console.log('═'.repeat(60));

    return {
      approved: true,
      smcResult,
      mlResult,
      ensembleResult,
      sentimentResult,
      correlationResult,
      positionSize,
      passedLayers,
      totalLayers,
    };
  }

  /**
   * Layer 1: SMC Filter
   */
  private checkSMC(setup: TradeSetup): { passed: boolean; score: number; reason: string } {
    const score = setup.smcScore;

    if (score < this.config.minSMCScore) {
      return {
        passed: false,
        score,
        reason: `SMC score ${score} < minimum ${this.config.minSMCScore}`
      };
    }

    if (setup.smcDirection !== setup.direction) {
      return {
        passed: false,
        score,
        reason: `SMC direction (${setup.smcDirection}) conflicts with trade direction (${setup.direction})`
      };
    }

    return {
      passed: true,
      score,
      reason: `SMC score ${score} meets threshold`
    };
  }

  /**
   * Layer 2: ML Filter
   */
  private checkML(setup: TradeSetup): { passed: boolean; winProb: number; reason: string } {
    const winProb = setup.mlWinProbability;
    const lossProb = 1 - winProb;

    // Strong loss signal = veto
    if (lossProb > this.config.maxMLLossProbability) {
      return {
        passed: false,
        winProb,
        reason: `Loss probability ${(lossProb * 100).toFixed(0)}% > max ${(this.config.maxMLLossProbability * 100).toFixed(0)}%`
      };
    }

    // Win probability below threshold = veto
    if (winProb < this.config.minMLWinProbability) {
      return {
        passed: false,
        winProb,
        reason: `Win probability ${(winProb * 100).toFixed(0)}% < minimum ${(this.config.minMLWinProbability * 100).toFixed(0)}%`
      };
    }

    // Direction mismatch check
    if (setup.mlDirection !== setup.direction && setup.mlDirection !== 'NEUTRAL') {
      return {
        passed: false,
        winProb,
        reason: `ML direction (${setup.mlDirection}) conflicts with trade direction (${setup.direction})`
      };
    }

    return {
      passed: true,
      winProb,
      reason: `Win probability ${(winProb * 100).toFixed(0)}% acceptable`
    };
  }

  /**
   * Layer 4: Sentiment Check
   */
  private checkSentiment(setup: TradeSetup): { passed: boolean; score: number; reason: string } {
    if (!this.config.useSentiment) {
      return { passed: true, score: 0, reason: 'Sentiment check disabled' };
    }

    const score = setup.sentimentScore ?? 0;

    // Strong negative sentiment = veto
    if (score < this.config.sentimentVetoThreshold) {
      return {
        passed: false,
        score,
        reason: `Sentiment ${score.toFixed(2)} < veto threshold ${this.config.sentimentVetoThreshold}`
      };
    }

    // Check sentiment alignment with direction
    if (setup.direction === 'LONG' && score < this.config.minSentimentScore) {
      return {
        passed: false,
        score,
        reason: `Negative sentiment (${score.toFixed(2)}) for LONG trade`
      };
    }

    if (setup.direction === 'SHORT' && score > -this.config.minSentimentScore) {
      // For shorts, we actually want negative or neutral sentiment
      // This is fine - don't veto
    }

    return {
      passed: true,
      score,
      reason: score > 0.3 ? 'Positive sentiment' : score < -0.3 ? 'Negative sentiment (ok for short)' : 'Neutral sentiment'
    };
  }

  /**
   * Layer 5: Correlation Check
   */
  private checkCorrelation(setup: TradeSetup): { passed: boolean; correlations: string[]; reason: string } {
    if (!this.config.useCorrelation) {
      return { passed: true, correlations: [], reason: 'Correlation check disabled' };
    }

    const correlatedPositions = setup.correlatedPositions ?? [];
    const openCount = setup.openPositionCount ?? 0;

    // Too many open positions
    if (openCount >= this.config.maxOpenPositions) {
      return {
        passed: false,
        correlations: correlatedPositions,
        reason: `Already at max positions (${openCount}/${this.config.maxOpenPositions})`
      };
    }

    // Too many correlated positions
    if (correlatedPositions.length >= this.config.maxCorrelatedPositions) {
      return {
        passed: false,
        correlations: correlatedPositions,
        reason: `Too many correlated positions: ${correlatedPositions.join(', ')}`
      };
    }

    // Check drawdown limit
    const drawdown = setup.currentDrawdown ?? 0;
    if (drawdown > this.config.maxDrawdownToTrade) {
      return {
        passed: false,
        correlations: correlatedPositions,
        reason: `Drawdown ${(drawdown * 100).toFixed(1)}% > max ${(this.config.maxDrawdownToTrade * 100).toFixed(0)}%`
      };
    }

    return {
      passed: true,
      correlations: correlatedPositions,
      reason: correlatedPositions.length > 0
        ? `${correlatedPositions.length} correlated positions (within limit)`
        : 'No correlated positions'
    };
  }

  /**
   * Create a veto result
   */
  private createVetoResult(
    layer: string,
    reason: string,
    results: Partial<PipelineResult>,
    passedLayers: string[],
    totalLayers: number
  ): PipelineResult {
    console.log(`  ❌ VETOED - ${reason}`);
    console.log('\n' + '═'.repeat(60));
    console.log(`PIPELINE RESULT: ❌ VETOED at ${layer}`);
    console.log('═'.repeat(60));
    console.log(`  Reason: ${reason}`);
    console.log(`  Layers Passed: ${passedLayers.length > 0 ? passedLayers.join(' → ') : 'None'}`);
    console.log('═'.repeat(60));

    return {
      approved: false,
      vetoedAt: layer,
      vetoReason: reason,
      ...results,
      passedLayers,
      totalLayers,
    };
  }

  /**
   * Calculate volatility from closes
   */
  private calculateVolatility(closes: number[]): number {
    if (closes.length < 2) return 0.03;

    const returns: number[] = [];
    for (let i = 1; i < closes.length; i++) {
      returns.push((closes[i] - closes[i - 1]) / closes[i - 1]);
    }

    const mean = returns.reduce((a, b) => a + b, 0) / returns.length;
    const variance = returns.reduce((sum, r) => sum + Math.pow(r - mean, 2), 0) / returns.length;
    return Math.sqrt(variance);
  }

  /**
   * Update RL from trade outcome
   */
  updateFromTrade(
    setup: TradeSetup,
    pipelineResult: PipelineResult,
    pnlPercent: number,
    hitStopLoss: boolean,
    hitTakeProfit: boolean
  ): void {
    if (!pipelineResult.approved || !pipelineResult.positionSize) {
      return; // Only learn from trades that were actually taken
    }

    const rlState: MarketState = {
      volatility: setup.marketData.closes ? this.calculateVolatility(setup.marketData.closes) : 0.03,
      trendStrength: pipelineResult.ensembleResult?.confidence ?? 0.5,
      recentWinRate: setup.recentWinRate ?? 0.5,
      currentDrawdown: setup.currentDrawdown ?? 0,
      openPositions: setup.openPositionCount ?? 0,
      mlConfidence: setup.mlWinProbability,
      smcScore: setup.smcScore,
    };

    this.rlSizer.updateFromTrade(rlState, pipelineResult.positionSize, pnlPercent, hitStopLoss, hitTakeProfit);
  }

  /**
   * Get pipeline statistics
   */
  getStats(): { rl: ReturnType<RLPositionSizer['getStats']> } {
    return {
      rl: this.rlSizer.getStats()
    };
  }
}

// Export singleton
export const tradePipeline = new TradeDecisionPipeline();
