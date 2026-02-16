#!/usr/bin/env node
/**
 * ML Trade Advisor
 * Loads the trained H2O model and provides predictions for trade setups.
 * This is the main interface for Claude to ask "should I take this trade?"
 *
 * Usage:
 *   import { MLTradeAdvisor } from './ml-trade-advisor.js';
 *   const advisor = new MLTradeAdvisor();
 *   const result = await advisor.shouldTakeTrade(symbol, direction, marketData);
 */

import fs from 'fs';
import path from 'path';
import { TradingMLModel, Prediction } from './ml-model.js';
import { FeatureExtractor, TradeFeatures } from './trade-features.js';
import { SMCIndicators, Candle } from './smc-indicators.js';
import { UnifiedScoring } from './unified-scoring.js';
import { TradeDecisionPipeline, TradeSetup, PipelineResult, PipelineConfig } from './trade-decision-pipeline.js';
import { H2OIntegration, H2OTrainingResult } from './h2o-integration.js';

// Type for prediction input (features without outcome data)
type PredictionFeatures = Omit<TradeFeatures, 'outcome' | 'pnl' | 'pnl_percent' | 'exit_reason' | 'holding_periods'>;

export interface MarketSnapshot {
  symbol: string;
  candles: Candle[];  // Recent candles (need at least 200)
  currentPrice: number;
}

export interface TradeAdvice {
  shouldTrade: boolean;
  direction: 'LONG' | 'SHORT' | 'NEUTRAL';
  confidence: number;  // 0-1
  winProbability: number;  // 0-1

  // Scores
  smcScore: number;
  mlScore: number;
  ensembleScore: number;

  // Position sizing (if approved)
  positionSizePct?: number;
  leverage?: number;
  effectiveSize?: number;

  // Entry/Exit levels
  suggestedEntry?: number;
  suggestedStopLoss?: number;
  suggestedTakeProfit?: number;

  // Reasoning
  reasons: string[];
  warnings: string[];

  // Full pipeline result for debugging
  pipelineResult?: PipelineResult;

  // Model info
  modelId: string;
  modelAccuracy: number;
}

export class MLTradeAdvisor {
  private mlModel: TradingMLModel;
  private h2o: H2OIntegration;
  private pipeline: TradeDecisionPipeline;
  private bestModel: H2OTrainingResult | null = null;
  private modelDir: string;
  private initialized: boolean = false;

  constructor(config?: Partial<PipelineConfig> & { modelDir?: string }) {
    // Allow explicit modelDir or default to cwd-relative path
    this.modelDir = config?.modelDir || path.join(process.cwd(), 'data', 'models');
    this.mlModel = new TradingMLModel();
    this.h2o = new H2OIntegration({ modelDir: this.modelDir });
    this.pipeline = new TradeDecisionPipeline(config);
  }

  /**
   * Initialize the advisor - loads the best model
   */
  async initialize(): Promise<void> {
    if (this.initialized) return;

    console.log('[MLAdvisor] Initializing...');

    // Try to load pre-trained weights first (faster, no retraining needed)
    const weightsLoaded = await this.loadModelWeights();

    if (!weightsLoaded) {
      // Fall back to loading model metadata + CSV retraining
      this.bestModel = await this.h2o.getBestModel();

      if (this.bestModel) {
        console.log(`[MLAdvisor] Loaded model: ${this.bestModel.modelId}`);
        console.log(`[MLAdvisor] Accuracy: ${(this.bestModel.accuracy * 100).toFixed(1)}%`);

        // Load training data to initialize the local ML model
        await this.loadTrainingData();
      } else {
        console.log('[MLAdvisor] No trained model found. Run npm run learn-loop first.');
      }
    }

    this.initialized = true;
  }

  /**
   * Load pre-trained model weights (much faster than retraining from CSV)
   */
  private async loadModelWeights(): Promise<boolean> {
    const weightsFile = path.join(this.modelDir, 'model-weights.json');
    const metaFile = path.join(this.modelDir, 'best-model.json');

    if (!fs.existsSync(weightsFile)) {
      console.log('[MLAdvisor] No weights file found');
      return false;
    }

    try {
      const weights = JSON.parse(fs.readFileSync(weightsFile, 'utf-8'));
      this.mlModel.importWeights(weights);

      // Load metadata
      if (fs.existsSync(metaFile)) {
        const meta = JSON.parse(fs.readFileSync(metaFile, 'utf-8'));
        console.log(`[MLAdvisor] Loaded weights: ${meta.modelId}`);
        console.log(`[MLAdvisor] Accuracy: ${(meta.finalAccuracy * 100).toFixed(1)}%`);
        console.log(`[MLAdvisor] Weights: ${meta.numWeights}`);

        this.bestModel = {
          modelId: meta.modelId,
          accuracy: meta.finalAccuracy,
          auc: 0,
          trainedAt: meta.trainedAt
        } as any;
      }

      return true;
    } catch (err: any) {
      console.log(`[MLAdvisor] Failed to load weights: ${err.message}`);
      return false;
    }
  }

  /**
   * Load training data to initialize local ML model
   */
  private async loadTrainingData(): Promise<void> {
    // Check both old (h2o-training) and new (learning-loop) directories
    const dirs = [
      { dir: path.join(process.cwd(), 'data', 'learning-loop'), prefix: 'training_data_' },
      { dir: path.join(process.cwd(), 'data', 'h2o-training'), prefix: 'train_' },
    ];

    let trainingDir = '';
    let prefix = '';

    for (const { dir, prefix: p } of dirs) {
      if (fs.existsSync(dir)) {
        trainingDir = dir;
        prefix = p;
        break;
      }
    }

    if (!trainingDir) {
      console.log('[MLAdvisor] No training data directory found');
      return;
    }

    // Find most recent training file
    const files = fs.readdirSync(trainingDir)
      .filter(f => f.startsWith(prefix) && f.endsWith('.csv'))
      .sort()
      .reverse();

    if (files.length === 0) {
      console.log('[MLAdvisor] No training CSV files found');
      return;
    }

    const trainFile = path.join(trainingDir, files[0]);
    console.log(`[MLAdvisor] Loading training data from ${files[0]}`);

    // Parse CSV and train local model
    const trades = this.parseCSV(trainFile);
    if (trades.length > 0) {
      this.mlModel.train(trades);
      console.log(`[MLAdvisor] Local model trained on ${trades.length} trades`);
    }
  }

  /**
   * Parse CSV file into trade features
   */
  private parseCSV(filePath: string): TradeFeatures[] {
    const content = fs.readFileSync(filePath, 'utf-8');
    const lines = content.trim().split('\n');
    if (lines.length < 2) return [];

    const headers = lines[0].split(',').map(h => h.trim().replace(/"/g, ''));
    const trades: TradeFeatures[] = [];

    for (let i = 1; i < lines.length; i++) {
      const values = lines[i].split(',');
      const trade: any = {};

      headers.forEach((header, idx) => {
        let val: any = values[idx]?.trim().replace(/"/g, '');
        if (val === '0' || val === '1') val = parseInt(val);
        else if (!isNaN(parseFloat(val))) val = parseFloat(val);
        trade[header] = val;
      });

      trades.push(trade as TradeFeatures);
    }

    return trades;
  }

  /**
   * Main function: Should I take this trade?
   */
  async shouldTakeTrade(market: MarketSnapshot): Promise<TradeAdvice> {
    // Ensure initialized
    if (!this.initialized) {
      await this.initialize();
    }

    const reasons: string[] = [];
    const warnings: string[] = [];

    // Step 1: Run SMC Analysis
    const candles = market.candles;
    if (candles.length < 200) {
      return this.createNoTradeAdvice('Insufficient candle data (need 200+)', warnings);
    }

    const analysis = SMCIndicators.analyze(candles);

    // Step 2: Calculate SMC Score
    const weights = {
      trend_structure: 40,
      order_blocks: 30,
      fvgs: 20,
      ema_alignment: 15,
      liquidity: 10,
      mtf_bonus: 35,
      rsi_penalty: 15
    };

    const scoring = UnifiedScoring.calculateConfluence(analysis, market.currentPrice, weights);

    if (scoring.bias === 'neutral') {
      return this.createNoTradeAdvice('No clear directional bias from SMC', warnings);
    }

    const direction = scoring.bias === 'bullish' ? 'LONG' : 'SHORT';
    reasons.push(`SMC bias: ${scoring.bias} (score: ${scoring.score})`);

    // Step 3: Extract features for ML
    const lastIndex = candles.length - 1;
    const features = FeatureExtractor.extractFeatures(
      candles,
      lastIndex,
      analysis,
      scoring.score,
      direction === 'LONG' ? 'long' : 'short'
    );

    // Step 4: Get ML prediction - try LightGBM first (77%), fall back to local (60.5%)
    let prediction: Prediction;
    let modelSource: 'lightgbm' | 'local' = 'local';
    let modelAccuracy = this.bestModel?.accuracy || 0.605;

    const lgbmPrediction = await this.tryLightGBMPrediction(features);
    if (lgbmPrediction) {
      prediction = {
        winProbability: lgbmPrediction.winProbability,
        confidence: lgbmPrediction.confidence,
        keyFeatures: ['atr_value', 'distance_to_low', 'volatility', 'trend_strength'], // Top LightGBM features
        reason: `LightGBM prediction with ${(lgbmPrediction.modelAccuracy * 100).toFixed(0)}% accuracy`,
      };
      modelSource = 'lightgbm';
      modelAccuracy = lgbmPrediction.modelAccuracy;
      // Update best model info
      this.bestModel = {
        modelId: lgbmPrediction.modelId,
        accuracy: lgbmPrediction.modelAccuracy,
        auc: 0.86,
      } as any;
      reasons.push(`ML win probability: ${(prediction.winProbability * 100).toFixed(1)}% (LightGBM ${(modelAccuracy * 100).toFixed(0)}%)`);
    } else {
      prediction = this.mlModel.predict(features as TradeFeatures);
      reasons.push(`ML win probability: ${(prediction.winProbability * 100).toFixed(1)}% (local ${(modelAccuracy * 100).toFixed(0)}%)`);
    }

    if (prediction.winProbability < 0.45) {
      warnings.push(`ML predicts low win probability: ${(prediction.winProbability * 100).toFixed(1)}%`);
    }

    // Step 5: Calculate entry/exit levels
    const atr = analysis.atr || (candles[lastIndex].high - candles[lastIndex].low);
    const isLong = direction === 'LONG';
    const entryPrice = market.currentPrice;
    const stopLoss = isLong ? entryPrice - (atr * 2) : entryPrice + (atr * 2);
    const riskDistance = Math.abs(entryPrice - stopLoss);
    const takeProfit = isLong ? entryPrice + (riskDistance * 3) : entryPrice - (riskDistance * 3);

    // Step 6: Run through full pipeline
    const recentCandles = candles.slice(-50);
    const lastCandle = candles[candles.length - 1];
    const tradeSetup: TradeSetup = {
      symbol: market.symbol,
      direction: direction as any,
      entryPrice,
      stopLoss,
      takeProfit,
      marketData: {
        close: lastCandle.close,
        high: lastCandle.high,
        low: lastCandle.low,
        open: lastCandle.open,
        volume: lastCandle.volume,
        closes: recentCandles.map(c => c.close),
        highs: recentCandles.map(c => c.high),
        lows: recentCandles.map(c => c.low),
        volumes: recentCandles.map(c => c.volume),
      },
      smcScore: scoring.score,
      smcDirection: direction as any,
      mlWinProbability: prediction.winProbability,
      mlDirection: prediction.winProbability > 0.5 ? direction as any : 'NEUTRAL',
    };

    const pipelineResult = this.pipeline.evaluate(tradeSetup);

    // Step 7: Build final advice
    const advice: TradeAdvice = {
      shouldTrade: pipelineResult.approved,
      direction: direction as any,
      confidence: prediction.confidence,
      winProbability: prediction.winProbability,

      smcScore: scoring.score,
      mlScore: prediction.winProbability,
      ensembleScore: pipelineResult.ensembleResult?.confidence || 0,

      suggestedEntry: entryPrice,
      suggestedStopLoss: stopLoss,
      suggestedTakeProfit: takeProfit,

      reasons,
      warnings,
      pipelineResult,

      modelId: this.bestModel?.modelId || 'local-model',
      modelAccuracy: this.bestModel?.accuracy || 0,
    };

    // Add position sizing if approved
    if (pipelineResult.approved && pipelineResult.positionSize) {
      advice.positionSizePct = pipelineResult.positionSize.allocationPct;
      advice.leverage = pipelineResult.positionSize.leverage;
      advice.effectiveSize = pipelineResult.positionSize.effectiveSize;
      reasons.push(`Position size: ${advice.positionSizePct}% @ ${advice.leverage}x leverage`);
    }

    // Add veto reason if rejected
    if (!pipelineResult.approved && pipelineResult.vetoReason) {
      warnings.push(`Vetoed at ${pipelineResult.vetoedAt}: ${pipelineResult.vetoReason}`);
    }

    return advice;
  }

  /**
   * Try to get prediction from LightGBM server (77% accuracy)
   */
  private async tryLightGBMPrediction(features: any): Promise<{
    winProbability: number;
    confidence: number;
    modelId: string;
    modelAccuracy: number;
  } | null> {
    const serverUrl = process.env.LIGHTGBM_SERVER_URL || 'http://localhost:5555';

    try {
      const response = await fetch(`${serverUrl}/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features }),
      });

      if (!response.ok) {
        return null;
      }

      const result = await response.json() as any;
      if (result.error) {
        return null;
      }

      return {
        winProbability: result.win_probability,
        confidence: result.confidence,
        modelId: result.model_id || 'lightgbm',
        modelAccuracy: result.model_accuracy || 0.77,
      };
    } catch {
      return null; // Server not available
    }
  }

  /**
   * Quick check - just get ML prediction without full pipeline
   * Tries LightGBM server (77%) first, falls back to local model (60.5%)
   */
  async quickPredict(market: MarketSnapshot): Promise<{
    direction: 'LONG' | 'SHORT' | 'NEUTRAL';
    winProbability: number;
    confidence: number;
    smcScore: number;
    modelSource?: 'lightgbm' | 'local';
    modelAccuracy?: number;
  }> {
    if (!this.initialized) {
      await this.initialize();
    }

    const candles = market.candles;
    if (candles.length < 200) {
      return { direction: 'NEUTRAL', winProbability: 0.5, confidence: 0, smcScore: 0 };
    }

    const analysis = SMCIndicators.analyze(candles);
    const scoring = UnifiedScoring.calculateConfluence(analysis, market.currentPrice, {
      trend_structure: 40,
      order_blocks: 30,
      fvgs: 20,
      ema_alignment: 15,
      liquidity: 10,
      mtf_bonus: 35,
      rsi_penalty: 15
    });

    if (scoring.bias === 'neutral') {
      return { direction: 'NEUTRAL', winProbability: 0.5, confidence: 0, smcScore: scoring.score };
    }

    const direction = scoring.bias === 'bullish' ? 'LONG' : 'SHORT';
    const features = FeatureExtractor.extractFeatures(
      candles,
      candles.length - 1,
      analysis,
      scoring.score,
      direction === 'LONG' ? 'long' : 'short'
    );

    // Try LightGBM server first (77% accuracy)
    const lgbmPrediction = await this.tryLightGBMPrediction(features);
    if (lgbmPrediction) {
      // Update best model info if using LightGBM
      if (!this.bestModel || lgbmPrediction.modelAccuracy > (this.bestModel.accuracy || 0)) {
        this.bestModel = {
          modelId: lgbmPrediction.modelId,
          accuracy: lgbmPrediction.modelAccuracy,
          auc: 0.86,
        } as any;
      }

      return {
        direction: lgbmPrediction.winProbability > 0.5 ? direction : 'NEUTRAL',
        winProbability: lgbmPrediction.winProbability,
        confidence: lgbmPrediction.confidence,
        smcScore: scoring.score,
        modelSource: 'lightgbm',
        modelAccuracy: lgbmPrediction.modelAccuracy,
      };
    }

    // Fall back to local model (60.5% accuracy)
    const prediction = this.mlModel.predict(features as TradeFeatures);

    return {
      direction: prediction.winProbability > 0.5 ? direction : 'NEUTRAL',
      winProbability: prediction.winProbability,
      confidence: prediction.confidence,
      smcScore: scoring.score,
      modelSource: 'local',
      modelAccuracy: this.bestModel?.accuracy || 0.605,
    };
  }

  /**
   * Get model status
   */
  getModelStatus(): {
    hasModel: boolean;
    modelId: string | null;
    accuracy: number;
    auc: number;
    trainedAt: string | null;
  } {
    return {
      hasModel: this.bestModel !== null,
      modelId: this.bestModel?.modelId || null,
      accuracy: this.bestModel?.accuracy || 0,
      auc: this.bestModel?.auc || 0,
      trainedAt: this.bestModel?.timestamp
        ? new Date(this.bestModel.timestamp).toISOString()
        : null,
    };
  }

  /**
   * Create a "no trade" advice response
   */
  private createNoTradeAdvice(reason: string, warnings: string[]): TradeAdvice {
    return {
      shouldTrade: false,
      direction: 'NEUTRAL',
      confidence: 0,
      winProbability: 0.5,
      smcScore: 0,
      mlScore: 0.5,
      ensembleScore: 0,
      reasons: [reason],
      warnings,
      modelId: this.bestModel?.modelId || 'none',
      modelAccuracy: this.bestModel?.accuracy || 0,
    };
  }
}

// CLI for testing
async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
ML Trade Advisor - Get ML-backed trade recommendations

Usage: npx ts-node src/ml-trade-advisor.ts [OPTIONS]

Options:
  --status          Show model status
  --test <symbol>   Run a test prediction (requires historical data)
  -h, --help        Show this help

Examples:
  npx ts-node src/ml-trade-advisor.ts --status
  npx ts-node src/ml-trade-advisor.ts --test BTCUSDT
    `);
    process.exit(0);
  }

  const advisor = new MLTradeAdvisor();
  await advisor.initialize();

  if (args.includes('--status')) {
    const status = advisor.getModelStatus();
    console.log('\n=== ML Trade Advisor Status ===');
    console.log(`Has Model: ${status.hasModel ? 'Yes' : 'No'}`);
    if (status.hasModel) {
      console.log(`Model ID: ${status.modelId}`);
      console.log(`Accuracy: ${(status.accuracy * 100).toFixed(1)}%`);
      console.log(`AUC: ${(status.auc * 100).toFixed(1)}%`);
      console.log(`Trained: ${status.trainedAt}`);
    }
    return;
  }

  const testIdx = args.indexOf('--test');
  if (testIdx !== -1) {
    const symbol = args[testIdx + 1] || 'BTCUSDT';

    // Load test data
    const { LocalDataLoader } = await import('./data-loader.js');
    const loader = new LocalDataLoader(path.join(process.cwd(), 'Historical_Data_Lite'));

    try {
      const result = await loader.loadData(symbol, '1d');
      const candles = result.candles.slice(-300);  // Last 300 candles
      const currentPrice = candles[candles.length - 1].close;

      console.log(`\n=== Testing ${symbol} ===`);
      console.log(`Candles: ${candles.length}`);
      console.log(`Current Price: $${currentPrice.toFixed(2)}`);

      const advice = await advisor.shouldTakeTrade({
        symbol,
        candles,
        currentPrice,
      });

      console.log('\n=== Trade Advice ===');
      console.log(`Should Trade: ${advice.shouldTrade ? 'YES' : 'NO'}`);
      console.log(`Direction: ${advice.direction}`);
      console.log(`Win Probability: ${(advice.winProbability * 100).toFixed(1)}%`);
      console.log(`Confidence: ${(advice.confidence * 100).toFixed(1)}%`);
      console.log(`SMC Score: ${advice.smcScore}`);

      if (advice.shouldTrade) {
        console.log(`\nEntry: $${advice.suggestedEntry?.toFixed(2)}`);
        console.log(`Stop Loss: $${advice.suggestedStopLoss?.toFixed(2)}`);
        console.log(`Take Profit: $${advice.suggestedTakeProfit?.toFixed(2)}`);
        console.log(`Position: ${advice.positionSizePct}% @ ${advice.leverage}x`);
      }

      console.log('\nReasons:');
      advice.reasons.forEach(r => console.log(`  - ${r}`));

      if (advice.warnings.length > 0) {
        console.log('\nWarnings:');
        advice.warnings.forEach(w => console.log(`  - ${w}`));
      }

    } catch (error: any) {
      console.error(`Failed to load data for ${symbol}: ${error.message}`);
    }
    return;
  }

  // Default: show status
  const status = advisor.getModelStatus();
  console.log('\nML Trade Advisor ready.');
  console.log(`Model: ${status.hasModel ? status.modelId : 'No model trained'}`);
  console.log('\nUsage in code:');
  console.log('  const advisor = new MLTradeAdvisor();');
  console.log('  const advice = await advisor.shouldTakeTrade({ symbol, candles, currentPrice });');
}

main().catch(console.error);
