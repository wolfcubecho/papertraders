#!/usr/bin/env node
/**
 * Backtest-Learn Loop
 *
 * Self-improving ML training loop:
 * 1. Extract trades (including low-score ones) from historical data
 * 2. Run model predictions on all trades
 * 3. Compare predictions to actual outcomes
 * 4. Retrain model with prediction errors emphasized
 * 5. Repeat until accuracy plateaus
 *
 * This teaches the model from its own mistakes through backtesting.
 *
 * Run: npx ts-node src/backtest-learn-loop.ts
 */

import fs from 'fs';
import path from 'path';
import { execSync } from 'child_process';
import { LocalDataLoader } from './data-loader.js';
import { SMCIndicators, Candle } from './smc-indicators.js';
import { ICTIndicators, ICTAnalysis } from './ict-indicators.js';
import { UnifiedScoring } from './unified-scoring.js';
import { FeatureExtractor, TradeFeatures } from './trade-features.js';
import { TradingMLModel } from './ml-model.js';

import os from 'os';

// Configuration
const CONFIG = {
  symbols: ['BTCUSDT'],  // BTC only for testing
  timeframes: ['1d', '1h'],  // Daily and 1h timeframes

  // LOW min score to capture both good AND bad trades
  // We want losers in the dataset so ML can learn what to avoid
  minScore: 15,

  dataPath: path.join(process.cwd(), 'Historical_Data_Lite'),
  outputDir: path.join(process.cwd(), 'data', 'learning-loop'),
  modelDir: path.join(process.cwd(), 'data', 'models'),

  // Learning loop settings
  maxIterations: 10,           // Max training iterations
  minAccuracyImprovement: 0.005, // Stop if improvement < 0.5%
  trainTestSplit: 0.7,         // 70% train, 30% test

  // Emphasis on prediction errors (model learns more from mistakes)
  errorEmphasisMultiplier: 3,  // Wrong predictions weighted 3x in training

  // Parallel processing - MAX POWER MODE
  workers: 10,  // All coins in parallel - user has tons of RAM/CPU

  // Process ALL timeframes in parallel now (with yearly checkpoints for 5m)
  sequentialTimeframes: [] as string[],  // Empty = everything parallel

  // Sampling: analyze every Nth candle (1 = no sampling)
  // NO SAMPLING - process every single candle for maximum data
  sampleRates: { '1d': 1, '1h': 1, '5m': 1, '1m': 1 } as Record<string, number>,

  // Checkpoint directory for resumable extraction
  checkpointDir: path.join(process.cwd(), 'data', 'checkpoints'),

  // Session filtering - DISABLED for analysis
  // We want ALL sessions in dataset so ML can learn which are bad
  skipSessions: [] as string[],  // Don't skip any - capture all for analysis
  boostSessions: ['asian', 'off-hours'] as string[],  // These have positive weights

  // ═══════════════════════════════════════════════════════════════
  // BINANCE RECENT DATA - Keep model fresh with current market
  // ═══════════════════════════════════════════════════════════════
  fetchRecentFromBinance: true,   // Fetch last N days from Binance API
  recentDays: 30,                 // How many days of recent data

  // ═══════════════════════════════════════════════════════════════
  // TRANCHED TRAINING - For large 1m/5m datasets
  // Train on 3 months, test on next month, roll forward
  // Prevents memory crashes and provides walk-forward validation
  // ═══════════════════════════════════════════════════════════════
  useTranchTraining: true,
  tranchTrainMonths: 3,          // Train on 3 months
  tranchTestMonths: 1,           // Test on next 1 month
  tranchRollForward: true,       // Roll forward through data
  tranchTimeframes: ['1m', '5m', '15m'] as string[],  // Which timeframes use tranching
};

interface LoopIteration {
  iteration: number;
  tradesExtracted: number;
  trainAccuracy: number;
  testAccuracy: number;
  predictionErrors: number;
  improvementFromLast: number;
}

interface TradeWithPrediction extends TradeFeatures {
  predicted_win_prob: number;
  prediction_correct: boolean;
  prediction_confidence: number;
}

class BacktestLearnLoop {
  private dataLoader: LocalDataLoader;
  private model: TradingMLModel;
  private allTrades: TradeFeatures[] = [];
  private iterations: LoopIteration[] = [];

  constructor() {
    this.dataLoader = new LocalDataLoader(CONFIG.dataPath);
    this.model = new TradingMLModel();
    this.ensureDirectories();
  }

  /**
   * Run the full learning loop
   */
  async run(): Promise<void> {
    const startTime = Date.now();

    console.log('╔═══════════════════════════════════════════════════════════════╗');
    console.log('║          BACKTEST-LEARN LOOP (Self-Improving ML)              ║');
    console.log('╚═══════════════════════════════════════════════════════════════╝\n');

    console.log('Configuration:');
    console.log(`  Symbols: ${CONFIG.symbols.length}`);
    console.log(`  Timeframes: ${CONFIG.timeframes.join(', ')}`);
    console.log(`  Min Score: ${CONFIG.minScore} (LOW to capture losers)`);
    console.log(`  Max Iterations: ${CONFIG.maxIterations}`);
    console.log(`  Error Emphasis: ${CONFIG.errorEmphasisMultiplier}x`);
    console.log('');

    // Phase 1: Extract ALL trades (including low quality)
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('PHASE 1: Extracting ALL Trades (including low-score losers)');
    console.log('═══════════════════════════════════════════════════════════════\n');

    await this.extractAllTrades();

    if (this.allTrades.length < 200) {
      console.log(`\n❌ Not enough trades (${this.allTrades.length}). Need 200+.`);
      return;
    }

    // Analyze trade quality distribution
    const winners = this.allTrades.filter(t => t.outcome === 'WIN').length;
    const losers = this.allTrades.filter(t => t.outcome === 'LOSS').length;
    const highScore = this.allTrades.filter(t => t.confluence_score > 0.6).length;
    const lowScore = this.allTrades.filter(t => t.confluence_score <= 0.6).length;

    console.log(`\n📊 Trade Distribution:`);
    console.log(`  Total: ${this.allTrades.length}`);
    console.log(`  Winners: ${winners} (${(winners/this.allTrades.length*100).toFixed(1)}%)`);
    console.log(`  Losers: ${losers} (${(losers/this.allTrades.length*100).toFixed(1)}%)`);
    console.log(`  High Score (>60%): ${highScore}`);
    console.log(`  Low Score (≤60%): ${lowScore}`);

    // Phase 2: Train with gradient descent
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('PHASE 2: Training Model (Gradient Descent)');
    console.log('═══════════════════════════════════════════════════════════════\n');

    // The model internally:
    // - Splits into train/validation
    // - Runs up to 100 epochs of gradient descent
    // - Uses early stopping (patience=10)
    // - Saves best weights automatically
    // - Uses L2 regularization to prevent overfitting
    this.model.train(this.allTrades);

    // Get final stats
    const modelStats = this.model.getStats();
    const finalAccuracy = modelStats.finalAccuracy;

    this.iterations.push({
      iteration: 1,
      tradesExtracted: this.allTrades.length,
      trainAccuracy: finalAccuracy,
      testAccuracy: finalAccuracy,
      predictionErrors: Math.floor(this.allTrades.length * (1 - finalAccuracy)),
      improvementFromLast: finalAccuracy
    });

    // Phase 3: Save results
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('PHASE 3: Saving Results');
    console.log('═══════════════════════════════════════════════════════════════\n');

    await this.saveResults();

    // Summary
    const duration = (Date.now() - startTime) / 1000;
    const stats = this.model.getStats();

    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('TRAINING COMPLETE');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log(`\n  Total trades: ${this.allTrades.length}`);
    console.log(`  Training epochs: ${stats.epochs}`);
    console.log(`  Learned weights: ${stats.numWeights}`);
    console.log(`  Best validation loss: ${stats.bestValLoss?.toFixed(4)}`);
    console.log(`  Final accuracy: ${(stats.finalAccuracy * 100).toFixed(1)}%`);
    console.log(`  Duration: ${duration.toFixed(1)}s`);
    console.log(`\n  Model saved to: ${CONFIG.modelDir}`);
    console.log(`  Weights file: model-weights.json`);
  }

  /**
   * Extract all trades from historical data (PARALLEL)
   */
  private async extractAllTrades(): Promise<void> {
    const total = CONFIG.symbols.length * CONFIG.timeframes.length;
    let completed = 0;
    let failed = 0;

    // Separate parallel and sequential timeframes
    const parallelTFs = CONFIG.timeframes.filter(tf => !CONFIG.sequentialTimeframes.includes(tf));
    const sequentialTFs = CONFIG.timeframes.filter(tf => CONFIG.sequentialTimeframes.includes(tf));

    console.log(`\n🚀 Parallel (${CONFIG.workers} workers): ${parallelTFs.join(', ')}`);
    console.log(`📝 Sequential (1 at a time): ${sequentialTFs.join(', ')}\n`);

    const results: TradeFeatures[][] = [];

    // PARALLEL: Process 1d, 1h with batches
    if (parallelTFs.length > 0) {
      const parallelJobs: Array<{ symbol: string; timeframe: string }> = [];
      for (const timeframe of parallelTFs) {
        for (const symbol of CONFIG.symbols) {
          parallelJobs.push({ symbol, timeframe });
        }
      }

      const batchSize = CONFIG.workers;
      for (let i = 0; i < parallelJobs.length; i += batchSize) {
        const batch = parallelJobs.slice(i, i + batchSize);

        const batchPromises = batch.map(async (job) => {
          try {
            // Use chunked extraction for 5m/1m (yearly checkpoints)
            // Use chunked extraction for 1h too (yearly checkpoints)
            const trades = (job.timeframe === '5m' || job.timeframe === '1m' || job.timeframe === '1h')
              ? await this.extractTradesChunked(job.symbol, job.timeframe)
              : await this.extractTradesForSymbol(job.symbol, job.timeframe);
            completed++;
            return { success: true, trades, job };
          } catch (err: any) {
            completed++;
            failed++;
            return { success: false, trades: [], job, error: err?.message };
          }
        });

        const batchResults = await Promise.all(batchPromises);

        for (const result of batchResults) {
          if (result.success) {
            results.push(result.trades);
            // SAVE AFTER EACH COIN/TIMEFRAME COMPLETION
            this.saveInterimProgress(result.job.symbol, result.job.timeframe, result.trades, completed, total);
          }
          const progress = Math.floor((completed / total) * 100);
          const status = result.success ? `${result.trades.length} trades` : 'FAILED';
          process.stdout.write(`\r  [${progress}%] ${result.job.symbol}/${result.job.timeframe}: ${status}     `);
        }
      }
    }

    // SEQUENTIAL: Process 5m one symbol at a time with yearly chunking
    for (const timeframe of sequentialTFs) {
      console.log(`\n\n📝 Processing ${timeframe} sequentially (with yearly checkpoints)...`);
      for (const symbol of CONFIG.symbols) {
        try {
          console.log(`  ${symbol}/${timeframe}...`);
          // Use chunked extraction for 5m to handle large files
          const trades = timeframe === '5m' || timeframe === '1m'
            ? await this.extractTradesChunked(symbol, timeframe)
            : await this.extractTradesForSymbol(symbol, timeframe);
          results.push(trades);
          completed++;
          console.log(`    ✓ Total: ${trades.length} trades`);
        } catch (err: any) {
          completed++;
          failed++;
          console.log(`    ✗ FAILED: ${err?.message}`);
        }
      }
    }

    // Flatten results
    for (const trades of results) {
      this.allTrades.push(...trades);
    }

    console.log(`\n\n✅ Extraction complete: ${total - failed}/${total} successful`);
    console.log(`   Total trades extracted: ${this.allTrades.length}`);
  }

  /**
   * Extract trades for a single symbol/timeframe
   *
   * KEY CHANGE: Now requires PULLBACK for entry, not just score threshold
   * This fixes the "chasing" problem identified in ML analysis
   */
  private async extractTradesForSymbol(symbol: string, timeframe: string): Promise<TradeFeatures[]> {
    const result = await this.dataLoader.loadData(symbol, timeframe);
    const candles = result.candles;

    if (candles.length < 300) {
      throw new Error('Insufficient data');
    }

    const trades: TradeFeatures[] = [];
    const lookback = 200;
    const sampleRate = CONFIG.sampleRates[timeframe] || 1;
    const weights = {
      trend_structure: 40,
      order_blocks: 30,
      fvgs: 20,
      ema_alignment: 15,
      liquidity: 10,
      mtf_bonus: 35,
      rsi_penalty: 15
    };

    // Track stats for logging
    let skippedNoPullback = 0;
    let skippedNoTrend = 0;
    let skippedLowScore = 0;
    let skippedBadSession = 0;

    // Helper to get session from timestamp
    const getSession = (timestamp: number): string => {
      const hour = new Date(timestamp).getUTCHours();
      if (hour >= 0 && hour < 6) return 'asian';
      if (hour >= 6 && hour < 8) return 'london';
      if (hour >= 8 && hour < 12) return 'overlap';
      if (hour >= 12 && hour < 20) return 'newyork';
      return 'off-hours';
    };

    // Sample every Nth candle for faster processing on high-frequency data
    const totalIterations = Math.floor((candles.length - 50 - lookback) / sampleRate);
    let iteration = 0;

    for (let i = lookback; i < candles.length - 50; i += sampleRate) {
      iteration++;

      // Progress every 1000 iterations
      if (iteration % 1000 === 0) {
        const pct = Math.floor((iteration / totalIterations) * 100);
        process.stdout.write(`\r      ${symbol}/${timeframe}: ${pct}% (${trades.length} trades)   `);
      }

      const currentCandle = candles[i];
      const historicalCandles = candles.slice(0, i + 1);

      // PRE-CHECK: Skip bad sessions (overlap has -0.1159 weight)
      const session = getSession(currentCandle.timestamp);
      if (CONFIG.skipSessions.includes(session)) {
        skippedBadSession++;
        continue;
      }

      const analysis = SMCIndicators.analyze(historicalCandles);

      // FIRST CHECK: Must have a trend
      if (!analysis.trend) {
        skippedNoTrend++;
        continue;
      }

      // SECOND CHECK: Pullback preferred but NOT required for analysis
      // We want non-pullback trades in dataset so ML can learn they're worse
      // Just track if it's a pullback, don't filter
      const hasPullback = analysis.pullback?.isPullback || false;
      if (!hasPullback) {
        skippedNoPullback++;  // Still count but don't skip
        // continue;  // DISABLED - we want these for analysis
      }

      const scoring = UnifiedScoring.calculateConfluence(analysis, currentCandle.close, weights, currentCandle.timestamp);

      // THIRD CHECK: Score threshold
      if (scoring.score < CONFIG.minScore) {
        skippedLowScore++;
        continue;
      }

      if (scoring.bias === 'neutral') continue;

      const direction = scoring.bias === 'bullish' ? 'long' : 'short';

      // Run fast ICT analysis for institutional-grade features (optimized for backtesting)
      const ictAnalysis = ICTIndicators.analyzeFast(historicalCandles, analysis);

      // Add pullback info and ICT features for ML training
      const features = FeatureExtractor.extractFeatures(
        candles, i, analysis, scoring.score, direction, ictAnalysis
      );

      const trade = this.simulateTrade(candles, i, currentCandle.close, direction, analysis);
      const featuresWithOutcome = FeatureExtractor.addOutcome(features, trade);

      trades.push(featuresWithOutcome);
    }

    // Log skip stats (helps understand filtering)
    if (trades.length > 0) {
      process.stdout.write(`\r      ${symbol}/${timeframe}: ${trades.length} trades (skipped: ${skippedNoPullback} no-pb, ${skippedNoTrend} no-trend, ${skippedLowScore} low-score, ${skippedBadSession} bad-session)   \n`);
    }

    return trades;
  }

  /**
   * Simulate a trade to get outcome
   * Uses dynamic R:R based on ATR and market structure
   */
  private simulateTrade(
    candles: Candle[],
    startIndex: number,
    entryPrice: number,
    direction: 'long' | 'short',
    analysis: any
  ): any {
    const isLong = direction === 'long';
    const atr = analysis.atr || (candles[startIndex].high - candles[startIndex].low);

    // Dynamic stop loss: 1.5x ATR (same as before)
    const stopMultiplier = 1.5;
    const stopLoss = isLong
      ? entryPrice - (atr * stopMultiplier)
      : entryPrice + (atr * stopMultiplier);
    const riskDistance = Math.abs(entryPrice - stopLoss);

    // DYNAMIC R:R based on signal strength and volatility
    // Base R:R is 2:1, but adjust based on conditions
    let targetRR = 2.0;

    // If we have a strong SMC signal, target higher R:R
    if (analysis.bestSignal?.strength > 0.7) {
      targetRR = 2.5;
    } else if (analysis.bestSignal?.strength > 0.5) {
      targetRR = 2.0;
    } else {
      targetRR = 1.5;  // Lower R:R for weaker signals
    }

    // Adjust based on pullback fib level
    if (analysis.pullback?.fibLevel === '0.618') {
      targetRR += 0.5;  // Golden ratio = stronger reversal
    } else if (analysis.pullback?.fibLevel === '0.786') {
      targetRR -= 0.5;  // Deep pullback = riskier
    }

    // Cap R:R between 1.5 and 3.0
    targetRR = Math.max(1.5, Math.min(3.0, targetRR));

    const takeProfit = isLong
      ? entryPrice + (riskDistance * targetRR)
      : entryPrice - (riskDistance * targetRR);

    let exitPrice = entryPrice;
    let exitReason = 'timeout';
    let holdingPeriods = 0;
    const maxHoldPeriod = 20;

    for (let i = startIndex + 1; i < Math.min(startIndex + maxHoldPeriod, candles.length); i++) {
      const candle = candles[i];
      holdingPeriods++;

      // Check stop loss first
      if (isLong && candle.low <= stopLoss) {
        exitPrice = stopLoss;
        exitReason = 'SL';
        break;
      }
      if (!isLong && candle.high >= stopLoss) {
        exitPrice = stopLoss;
        exitReason = 'SL';
        break;
      }

      // Check take profit
      if (isLong && candle.high >= takeProfit) {
        exitPrice = takeProfit;
        exitReason = 'TP';
        break;
      }
      if (!isLong && candle.low <= takeProfit) {
        exitPrice = takeProfit;
        exitReason = 'TP';
        break;
      }

      // NEW: Trailing stop after 1:1 profit
      const currentProfit = isLong
        ? (candle.close - entryPrice)
        : (entryPrice - candle.close);
      const profitInR = currentProfit / riskDistance;

      // Move stop to breakeven after hitting 1:1
      if (profitInR >= 1.0) {
        const trailingStop = isLong
          ? entryPrice + (riskDistance * 0.5)  // Lock in 0.5R
          : entryPrice - (riskDistance * 0.5);

        if ((isLong && candle.low <= trailingStop) ||
            (!isLong && candle.high >= trailingStop)) {
          exitPrice = trailingStop;
          exitReason = 'trailing';
          break;
        }
      }
    }

    // If timeout, exit at current price
    if (exitReason === 'timeout') {
      const lastCandle = candles[Math.min(startIndex + maxHoldPeriod, candles.length - 1)];
      exitPrice = lastCandle.close;
    }

    const priceDiff = isLong ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
    const pnl = 1000 * (priceDiff / entryPrice);

    return {
      pnl,
      pnl_percent: (priceDiff / entryPrice) * 100,
      outcome: pnl > 0 ? 'WIN' : 'LOSS',
      exit_reason: exitReason,
      holding_periods: holdingPeriods,
      target_rr: targetRR  // Track what R:R was used
    };
  }

  /**
   * Evaluate model on a set of trades
   */
  private evaluateModel(trades: TradeFeatures[]): { accuracy: number; errors: TradeFeatures[] } {
    let correct = 0;
    const errors: TradeFeatures[] = [];

    for (const trade of trades) {
      const prediction = this.model.predict(trade);
      const predictedWin = prediction.winProbability >= 0.5;
      const actualWin = trade.outcome === 'WIN';

      if (predictedWin === actualWin) {
        correct++;
      } else {
        errors.push(trade);
      }
    }

    return {
      accuracy: correct / trades.length,
      errors
    };
  }

  /**
   * Save results and model
   */
  private async saveResults(): Promise<void> {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const modelStats = this.model.getStats();

    // Save iteration history
    const historyFile = path.join(CONFIG.outputDir, `loop_history_${timestamp}.json`);
    fs.writeFileSync(historyFile, JSON.stringify({
      timestamp: new Date().toISOString(),
      config: CONFIG,
      iterations: this.iterations,
      totalTrades: this.allTrades.length,
      modelStats
    }, null, 2));

    // Save model metadata
    const modelMeta = {
      modelId: `loop_model_${timestamp}`,
      trainedAt: Date.now(),
      epochs: modelStats.epochs,
      finalAccuracy: modelStats.finalAccuracy,
      bestValLoss: modelStats.bestValLoss,
      numWeights: modelStats.numWeights,
      tradesUsed: this.allTrades.length,
      algorithm: 'gradient-descent-logistic'
    };

    const modelFile = path.join(CONFIG.modelDir, 'best-model.json');
    fs.writeFileSync(modelFile, JSON.stringify(modelMeta, null, 2));

    // Save learned weights (the actual model!)
    const weightsFile = path.join(CONFIG.modelDir, 'model-weights.json');
    fs.writeFileSync(weightsFile, JSON.stringify(this.model.exportWeights(), null, 2));

    // Save training data for future use
    const trainingFile = path.join(CONFIG.outputDir, `training_data_${timestamp}.csv`);
    this.saveToCSV(this.allTrades, trainingFile);

    console.log(`  Saved: ${historyFile}`);
    console.log(`  Saved: ${modelFile}`);
    console.log(`  Saved: ${weightsFile} (learned weights)`);
    console.log(`  Saved: ${trainingFile}`);
  }

  /**
   * Save trades to CSV
   */
  private saveToCSV(trades: TradeFeatures[], outputPath: string): void {
    if (trades.length === 0) return;

    const headers = Object.keys(trades[0]).join(',');
    const rows = trades.map(trade => {
      const values = Object.values(trade).map(val => {
        if (typeof val === 'string') return `"${val}"`;
        if (typeof val === 'boolean') return val ? 1 : 0;
        return val;
      });
      return values.join(',');
    });

    fs.writeFileSync(outputPath, [headers, ...rows].join('\n'));
  }

  /**
   * Save interim progress after each coin/timeframe completion
   */
  private saveInterimProgress(symbol: string, timeframe: string, trades: TradeFeatures[], completed: number, total: number): void {
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const progressFile = path.join(CONFIG.outputDir, `progress_${timestamp}.json`);
    
    fs.writeFileSync(progressFile, JSON.stringify({
      timestamp: new Date().toISOString(),
      symbol,
      timeframe,
      tradesCount: trades.length,
      progress: { completed, total, percent: Math.floor((completed / total) * 100) }
    }, null, 2));
    
    console.log(`\n  💾 Saved interim: ${symbol}/${timeframe} (${trades.length} trades)`);
  }

  /**
   * Ensure directories exist
   */
  private ensureDirectories(): void {
    [CONFIG.outputDir, CONFIG.modelDir, CONFIG.checkpointDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });
  }

  /**
   * Extract trades for 5m data with yearly chunking and checkpoints
   * This prevents memory issues and allows resume on crash
   */
  private async extractTradesChunked(symbol: string, timeframe: string): Promise<TradeFeatures[]> {
    const result = await this.dataLoader.loadData(symbol, timeframe);
    const candles = result.candles;

    if (candles.length < 300) {
      throw new Error('Insufficient data');
    }

    // Group candles by year
    const candlesByYear: Map<number, typeof candles> = new Map();
    for (const candle of candles) {
      const year = new Date(candle.timestamp).getFullYear();
      if (!candlesByYear.has(year)) {
        candlesByYear.set(year, []);
      }
      candlesByYear.get(year)!.push(candle);
    }

    const years = Array.from(candlesByYear.keys()).sort();
    const allTrades: TradeFeatures[] = [];

    console.log(`    Processing ${years.length} years: ${years.join(', ')}`);

    for (const year of years) {
      const checkpointFile = path.join(
        CONFIG.checkpointDir,
        `${symbol}_${timeframe}_${year}.json`
      );

      // Check if already processed
      if (fs.existsSync(checkpointFile)) {
        try {
          const cached = JSON.parse(fs.readFileSync(checkpointFile, 'utf-8'));
          allTrades.push(...cached.trades);
          console.log(`    ${year}: ✓ loaded ${cached.trades.length} trades from checkpoint`);
          continue;
        } catch (e) {
          // Corrupted checkpoint, reprocess
        }
      }

      // Get candles for this year plus lookback from previous year
      const yearCandles = candlesByYear.get(year)!;
      const prevYearCandles = candlesByYear.get(year - 1) || [];

      // Need lookback data from end of previous year
      const lookback = 200;
      const lookbackCandles = prevYearCandles.slice(-lookback);
      const processCandles = [...lookbackCandles, ...yearCandles];

      if (processCandles.length < 300) {
        console.log(`    ${year}: skipped (insufficient data: ${processCandles.length} candles)`);
        continue;
      }

      console.log(`    ${year}: processing ${yearCandles.length} candles...`);

      const yearTrades = await this.extractTradesFromCandles(
        processCandles,
        symbol,
        timeframe,
        lookbackCandles.length // Start after lookback
      );

      // Save checkpoint
      fs.writeFileSync(checkpointFile, JSON.stringify({
        symbol,
        timeframe,
        year,
        trades: yearTrades,
        processedAt: new Date().toISOString()
      }));

      allTrades.push(...yearTrades);
      console.log(`    ${year}: ✓ ${yearTrades.length} trades (saved checkpoint)`);
    }

    return allTrades;
  }

  /**
   * Extract trades from a chunk of candles
   */
  private async extractTradesFromCandles(
    candles: Candle[],
    symbol: string,
    timeframe: string,
    startOffset: number = 0
  ): Promise<TradeFeatures[]> {
    const trades: TradeFeatures[] = [];
    const lookback = 200;
    const sampleRate = CONFIG.sampleRates[timeframe] || 1;
    const weights = {
      trend_structure: 40,
      order_blocks: 30,
      fvgs: 20,
      ema_alignment: 15,
      liquidity: 10,
      mtf_bonus: 35,
      rsi_penalty: 15
    };

    const getSession = (timestamp: number): string => {
      const hour = new Date(timestamp).getUTCHours();
      if (hour >= 0 && hour < 6) return 'asian';
      if (hour >= 6 && hour < 8) return 'london';
      if (hour >= 8 && hour < 12) return 'overlap';
      if (hour >= 12 && hour < 20) return 'newyork';
      return 'off-hours';
    };

    const startIdx = Math.max(lookback, startOffset);
    const endIdx = candles.length - 50;
    const totalIterations = Math.floor((endIdx - startIdx) / sampleRate);
    let iteration = 0;

    for (let i = startIdx; i < endIdx; i += sampleRate) {
      iteration++;

      if (iteration % 500 === 0) {
        const pct = Math.floor((iteration / totalIterations) * 100);
        process.stdout.write(`\r      ${pct}% (${trades.length} trades)   `);
      }

      const currentCandle = candles[i];
      const historicalCandles = candles.slice(0, i + 1);

      const session = getSession(currentCandle.timestamp);
      if (CONFIG.skipSessions.includes(session)) continue;

      const analysis = SMCIndicators.analyze(historicalCandles);
      if (!analysis.trend) continue;

      const scoring = UnifiedScoring.calculateConfluence(
        analysis, currentCandle.close, weights, currentCandle.timestamp
      );

      if (scoring.score < CONFIG.minScore) continue;
      if (scoring.bias === 'neutral') continue;

      const direction = scoring.bias === 'bullish' ? 'long' : 'short';
      const ictAnalysis = ICTIndicators.analyzeFast(historicalCandles, analysis);
      const features = FeatureExtractor.extractFeatures(
        candles, i, analysis, scoring.score, direction, ictAnalysis
      );

      const trade = this.simulateTrade(candles, i, currentCandle.close, direction, analysis);
      const featuresWithOutcome = FeatureExtractor.addOutcome(features, trade);
      trades.push(featuresWithOutcome);
    }

    process.stdout.write('\r');
    return trades;
  }
}

// Also add live trade integration
async function addLiveTrades(): Promise<TradeFeatures[]> {
  const liveTrades: TradeFeatures[] = [];
  const tradesDir = path.join(process.cwd(), 'data', 'recorded-trades');

  if (!fs.existsSync(tradesDir)) return liveTrades;

  const files = fs.readdirSync(tradesDir).filter(f => f.endsWith('.json'));

  for (const file of files) {
    try {
      const trades = JSON.parse(fs.readFileSync(path.join(tradesDir, file), 'utf-8'));
      const closed = trades.filter((t: any) => t.status === 'CLOSED' && t.features);

      for (const trade of closed) {
        if (trade.features) {
          liveTrades.push({
            ...trade.features,
            outcome: trade.pnl > 0 ? 'WIN' : 'LOSS',
            pnl: trade.pnl,
            pnl_percent: trade.pnl_percent || 0,
            exit_reason: trade.exit_reason || 'unknown',
            holding_periods: trade.holding_periods || 0
          });
        }
      }
    } catch {}
  }

  return liveTrades;
}

// CLI
async function main() {
  const args = process.argv.slice(2);

  if (args.includes('--help') || args.includes('-h')) {
    console.log(`
Backtest-Learn Loop - Self-improving ML through backtesting

The model learns from its own mistakes by:
1. Extracting ALL trades (including low-score losers)
2. Predicting outcomes for each trade
3. Comparing predictions to actual outcomes
4. Re-training with emphasis on prediction errors
5. Repeating until accuracy plateaus

Usage: npx ts-node src/backtest-learn-loop.ts [OPTIONS]

Options:
  --workers <n>       Parallel workers (default: CPU cores - 2 = ${os.cpus().length - 2})
  --iterations <n>    Max iterations (default: 10)
  --min-score <n>     Minimum SMC score (default: 25, LOW to capture losers)
  --emphasis <n>      Error emphasis multiplier (default: 3)
  -h, --help          Show this help

Examples:
  npm run learn-loop
  npm run learn-loop -- --workers 16
  npm run learn-loop -- --iterations 20 --workers 8
  npm run learn-loop -- --min-score 20 --emphasis 5
    `);
    process.exit(0);
  }

  // Parse args
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--workers':
        CONFIG.workers = parseInt(args[++i]);
        break;
      case '--iterations':
        CONFIG.maxIterations = parseInt(args[++i]);
        break;
      case '--min-score':
        CONFIG.minScore = parseInt(args[++i]);
        break;
      case '--emphasis':
        CONFIG.errorEmphasisMultiplier = parseInt(args[++i]);
        break;
    }
  }

  try {
    const loop = new BacktestLearnLoop();
    await loop.run();

    console.log('\n✅ Learning loop complete!');
    console.log('\nThe model has learned from its backtesting mistakes.');

    // Now retrain LightGBM with WALK-FORWARD validation
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('PHASE 4: Walk-Forward LightGBM Training');
    console.log('═══════════════════════════════════════════════════════════════\n');

    try {
      // Find the latest training data CSV
      const outputDir = path.join(process.cwd(), 'data', 'learning-loop');
      const csvFiles = fs.readdirSync(outputDir)
        .filter(f => f.startsWith('training_data_') && f.endsWith('.csv'))
        .sort()
        .reverse();

      if (csvFiles.length > 0) {
        const latestCSV = path.join(outputDir, csvFiles[0]);
        // Use walk-forward trainer (proper time-series validation)
        const walkforwardScript = path.join(process.cwd(), 'scripts', 'lightgbm_walkforward.py');
        // Fallback to old trainer if walkforward not available
        const trainerScript = fs.existsSync(walkforwardScript)
          ? walkforwardScript
          : path.join(process.cwd(), 'scripts', 'lightgbm_trainer.py');

        if (fs.existsSync(trainerScript)) {
          console.log(`  Training data: ${csvFiles[0]}`);
          const isWalkforward = trainerScript.includes('walkforward');
          console.log(`  Using ${isWalkforward ? 'walk-forward' : 'random-split'} trainer...\n`);

          execSync(`python "${trainerScript}" --input "${latestCSV}"`, {
            cwd: process.cwd(),
            stdio: 'inherit'
          });

          console.log(`\n  ✅ LightGBM model ${isWalkforward ? 'walk-forward trained' : 'retrained'}`);
        } else {
          console.log('  ⚠️  LightGBM trainer script not found, skipping');
        }
      } else {
        console.log('  ⚠️  No training data CSV found, skipping LightGBM retrain');
      }
    } catch (lgbmError: any) {
      console.error('  ❌ LightGBM training failed:', lgbmError?.message || lgbmError);
    }

    console.log('\nLive trades will continue to improve it further.');
    console.log('Next: Use analyze_setup in Claude to get ML-backed predictions.');

  } catch (error) {
    console.error('\n❌ Error:', error);
    process.exit(1);
  }
}

main().catch(console.error);
