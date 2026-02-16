#!/usr/bin/env node
/**
 * Unified H2O Pipeline
 * Single script that:
 *   1. Extracts historical data from all symbols/timeframes
 *   2. Processes trades with SMC features
 *   3. Merges into training dataset
 *   4. Trains H2O model directly
 *   5. Saves model for production use
 *
 * Run: npx ts-node src/unified-h2o-pipeline.ts
 */

import fs from 'fs';
import path from 'path';
import { LocalDataLoader } from './data-loader.js';
import { SMCIndicators, Candle } from './smc-indicators.js';
import { UnifiedScoring } from './unified-scoring.js';
import { FeatureExtractor, TradeFeatures } from './trade-features.js';
import { H2OIntegration } from './h2o-integration.js';

// Configuration
const CONFIG = {
  symbols: [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
    'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT'
  ],
  timeframes: ['1d', '1h', '5m'],
  minScore: 70,
  dataPath: path.join(process.cwd(), 'Historical_Data_Lite'),
  outputDir: path.join(process.cwd(), 'data', 'h2o-training'),
  modelDir: path.join(process.cwd(), 'data', 'models'),
  trainSplit: 0.8,  // 80% train, 20% test
  algorithm: 'gbm' as const,  // GBM, XGBoost, DRF, GLM
};

interface PipelineResult {
  totalTrades: number;
  trainTrades: number;
  testTrades: number;
  modelId: string;
  accuracy: number;
  auc: number;
  duration: number;
}

class UnifiedH2OPipeline {
  private dataLoader: LocalDataLoader;
  private h2o: H2OIntegration;
  private allTrades: TradeFeatures[] = [];

  constructor() {
    this.dataLoader = new LocalDataLoader(CONFIG.dataPath);
    this.h2o = new H2OIntegration({
      modelDir: CONFIG.modelDir,
      dataDir: CONFIG.outputDir,
    });
  }

  /**
   * Run the complete pipeline
   */
  async run(): Promise<PipelineResult> {
    const startTime = Date.now();

    console.log('╔═══════════════════════════════════════════════════════════════╗');
    console.log('║          UNIFIED H2O TRAINING PIPELINE                        ║');
    console.log('╚═══════════════════════════════════════════════════════════════╝\n');

    console.log('Configuration:');
    console.log(`  Symbols: ${CONFIG.symbols.length}`);
    console.log(`  Timeframes: ${CONFIG.timeframes.join(', ')}`);
    console.log(`  Min Score: ${CONFIG.minScore}`);
    console.log(`  Algorithm: ${CONFIG.algorithm.toUpperCase()}`);
    console.log(`  Train/Test Split: ${CONFIG.trainSplit * 100}% / ${(1 - CONFIG.trainSplit) * 100}%`);
    console.log('');

    // Ensure directories exist
    this.ensureDirectories();

    // Phase 1: Extract trades from all symbols/timeframes
    console.log('═══════════════════════════════════════════════════════════════');
    console.log('PHASE 1: Extracting Historical Trades');
    console.log('═══════════════════════════════════════════════════════════════\n');

    await this.extractAllTrades();

    if (this.allTrades.length < 100) {
      throw new Error(`Not enough trades extracted (${this.allTrades.length}). Need at least 100.`);
    }

    console.log(`\n✅ Total trades extracted: ${this.allTrades.length}`);

    // Phase 2: Split into train/test and save CSV
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('PHASE 2: Preparing Training Data');
    console.log('═══════════════════════════════════════════════════════════════\n');

    const { trainFile, testFile, trainCount, testCount } = await this.prepareTrainingData();

    console.log(`  Train set: ${trainCount} trades`);
    console.log(`  Test set: ${testCount} trades`);
    console.log(`  Train file: ${trainFile}`);
    console.log(`  Test file: ${testFile}`);

    // Phase 3: Train H2O model
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('PHASE 3: Training H2O Model');
    console.log('═══════════════════════════════════════════════════════════════\n');

    const trainResult = await this.h2o.trainModel(
      trainFile,
      'outcome',
      CONFIG.algorithm,
      1 - CONFIG.trainSplit
    );

    // Phase 4: Evaluate on test set
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('PHASE 4: Evaluating Model');
    console.log('═══════════════════════════════════════════════════════════════\n');

    const evalResult = await this.h2o.evaluateModel(trainResult.modelId, testFile);

    // Phase 5: Set as best model if it performs well
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('PHASE 5: Saving Model');
    console.log('═══════════════════════════════════════════════════════════════\n');

    const existingBest = await this.h2o.getBestModel();

    if (!existingBest || trainResult.accuracy > existingBest.accuracy) {
      await this.h2o.setBestModel(trainResult.modelId);
      console.log(`  ✅ New best model saved: ${trainResult.modelId}`);
      console.log(`     Previous best: ${existingBest?.accuracy ? (existingBest.accuracy * 100).toFixed(1) + '%' : 'none'}`);
      console.log(`     New accuracy: ${(trainResult.accuracy * 100).toFixed(1)}%`);
    } else {
      console.log(`  ℹ️  Existing model is better (${(existingBest.accuracy * 100).toFixed(1)}% vs ${(trainResult.accuracy * 100).toFixed(1)}%)`);
    }

    // Save pipeline run metadata
    await this.savePipelineMetadata(trainResult, evalResult);

    const duration = (Date.now() - startTime) / 1000;

    // Print summary
    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log('PIPELINE COMPLETE');
    console.log('═══════════════════════════════════════════════════════════════');
    console.log(`\n  Total trades processed: ${this.allTrades.length}`);
    console.log(`  Model ID: ${trainResult.modelId}`);
    console.log(`  Accuracy: ${(trainResult.accuracy * 100).toFixed(1)}%`);
    console.log(`  AUC: ${(trainResult.auc * 100).toFixed(1)}%`);
    console.log(`  Duration: ${duration.toFixed(1)}s`);
    console.log('\n  📁 Files saved to:');
    console.log(`     ${CONFIG.outputDir}`);
    console.log(`     ${CONFIG.modelDir}`);

    return {
      totalTrades: this.allTrades.length,
      trainTrades: trainCount,
      testTrades: testCount,
      modelId: trainResult.modelId,
      accuracy: trainResult.accuracy,
      auc: trainResult.auc,
      duration,
    };
  }

  /**
   * Extract trades from all symbol/timeframe combinations
   */
  private async extractAllTrades(): Promise<void> {
    const total = CONFIG.symbols.length * CONFIG.timeframes.length;
    let processed = 0;
    let failed = 0;

    for (const timeframe of CONFIG.timeframes) {
      console.log(`\n[${timeframe}] Processing ${CONFIG.symbols.length} symbols...`);

      for (const symbol of CONFIG.symbols) {
        processed++;
        const progress = Math.floor((processed / total) * 100);

        try {
          const trades = await this.extractTradesForSymbol(symbol, timeframe);
          this.allTrades.push(...trades);
          process.stdout.write(`\r  [${progress}%] ${symbol}: ${trades.length} trades extracted`);
        } catch (err: any) {
          failed++;
          process.stdout.write(`\r  [${progress}%] ${symbol}: FAILED - ${err.message?.slice(0, 50)}`);
        }
      }
      console.log('');
    }

    console.log(`\nExtraction summary:`);
    console.log(`  Successful: ${total - failed}/${total}`);
    console.log(`  Failed: ${failed}/${total}`);
  }

  /**
   * Extract trades for a single symbol/timeframe
   */
  private async extractTradesForSymbol(symbol: string, timeframe: string): Promise<TradeFeatures[]> {
    const result = await this.dataLoader.loadData(symbol, timeframe);
    const candles = result.candles;

    if (candles.length < 300) {
      throw new Error(`Insufficient data: ${candles.length} candles`);
    }

    const trades: TradeFeatures[] = [];
    const lookback = 200;

    for (let i = lookback; i < candles.length - 50; i++) {
      const currentCandle = candles[i];
      const historicalCandles = candles.slice(0, i + 1);

      // Run SMC analysis
      const analysis = SMCIndicators.analyze(historicalCandles);

      // Calculate score
      const scoring = UnifiedScoring.calculateConfluence(
        analysis,
        currentCandle.close,
        {
          trend_structure: 40,
          order_blocks: 30,
          fvgs: 20,
          ema_alignment: 15,
          liquidity: 10,
          mtf_bonus: 35,
          rsi_penalty: 15
        }
      );

      // Skip low score trades
      if (scoring.score < CONFIG.minScore) continue;
      if (scoring.bias === 'neutral') continue;

      const direction = scoring.bias === 'bullish' ? 'long' : 'short';

      // Extract features
      const features = FeatureExtractor.extractFeatures(
        candles,
        i,
        analysis,
        scoring.score,
        direction
      );

      // Simulate trade outcome
      const trade = this.simulateTrade(candles, i, currentCandle.close, direction, analysis);
      const featuresWithOutcome = FeatureExtractor.addOutcome(features, trade);

      trades.push(featuresWithOutcome);
    }

    return trades;
  }

  /**
   * Simulate a trade to get outcome
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
    const stopLoss = isLong ? entryPrice - (atr * 2) : entryPrice + (atr * 2);
    const riskDistance = Math.abs(entryPrice - stopLoss);
    const tp2 = isLong ? entryPrice + (riskDistance * 3) : entryPrice - (riskDistance * 3);

    let exitPrice = candles[startIndex].close;
    let pnl = 0;

    // Simulate forward (max 100 candles)
    for (let i = startIndex + 1; i < Math.min(startIndex + 100, candles.length); i++) {
      const candle = candles[i];

      // Check SL
      if (isLong && candle.low <= stopLoss) {
        exitPrice = stopLoss;
        break;
      }
      if (!isLong && candle.high >= stopLoss) {
        exitPrice = stopLoss;
        break;
      }

      // Check TP2
      if (isLong && candle.high >= tp2) {
        exitPrice = tp2;
        break;
      }
      if (!isLong && candle.low <= tp2) {
        exitPrice = tp2;
        break;
      }
    }

    const priceDiff = isLong ? (exitPrice - entryPrice) : (entryPrice - exitPrice);
    pnl = 1000 * (priceDiff / entryPrice);

    return {
      pnl,
      outcome: pnl > 0 ? 'WIN' : 'LOSS'
    };
  }

  /**
   * Prepare training data - split and save to CSV
   */
  private async prepareTrainingData(): Promise<{
    trainFile: string;
    testFile: string;
    trainCount: number;
    testCount: number;
  }> {
    // Shuffle trades for random split
    const shuffled = [...this.allTrades].sort(() => Math.random() - 0.5);

    const splitIndex = Math.floor(shuffled.length * CONFIG.trainSplit);
    const trainTrades = shuffled.slice(0, splitIndex);
    const testTrades = shuffled.slice(splitIndex);

    const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
    const trainFile = path.join(CONFIG.outputDir, `train_${timestamp}.csv`);
    const testFile = path.join(CONFIG.outputDir, `test_${timestamp}.csv`);

    // Export to CSV
    this.exportToCSV(trainTrades, trainFile);
    this.exportToCSV(testTrades, testFile);

    return {
      trainFile,
      testFile,
      trainCount: trainTrades.length,
      testCount: testTrades.length,
    };
  }

  /**
   * Export trades to CSV
   */
  private exportToCSV(trades: TradeFeatures[], outputPath: string): void {
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

    const csv = [headers, ...rows].join('\n');
    fs.writeFileSync(outputPath, csv);
  }

  /**
   * Save pipeline metadata
   */
  private async savePipelineMetadata(trainResult: any, evalResult: any): Promise<void> {
    const metadataFile = path.join(CONFIG.modelDir, 'pipeline_history.json');

    let history: any[] = [];
    if (fs.existsSync(metadataFile)) {
      try {
        history = JSON.parse(fs.readFileSync(metadataFile, 'utf-8'));
      } catch {
        history = [];
      }
    }

    history.push({
      timestamp: new Date().toISOString(),
      modelId: trainResult.modelId,
      totalTrades: this.allTrades.length,
      symbols: CONFIG.symbols.length,
      timeframes: CONFIG.timeframes,
      algorithm: trainResult.algorithm,
      accuracy: trainResult.accuracy,
      auc: trainResult.auc,
      logloss: trainResult.logloss,
      evaluation: evalResult,
    });

    // Keep last 20 runs
    if (history.length > 20) {
      history = history.slice(-20);
    }

    fs.writeFileSync(metadataFile, JSON.stringify(history, null, 2));
  }

  /**
   * Ensure output directories exist
   */
  private ensureDirectories(): void {
    [CONFIG.outputDir, CONFIG.modelDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });
  }
}

// CLI
async function main() {
  const args = process.argv.slice(2);

  // Parse args
  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--symbols':
        CONFIG.symbols = args[++i].split(',');
        break;
      case '--timeframes':
        CONFIG.timeframes = args[++i].split(',');
        break;
      case '--min-score':
        CONFIG.minScore = parseInt(args[++i]);
        break;
      case '--algorithm':
        CONFIG.algorithm = args[++i] as any;
        break;
      case '--help':
      case '-h':
        console.log(`
Unified H2O Training Pipeline
Extracts historical data → Trains H2O model → Saves for production

Usage: npx ts-node src/unified-h2o-pipeline.ts [OPTIONS]

Options:
  --symbols <list>       Comma-separated symbols (default: top 10)
  --timeframes <list>    Comma-separated timeframes (default: 1d,1h,5m)
  --min-score <n>        Minimum SMC score (default: 70)
  --algorithm <algo>     H2O algorithm: gbm, xgboost, drf, glm (default: gbm)
  -h, --help             Show this help

Examples:
  npx ts-node src/unified-h2o-pipeline.ts
  npx ts-node src/unified-h2o-pipeline.ts --algorithm xgboost
  npx ts-node src/unified-h2o-pipeline.ts --symbols BTCUSDT,ETHUSDT --timeframes 1d
        `);
        process.exit(0);
    }
  }

  try {
    const pipeline = new UnifiedH2OPipeline();
    const result = await pipeline.run();

    console.log('\n✅ Pipeline completed successfully!');
    console.log('\nNext steps:');
    console.log('  1. Start H2O learning service: node dist/h2o-learning-service.js');
    console.log('  2. Check model: cat data/models/best-model.json');
    console.log('  3. Use in trading: H2OIntegration.predict(modelId, features)');

  } catch (error) {
    console.error('\n❌ Pipeline failed:', error);
    process.exit(1);
  }
}

main().catch(console.error);
