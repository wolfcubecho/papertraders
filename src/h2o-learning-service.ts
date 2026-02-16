#!/usr/bin/env node
/**
 * H2O Learning Service
 * Runs independently in the background, continuously learning from trades
 *
 * Features:
 * - Monitors for new trade data
 * - Periodically retrains models
 * - Saves models for Claude to use
 * - Runs 24/7 without Claude
 */

import fs from 'fs';
import path from 'path';
import { TradingMLModel } from './ml-model.js';

interface LearningConfig {
  // Retrain triggers
  retrainIntervalMs: number;      // Time-based retrain (default: 24h)
  minNewTrades: number;           // Data-based retrain (default: 50)
  performanceDropThreshold: number; // Performance-based (default: 0.1 = 10%)

  // Data paths
  trainingDataDir: string;
  modelOutputDir: string;
  tradeRecordsDir: string;

  // Symbols and timeframes to monitor
  symbols: string[];
  timeframes: string[];
}

const DEFAULT_CONFIG: LearningConfig = {
  retrainIntervalMs: 24 * 60 * 60 * 1000, // 24 hours
  minNewTrades: 50,
  performanceDropThreshold: 0.1,
  trainingDataDir: path.join(process.cwd(), 'data', 'h2o-training'),
  modelOutputDir: path.join(process.cwd(), 'data', 'models'),
  tradeRecordsDir: path.join(process.cwd(), 'data', 'recorded-trades'),
  symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'XRPUSDT', 'SOLUSDT', 'ADAUSDT', 'DOGEUSDT', 'AVAXUSDT', 'DOTUSDT', 'LINKUSDT'],
  timeframes: ['1d']
};

interface ModelState {
  modelId: string;
  trainedAt: number;
  tradesUsed: number;
  accuracy: number;
  winRate: number;
  version: number;
}

class H2OLearningService {
  private config: LearningConfig;
  private model: TradingMLModel;
  private modelState: ModelState | null = null;
  private lastRetrainTime: number = 0;
  private tradesProcessed: number = 0;
  private running: boolean = false;

  constructor(config: Partial<LearningConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.model = new TradingMLModel();
    this.ensureDirectories();
    this.loadModelState();
  }

  private ensureDirectories(): void {
    [this.config.trainingDataDir, this.config.modelOutputDir, this.config.tradeRecordsDir].forEach(dir => {
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }
    });
  }

  private loadModelState(): void {
    const statePath = path.join(this.config.modelOutputDir, 'model_state.json');
    if (fs.existsSync(statePath)) {
      try {
        this.modelState = JSON.parse(fs.readFileSync(statePath, 'utf-8'));
        console.log(`[H2O] Loaded model state: v${this.modelState?.version}, accuracy: ${(this.modelState?.accuracy || 0 * 100).toFixed(1)}%`);
      } catch (e) {
        console.error('[H2O] Error loading model state:', e);
      }
    }
  }

  private saveModelState(): void {
    const statePath = path.join(this.config.modelOutputDir, 'model_state.json');
    fs.writeFileSync(statePath, JSON.stringify(this.modelState, null, 2));
  }

  /**
   * Check if retraining is needed
   */
  private shouldRetrain(): { retrain: boolean; reason: string } {
    const now = Date.now();

    // No model yet
    if (!this.modelState) {
      return { retrain: true, reason: 'No model exists' };
    }

    // Time-based
    if (now - this.lastRetrainTime > this.config.retrainIntervalMs) {
      return { retrain: true, reason: `${this.config.retrainIntervalMs / 3600000}h since last retrain` };
    }

    // Data-based - check for new trades
    const newTrades = this.countNewTrades();
    if (newTrades >= this.config.minNewTrades) {
      return { retrain: true, reason: `${newTrades} new trades available` };
    }

    return { retrain: false, reason: 'No retrain needed' };
  }

  /**
   * Count new trades since last training
   */
  private countNewTrades(): number {
    let count = 0;
    try {
      const files = fs.readdirSync(this.config.tradeRecordsDir)
        .filter(f => f.endsWith('.json'));

      for (const file of files) {
        const filePath = path.join(this.config.tradeRecordsDir, file);
        const trades = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        const newTrades = trades.filter((t: any) =>
          t.timestamp > (this.modelState?.trainedAt || 0) && t.status === 'CLOSED'
        );
        count += newTrades.length;
      }
    } catch (e) {
      // No trades yet
    }
    return count;
  }

  /**
   * Load all training data
   */
  private loadTrainingData(): any[] {
    const allTrades: any[] = [];

    // Load from CSV files
    const csvFiles = fs.readdirSync(this.config.trainingDataDir)
      .filter(f => f.endsWith('.csv'));

    for (const file of csvFiles) {
      const filePath = path.join(this.config.trainingDataDir, file);
      const content = fs.readFileSync(filePath, 'utf-8');
      const lines = content.trim().split('\n');
      const headers = lines[0].split(',');

      for (let i = 1; i < lines.length; i++) {
        const values = lines[i].split(',');
        const trade: any = {};
        headers.forEach((h, idx) => {
          let val: any = values[idx];
          if (val === '0' || val === '1') val = parseInt(val);
          else if (!isNaN(parseFloat(val))) val = parseFloat(val);
          trade[h] = val;
        });
        allTrades.push(trade);
      }
    }

    // Load from recorded trades (live trades)
    try {
      const tradeFiles = fs.readdirSync(this.config.tradeRecordsDir)
        .filter(f => f.endsWith('.json'));

      for (const file of tradeFiles) {
        const filePath = path.join(this.config.tradeRecordsDir, file);
        const trades = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        const closedTrades = trades.filter((t: any) => t.status === 'CLOSED');
        allTrades.push(...closedTrades);
      }
    } catch (e) {
      // No recorded trades yet
    }

    return allTrades;
  }

  /**
   * Train the model
   */
  private async trainModel(): Promise<void> {
    console.log('\n[H2O] === Starting Model Training ===');
    const startTime = Date.now();

    const trainingData = this.loadTrainingData();
    console.log(`[H2O] Loaded ${trainingData.length} trades for training`);

    if (trainingData.length < 50) {
      console.log('[H2O] Not enough data for training (need 50+)');
      return;
    }

    // Train the model
    this.model.train(trainingData);
    const stats = this.model.getStats();

    // Update model state
    this.modelState = {
      modelId: `model_${Date.now()}`,
      trainedAt: Date.now(),
      tradesUsed: trainingData.length,
      accuracy: stats.accuracy || 0,
      winRate: stats.winRate || 0,
      version: (this.modelState?.version || 0) + 1
    };

    this.lastRetrainTime = Date.now();
    this.saveModelState();

    // Save model
    const modelPath = path.join(this.config.modelOutputDir, 'current_model.json');
    fs.writeFileSync(modelPath, JSON.stringify({
      state: this.modelState,
      stats: stats,
      trainedAt: new Date().toISOString()
    }, null, 2));

    const duration = ((Date.now() - startTime) / 1000).toFixed(1);
    console.log(`[H2O] Training complete in ${duration}s`);
    console.log(`[H2O] Model v${this.modelState.version}: ${trainingData.length} trades, ${(this.modelState.accuracy * 100).toFixed(1)}% accuracy`);
  }

  /**
   * Main learning loop
   */
  async start(): Promise<void> {
    console.log('╔═══════════════════════════════════════════════════════════════╗');
    console.log('║         H2O LEARNING SERVICE - STARTED                        ║');
    console.log('╚═══════════════════════════════════════════════════════════════╝');
    console.log(`\nConfig:`);
    console.log(`  Retrain interval: ${this.config.retrainIntervalMs / 3600000}h`);
    console.log(`  Min new trades: ${this.config.minNewTrades}`);
    console.log(`  Symbols: ${this.config.symbols.length}`);
    console.log(`  Timeframes: ${this.config.timeframes.join(', ')}`);
    console.log(`\nData directories:`);
    console.log(`  Training: ${this.config.trainingDataDir}`);
    console.log(`  Models: ${this.config.modelOutputDir}`);
    console.log(`  Trades: ${this.config.tradeRecordsDir}`);

    this.running = true;

    // Initial training if no model
    if (!this.modelState) {
      console.log('\n[H2O] No existing model found, running initial training...');
      await this.trainModel();
    }

    // Learning loop
    const checkInterval = 5 * 60 * 1000; // Check every 5 minutes
    console.log(`\n[H2O] Learning loop started (checking every 5 min)`);
    console.log('[H2O] Press Ctrl+C to stop\n');

    while (this.running) {
      const check = this.shouldRetrain();

      if (check.retrain) {
        console.log(`[H2O] Retrain triggered: ${check.reason}`);
        await this.trainModel();
      } else {
        const nextCheck = new Date(Date.now() + checkInterval).toLocaleTimeString();
        process.stdout.write(`\r[H2O] ${new Date().toLocaleTimeString()} - Model v${this.modelState?.version || 0} OK. Next check: ${nextCheck}   `);
      }

      await this.sleep(checkInterval);
    }
  }

  private sleep(ms: number): Promise<void> {
    return new Promise(resolve => setTimeout(resolve, ms));
  }

  stop(): void {
    console.log('\n[H2O] Stopping learning service...');
    this.running = false;
    this.saveModelState();
    console.log('[H2O] Service stopped. Model state saved.');
  }
}

// CLI
async function main() {
  const args = process.argv.slice(2);

  let config: Partial<LearningConfig> = {};

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--interval':
        config.retrainIntervalMs = parseFloat(args[++i]) * 3600000; // hours to ms
        break;
      case '--min-trades':
        config.minNewTrades = parseInt(args[++i]);
        break;
      case '--help':
      case '-h':
        console.log(`
H2O Learning Service - Runs continuously in the background

Usage: node dist/h2o-learning-service.js [OPTIONS]

Options:
  --interval <hours>     Retrain interval in hours (default: 24)
  --min-trades <n>       Min new trades to trigger retrain (default: 50)
  -h, --help             Show this help

Examples:
  node dist/h2o-learning-service.js
  node dist/h2o-learning-service.js --interval 12
  node dist/h2o-learning-service.js --min-trades 100

The service will:
  1. Load existing training data
  2. Train initial model if none exists
  3. Monitor for new trades
  4. Retrain periodically or when enough new data arrives
  5. Save models for Claude to use
        `);
        process.exit(0);
    }
  }

  const service = new H2OLearningService(config);

  // Handle graceful shutdown
  process.on('SIGINT', () => {
    service.stop();
    process.exit(0);
  });
  process.on('SIGTERM', () => {
    service.stop();
    process.exit(0);
  });

  await service.start();
}

main().catch(console.error);
