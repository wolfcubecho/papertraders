#!/usr/bin/env node
/**
 * Paper Trading with Live Data
 *
 * Monitors BTC in real-time, runs the ML model, and tracks paper trades.
 * No real money - just testing the model on live market conditions.
 *
 * Usage: npm run paper-trade
 */

import { default as Binance } from 'binance-api-node';
import fs from 'fs';
import path from 'path';
import { Candle, SMCIndicators } from './smc-indicators.js';
import { ICTIndicators } from './ict-indicators.js';
import { FeatureExtractor, TradeFeatures } from './trade-features.js';
import { UnifiedScoring } from './unified-scoring.js';
import { TradingMLModel } from './ml-model.js';

// Configuration
const CONFIG = {
  symbol: 'BTCUSDT',
  interval: '1h' as const,
  checkIntervalMs: 60000,  // Check every 1 minute
  minCandlesRequired: 200,

  // Model thresholds
  minWinProbability: 0.60,  // Need 60%+ to enter
  minSMCScore: 30,

  // Risk management
  virtualBalance: 10000,    // $10k starting balance
  maxPositionPct: 10,       // Max 10% per trade
  stopLossATRMultiple: 1.5,
  takeProfitRR: 2.0,        // 2:1 R:R

  // Tracking
  tradesFile: path.join(process.cwd(), 'data', 'paper-trades.json'),
};

interface PaperTrade {
  id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  entryTime: number;
  stopLoss: number;
  takeProfit: number;
  positionSize: number;
  status: 'OPEN' | 'CLOSED';
  exitPrice?: number;
  exitTime?: number;
  exitReason?: 'TP' | 'SL' | 'MANUAL';
  pnl?: number;
  pnlPercent?: number;
  mlPrediction: number;
  smcScore: number;
  ictScore: number;
}

interface TradingState {
  balance: number;
  trades: PaperTrade[];
  openTrade: PaperTrade | null;
  stats: {
    totalTrades: number;
    wins: number;
    losses: number;
    totalPnl: number;
    winRate: number;
  };
}

class PaperTrader {
  private client: ReturnType<typeof Binance>;
  private mlModel: TradingMLModel;
  private state: TradingState;
  private candles: Candle[] = [];
  private running: boolean = false;

  constructor() {
    this.client = Binance();
    this.mlModel = new TradingMLModel();
    this.state = this.loadState();
  }

  /**
   * Load saved state or create new
   */
  private loadState(): TradingState {
    if (fs.existsSync(CONFIG.tradesFile)) {
      try {
        return JSON.parse(fs.readFileSync(CONFIG.tradesFile, 'utf-8'));
      } catch (e) {
        console.log('Could not load saved state, starting fresh');
      }
    }

    return {
      balance: CONFIG.virtualBalance,
      trades: [],
      openTrade: null,
      stats: {
        totalTrades: 0,
        wins: 0,
        losses: 0,
        totalPnl: 0,
        winRate: 0,
      },
    };
  }

  /**
   * Save state to file
   */
  private saveState(): void {
    const dir = path.dirname(CONFIG.tradesFile);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    fs.writeFileSync(CONFIG.tradesFile, JSON.stringify(this.state, null, 2));
  }

  /**
   * Initialize - load model and fetch initial candles
   */
  async initialize(): Promise<void> {
    console.log('\n╔═══════════════════════════════════════════════════════════════╗');
    console.log('║           PAPER TRADING - Live Model Testing                   ║');
    console.log('╚═══════════════════════════════════════════════════════════════╝\n');

    // Load ML model weights
    const weightsFile = path.join(process.cwd(), 'data', 'models', 'model-weights.json');
    if (fs.existsSync(weightsFile)) {
      const weights = JSON.parse(fs.readFileSync(weightsFile, 'utf-8'));
      this.mlModel.importWeights(weights);
      console.log('✓ ML model loaded');
    } else {
      console.log('⚠ No model weights found. Run npm run learn-loop first.');
    }

    // Fetch historical candles
    console.log(`\nFetching ${CONFIG.symbol} ${CONFIG.interval} candles...`);
    await this.fetchCandles();
    console.log(`✓ Loaded ${this.candles.length} candles`);

    // Show current state
    console.log(`\n📊 Account Status:`);
    console.log(`  Balance: $${this.state.balance.toFixed(2)}`);
    console.log(`  Total Trades: ${this.state.stats.totalTrades}`);
    console.log(`  Win Rate: ${(this.state.stats.winRate * 100).toFixed(1)}%`);
    console.log(`  Total P&L: $${this.state.stats.totalPnl.toFixed(2)}`);

    if (this.state.openTrade) {
      console.log(`\n🔔 Open Position: ${this.state.openTrade.direction} @ $${this.state.openTrade.entryPrice}`);
    }
  }

  /**
   * Fetch candles from Binance
   */
  private async fetchCandles(): Promise<void> {
    const klines = await this.client.candles({
      symbol: CONFIG.symbol,
      interval: CONFIG.interval,
      limit: 500,
    });

    this.candles = klines.map(k => ({
      timestamp: k.openTime,
      open: parseFloat(k.open),
      high: parseFloat(k.high),
      low: parseFloat(k.low),
      close: parseFloat(k.close),
      volume: parseFloat(k.volume),
    }));
  }

  /**
   * Run the trading loop
   */
  async run(): Promise<void> {
    this.running = true;
    console.log('\n🚀 Starting paper trading loop...');
    console.log(`   Checking every ${CONFIG.checkIntervalMs / 1000}s`);
    console.log('   Press Ctrl+C to stop\n');

    while (this.running) {
      try {
        await this.tick();
      } catch (error: any) {
        console.error(`Error in tick: ${error.message}`);
      }

      // Wait for next check
      await new Promise(resolve => setTimeout(resolve, CONFIG.checkIntervalMs));
    }
  }

  /**
   * Single trading tick
   */
  private async tick(): Promise<void> {
    // Update candles
    await this.fetchCandles();

    const currentPrice = this.candles[this.candles.length - 1].close;
    const timestamp = new Date().toLocaleTimeString();

    // Check if we have an open trade
    if (this.state.openTrade) {
      await this.checkOpenTrade(currentPrice);
      return;
    }

    // Run analysis
    const analysis = this.analyzeMarket();

    // Log status
    process.stdout.write(`\r[${timestamp}] BTC: $${currentPrice.toFixed(2)} | SMC: ${analysis.smcScore} | ML: ${(analysis.mlPrediction * 100).toFixed(0)}% | ICT: ${analysis.ictScore} | ${analysis.direction}     `);

    // Check for entry
    if (analysis.shouldEnter) {
      await this.enterTrade(analysis, currentPrice);
    }
  }

  /**
   * Analyze current market
   */
  private analyzeMarket(): {
    shouldEnter: boolean;
    direction: 'LONG' | 'SHORT' | 'NEUTRAL';
    smcScore: number;
    mlPrediction: number;
    ictScore: number;
    reasons: string[];
  } {
    const result: {
      shouldEnter: boolean;
      direction: 'LONG' | 'SHORT' | 'NEUTRAL';
      smcScore: number;
      mlPrediction: number;
      ictScore: number;
      reasons: string[];
    } = {
      shouldEnter: false,
      direction: 'NEUTRAL',
      smcScore: 0,
      mlPrediction: 0.5,
      ictScore: 0,
      reasons: [],
    };

    if (this.candles.length < CONFIG.minCandlesRequired) {
      result.reasons.push('Insufficient candles');
      return result;
    }

    // SMC Analysis
    const smcAnalysis = SMCIndicators.analyze(this.candles);
    const currentPrice = this.candles[this.candles.length - 1].close;

    const weights = {
      trend_structure: 40,
      order_blocks: 30,
      fvgs: 20,
      ema_alignment: 15,
      liquidity: 10,
      mtf_bonus: 35,
      rsi_penalty: 15,
    };

    const scoring = UnifiedScoring.calculateConfluence(smcAnalysis, currentPrice, weights);
    result.smcScore = scoring.score;

    if (scoring.bias === 'neutral') {
      result.reasons.push('No SMC bias');
      return result;
    }

    const direction = scoring.bias === 'bullish' ? 'LONG' : 'SHORT';
    result.direction = direction;

    // ICT Analysis
    const ictAnalysis = ICTIndicators.analyzeFast(this.candles, smcAnalysis);
    result.ictScore = ictAnalysis.entryScore;

    // ML Prediction
    const features = FeatureExtractor.extractFeatures(
      this.candles,
      this.candles.length - 1,
      smcAnalysis,
      scoring.score,
      direction === 'LONG' ? 'long' : 'short',
      ictAnalysis
    );

    const prediction = this.mlModel.predict(features as TradeFeatures);
    result.mlPrediction = prediction.winProbability;

    // Check entry conditions
    if (result.mlPrediction >= CONFIG.minWinProbability &&
        result.smcScore >= CONFIG.minSMCScore) {
      result.shouldEnter = true;
      result.reasons.push(`ML: ${(result.mlPrediction * 100).toFixed(0)}%`);
      result.reasons.push(`SMC: ${result.smcScore}`);
      result.reasons.push(`ICT: ${result.ictScore}`);
    } else {
      if (result.mlPrediction < CONFIG.minWinProbability) {
        result.reasons.push(`ML too low: ${(result.mlPrediction * 100).toFixed(0)}%`);
      }
      if (result.smcScore < CONFIG.minSMCScore) {
        result.reasons.push(`SMC too low: ${result.smcScore}`);
      }
    }

    return result;
  }

  /**
   * Enter a new trade
   */
  private async enterTrade(analysis: any, currentPrice: number): Promise<void> {
    const atr = SMCIndicators.atr(this.candles, 14);
    const currentATR = atr[atr.length - 1];
    const isLong = analysis.direction === 'LONG';

    // Calculate position size
    const riskAmount = this.state.balance * (CONFIG.maxPositionPct / 100);
    const stopDistance = currentATR * CONFIG.stopLossATRMultiple;
    const positionSize = riskAmount / stopDistance;

    // Calculate levels
    const stopLoss = isLong
      ? currentPrice - stopDistance
      : currentPrice + stopDistance;
    const takeProfit = isLong
      ? currentPrice + (stopDistance * CONFIG.takeProfitRR)
      : currentPrice - (stopDistance * CONFIG.takeProfitRR);

    const trade: PaperTrade = {
      id: `PT-${Date.now()}`,
      symbol: CONFIG.symbol,
      direction: analysis.direction,
      entryPrice: currentPrice,
      entryTime: Date.now(),
      stopLoss,
      takeProfit,
      positionSize,
      status: 'OPEN',
      mlPrediction: analysis.mlPrediction,
      smcScore: analysis.smcScore,
      ictScore: analysis.ictScore,
    };

    this.state.openTrade = trade;
    this.state.trades.push(trade);
    this.saveState();

    console.log(`\n\n🔔 ENTERED ${trade.direction}!`);
    console.log(`   Entry: $${trade.entryPrice.toFixed(2)}`);
    console.log(`   SL: $${trade.stopLoss.toFixed(2)}`);
    console.log(`   TP: $${trade.takeProfit.toFixed(2)}`);
    console.log(`   ML: ${(trade.mlPrediction * 100).toFixed(0)}%`);
    console.log(`   Size: ${trade.positionSize.toFixed(4)} BTC\n`);
  }

  /**
   * Check and potentially close open trade
   */
  private async checkOpenTrade(currentPrice: number): Promise<void> {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';
    const timestamp = new Date().toLocaleTimeString();

    // Calculate unrealized P&L
    const priceDiff = isLong
      ? currentPrice - trade.entryPrice
      : trade.entryPrice - currentPrice;
    const unrealizedPnl = priceDiff * trade.positionSize;
    const pnlPercent = (priceDiff / trade.entryPrice) * 100;

    // Check stop loss
    if ((isLong && currentPrice <= trade.stopLoss) ||
        (!isLong && currentPrice >= trade.stopLoss)) {
      await this.closeTrade(trade.stopLoss, 'SL');
      return;
    }

    // Check take profit
    if ((isLong && currentPrice >= trade.takeProfit) ||
        (!isLong && currentPrice <= trade.takeProfit)) {
      await this.closeTrade(trade.takeProfit, 'TP');
      return;
    }

    // Log position status
    const pnlColor = unrealizedPnl >= 0 ? '+' : '';
    process.stdout.write(`\r[${timestamp}] ${trade.direction} @ $${trade.entryPrice.toFixed(0)} | Now: $${currentPrice.toFixed(0)} | P&L: ${pnlColor}$${unrealizedPnl.toFixed(2)} (${pnlColor}${pnlPercent.toFixed(2)}%)     `);
  }

  /**
   * Close the open trade
   */
  private async closeTrade(exitPrice: number, reason: 'TP' | 'SL' | 'MANUAL'): Promise<void> {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    const priceDiff = isLong
      ? exitPrice - trade.entryPrice
      : trade.entryPrice - exitPrice;
    const pnl = priceDiff * trade.positionSize;
    const pnlPercent = (priceDiff / trade.entryPrice) * 100;

    trade.exitPrice = exitPrice;
    trade.exitTime = Date.now();
    trade.exitReason = reason;
    trade.pnl = pnl;
    trade.pnlPercent = pnlPercent;
    trade.status = 'CLOSED';

    // Update balance
    this.state.balance += pnl;

    // Update stats
    this.state.stats.totalTrades++;
    this.state.stats.totalPnl += pnl;
    if (pnl > 0) {
      this.state.stats.wins++;
    } else {
      this.state.stats.losses++;
    }
    this.state.stats.winRate = this.state.stats.wins / this.state.stats.totalTrades;

    this.state.openTrade = null;
    this.saveState();

    const emoji = pnl > 0 ? '✅' : '❌';
    const pnlSign = pnl >= 0 ? '+' : '';

    console.log(`\n\n${emoji} CLOSED ${trade.direction} (${reason})`);
    console.log(`   Entry: $${trade.entryPrice.toFixed(2)}`);
    console.log(`   Exit: $${exitPrice.toFixed(2)}`);
    console.log(`   P&L: ${pnlSign}$${pnl.toFixed(2)} (${pnlSign}${pnlPercent.toFixed(2)}%)`);
    console.log(`\n📊 Stats: ${this.state.stats.wins}W/${this.state.stats.losses}L (${(this.state.stats.winRate * 100).toFixed(0)}%) | Balance: $${this.state.balance.toFixed(2)}\n`);
  }

  /**
   * Stop the trading loop
   */
  stop(): void {
    this.running = false;
    console.log('\n\nStopping paper trader...');
    this.saveState();
  }
}

// Main
async function main() {
  const trader = new PaperTrader();

  // Handle graceful shutdown
  process.on('SIGINT', () => {
    trader.stop();
    process.exit(0);
  });

  await trader.initialize();
  await trader.run();
}

main().catch(console.error);
