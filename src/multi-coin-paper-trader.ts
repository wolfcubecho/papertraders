#!/usr/bin/env node
/**
 * Multi-Coin Paper Trading with Live Data
 *
 * Monitors top 20 coins in real-time, runs ML model, and tracks paper trades.
 * Uses multi-timeframe analysis (15m, 1h, 4h, 1d) for better confluence.
 * No real money - just testing model on live market conditions.
 *
 * Usage: npm run paper-trade-multi
 */

import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';

const require = createRequire(import.meta.url);
const Binance = require('binance-api-node').default;
import { Candle, SMCIndicators } from './smc-indicators.js';
import { ICTIndicators } from './ict-indicators.js';
import { FeatureExtractor, TradeFeatures } from './trade-features.js';
import { UnifiedScoring } from './unified-scoring.js';
import { TradingMLModel } from './ml-model.js';
import { LightGBMPredictor } from './lightgbm-predictor.js';

// Top 20 coins
const SYMBOLS = [
  'BTCUSDT',  'ETHUSDT',
  'BNBUSDT',  'ADAUSDT',
  'SOLUSDT',  'XRPUSDT',
  'DOGEUSDT', 'DOTUSDT',
  'AVAXUSDT', 'LINKUSDT',
  'ATOMUSDT', 'NEARUSDT',
  'MATICUSDT', 'UNIUSDT',
  'LDOUSDT',  'ARBUSDT',
  'OPUSDT',   'SUIUSDT',
  'INJUSDT',  'TONUSDT',
] as const;

// Configuration
const CONFIG = {
  intervals: ['15m', '1h', '4h', '1d'] as const,
  primaryInterval: '4h' as const,
  checkIntervalMs: 30000,
  minCandlesRequired: 200,

  // Model thresholds
  minWinProbability: 0.40,
  minSMCScore: 25,
  maxSMCScore: 150,
  
  // MTF Confluence requirements
  require4HTrend: true,
  require1HAlignment: true,
  require15MAlignment: false,

  // Risk management
  virtualBalancePerCoin: 10000,
  minRiskPct: 2,      // Minimum 2% risk for low-quality trades
  maxRiskPct: 15,     // Maximum 15% risk for high-quality trades
  leverage: 1,         // 1 = spot trading, 10 = 10x leverage
  stopLossATRMultiple: 1.5,
  baseTakeProfitRR: 2.0,
  
  // Dynamic risk & TP settings based on trade quality
  qualityThresholds: {
    // SMC score thresholds for quality levels
    poor: 25,      // < 25 = poor quality
    low: 50,       // 25-50 = low quality
    medium: 75,    // 50-75 = medium quality
    high: 100,     // 75-100 = high quality
    excellent: 125 // > 125 = excellent quality
  } as const,
  mlThresholds: {
    // ML prediction thresholds for quality boost
    low: 0.45,     // < 45% = low confidence
    medium: 0.55,  // 45-55% = medium confidence
    high: 0.65,    // 55-65% = high confidence
    excellent: 0.75 // > 75% = excellent confidence
  } as const,
  qualityRiskMultipliers: {
    poor: 0.3,      // 30% of max risk = 4.5%
    low: 0.5,       // 50% of max risk = 7.5%
    medium: 0.7,    // 70% of max risk = 10.5%
    high: 0.85,     // 85% of max risk = 12.75%
    excellent: 1.0   // 100% of max risk = 15%
  } as const,
  qualityTPMultipliers: {
    poor: 1.0,      // Conservative 1.0x, 1.25x, 1.5x risk
    low: 1.25,       // 1.25x, 1.5x, 1.75x risk
    medium: 1.5,     // 1.5x, 1.75x, 2.0x risk
    high: 1.75,      // 1.75x, 2.0x, 2.25x risk
    excellent: 2.0   // Aggressive 2.0x, 2.25x, 2.5x risk
  } as const,

  // Tracking
  tradesDir: path.join(process.cwd(), 'data', 'paper-trades'),
  summaryFile: path.join(process.cwd(), 'data', 'paper-trades-summary.json'),

  // Persistence
  // State is always saved; backups are optional and throttled.
  enableStateBackups: false,
  backupIntervalMs: 12 * 60 * 60_000,
  maxBackupsPerSymbol: 24,
};

interface PaperTrade {
  id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  entryTime: number;
  stopLoss: number;
  originalStopLoss: number;
  takeProfit1: number;
  takeProfit2: number;
  takeProfit3: number;
  originalPositionSize: number;
  currentPositionSize: number;
  tp1Hit: boolean;
  tp2Hit: boolean;
  tp3Hit: boolean;
  stopLossMovedToBreakeven: boolean;
  status: 'OPEN' | 'CLOSED';
  exitPrice?: number;
  exitTime?: number;
  exitReason?: 'TP1' | 'TP2' | 'TP3' | 'SL' | 'MANUAL';
  pnl?: number;
  pnlPercent?: number;
  mlPrediction: number;
  smcScore: number;
  ictScore: number;
}

interface TimeframeData {
  interval: string;
  candles: Candle[];
  smcAnalysis: any;
  ictAnalysis: any;
  smcScore: number;
  bias: 'bullish' | 'bearish' | 'neutral';
}

interface CoinTradingState {
  symbol: string;
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
  timeframes: Map<string, TimeframeData>;
  lastCheckTime: number;
}

class CoinTrader {
  private static readonly lastBackupAtBySymbol: Map<string, number> = new Map();
  public state: CoinTradingState;
  private mlModel: TradingMLModel;
  public lgbmPredictor: LightGBMPredictor;
  public useLightGBM: boolean = false;

  constructor(symbol: typeof SYMBOLS[number]) {
    this.state = {
      symbol,
      balance: CONFIG.virtualBalancePerCoin,
      trades: [],
      openTrade: null,
      stats: {
        totalTrades: 0,
        wins: 0,
        losses: 0,
        totalPnl: 0,
        winRate: 0,
      },
      timeframes: new Map(),
      lastCheckTime: 0,
    };
    this.mlModel = new TradingMLModel();
    this.lgbmPredictor = new LightGBMPredictor();
  }

  loadState(loadedStates: Map<string, any>): void {
    const tradesFile = path.join(CONFIG.tradesDir, `${this.state.symbol}.json`);
    if (fs.existsSync(tradesFile)) {
      try {
        const savedState = JSON.parse(fs.readFileSync(tradesFile, 'utf-8'));
        
        // Add missing fields for backward compatibility
        if (savedState.openTrade) {
          if (savedState.openTrade.originalStopLoss === undefined) {
            savedState.openTrade.originalStopLoss = savedState.openTrade.stopLoss;
          }
          if (savedState.openTrade.stopLossMovedToBreakeven === undefined) {
            savedState.openTrade.stopLossMovedToBreakeven = false;
          }
          if (savedState.openTrade.currentPositionSize === undefined) {
            savedState.openTrade.currentPositionSize = savedState.openTrade.originalPositionSize || savedState.openTrade.positionSize;
          }
          if (savedState.openTrade.originalPositionSize === undefined) {
            savedState.openTrade.originalPositionSize = savedState.openTrade.currentPositionSize || savedState.openTrade.positionSize;
          }
          
          // Add missing TP fields for backward compatibility
          const trade = savedState.openTrade;
          if (trade.takeProfit1 === undefined || trade.takeProfit2 === undefined || trade.takeProfit3 === undefined) {
            const stopDistance = Math.abs(trade.entryPrice - trade.stopLoss);
            trade.takeProfit1 = trade.direction === 'LONG' 
              ? trade.entryPrice + stopDistance * 1.0
              : trade.entryPrice - stopDistance * 1.0;
            trade.takeProfit2 = trade.direction === 'LONG'
              ? trade.entryPrice + stopDistance * 1.5
              : trade.entryPrice - stopDistance * 1.5;
            trade.takeProfit3 = trade.direction === 'LONG'
              ? trade.entryPrice + stopDistance * 2.0
              : trade.entryPrice - stopDistance * 2.0;
          }
        }
        
        this.state = {
          ...this.state,
          ...savedState,
          timeframes: new Map(),
        };
        console.log(`  ✓ ${this.state.symbol}: Loaded ${this.state.stats.totalTrades} trades, Balance: $${this.state.balance.toFixed(2)}`);
      } catch (e) {
        console.log(`  ⚠ ${this.state.symbol}: Could not load state, starting fresh`);
      }
    }
  }

  saveState(): void {
    if (!fs.existsSync(CONFIG.tradesDir)) {
      fs.mkdirSync(CONFIG.tradesDir, { recursive: true });
    }
    
    const tradesFile = path.join(CONFIG.tradesDir, `${this.state.symbol}.json`);
    
    if (CONFIG.enableStateBackups && fs.existsSync(tradesFile)) {
      try {
        const now = Date.now();
        const lastBackupAt = CoinTrader.lastBackupAtBySymbol.get(this.state.symbol) ?? 0;
        if ((now - lastBackupAt) >= CONFIG.backupIntervalMs) {
          const backupDir = path.join(CONFIG.tradesDir, 'backups');
          if (!fs.existsSync(backupDir)) {
            fs.mkdirSync(backupDir, { recursive: true });
          }

          const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, -5);
          const backupFile = path.join(backupDir, `${this.state.symbol}-${timestamp}.json`);
          fs.copyFileSync(tradesFile, backupFile);
          CoinTrader.lastBackupAtBySymbol.set(this.state.symbol, now);

          const prefix = `${this.state.symbol}-`;
          const backups = fs
            .readdirSync(backupDir)
            .filter(f => f.startsWith(prefix) && f.endsWith('.json'))
            .sort()
            .reverse();
          for (const old of backups.slice(CONFIG.maxBackupsPerSymbol)) {
            try {
              fs.unlinkSync(path.join(backupDir, old));
            } catch {
              // ignore
            }
          }
        }
      } catch (e) {
        console.error(`Backup failed for ${this.state.symbol}:`, e);
      }
    }
    
    fs.writeFileSync(tradesFile, JSON.stringify(this.state, null, 2));
  }

  async initialize(client: any, modelWeights: any): Promise<void> {
    this.useLightGBM = this.lgbmPredictor.load();
    
    if (!this.useLightGBM) {
      if (modelWeights) {
        this.mlModel.importWeights(modelWeights);
      }
    }

    await this.fetchAllTimeframes(client);
  }

  async fetchAllTimeframes(client: any): Promise<void> {
    for (const interval of CONFIG.intervals) {
      await this.fetchCandles(client, interval);
    }
  }

  async fetchCandles(client: any, interval: string): Promise<void> {
    try {
      const klines = await client.candles({
        symbol: this.state.symbol,
        interval,
        limit: 500,
      });

      const candles = klines.map((k: any) => ({
        timestamp: k.openTime,
        open: parseFloat(k.open),
        high: parseFloat(k.high),
        low: parseFloat(k.low),
        close: parseFloat(k.close),
        volume: parseFloat(k.volume),
      }));

      let tfData = this.state.timeframes.get(interval);
      if (!tfData) {
        tfData = {
          interval,
          candles: [],
          smcAnalysis: null,
          ictAnalysis: null,
          smcScore: 0,
          bias: 'neutral',
        };
      }

      tfData.candles = candles;
      
      if (candles.length >= CONFIG.minCandlesRequired) {
        const smcAnalysis = SMCIndicators.analyze(candles);
        const currentPrice = candles[candles.length - 1].close;
        
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
        const ictAnalysis = ICTIndicators.analyzeFast(candles, smcAnalysis);

        tfData.smcAnalysis = smcAnalysis;
        tfData.ictAnalysis = ictAnalysis;
        tfData.smcScore = scoring.score;
        tfData.bias = scoring.bias;
      }

      this.state.timeframes.set(interval, tfData);
    } catch (error: any) {
      console.error(`Error fetching candles for ${this.state.symbol} ${interval}:`, error.message);
    }
  }

  async tick(client: any, timestamp: string): Promise<{ status: string; details: string; tfData: any }> {
    if (this.useLightGBM) {
      const updated = this.lgbmPredictor.checkForUpdates();
      if (updated) {
        console.log(`  🔄 ${this.state.symbol}: LightGBM model reloaded`);
      }
    }

    await this.fetchAllTimeframes(client);

    const primaryTf = this.state.timeframes.get(CONFIG.primaryInterval);
    if (!primaryTf || primaryTf.candles.length === 0) {
      return { status: 'ERROR', details: 'No candles', tfData: null };
    }

    const currentPrice = primaryTf.candles[primaryTf.candles.length - 1].close;

    if (this.state.openTrade) {
      const result = await this.checkOpenTrade(currentPrice);
      return {
        status: result.closed ? 'CLOSED' : 'OPEN',
        details: `${result.message} | P&L: ${result.pnlSign}$${result.pnl.toFixed(2)} (${result.pnlSign}${result.pnlPercent.toFixed(2)}%)`,
        tfData: this.getTimeframeScores(),
      };
    }

    const analysis = this.analyzeMarket();
    const status = `${analysis.direction} | ML: ${(analysis.mlPrediction * 100).toFixed(0)}% | SMC: ${analysis.smcScore}`;

    if (analysis.shouldEnter) {
      await this.enterTrade(analysis, currentPrice);
      return { status: 'ENTERED', details: status, tfData: this.getTimeframeScores() };
    }

    return { status: 'MONITOR', details: status, tfData: this.getTimeframeScores() };
  }

  private getTimeframeScores(): any {
    const scores: any = {};
    for (const [interval, data] of this.state.timeframes) {
      scores[interval] = {
        score: data.smcScore,
        bias: data.bias,
      };
    }
    return scores;
  }

  private analyzeMarket(): {
    shouldEnter: boolean;
    direction: 'LONG' | 'SHORT' | 'NEUTRAL';
    smcScore: number;
    mlPrediction: number;
    ictScore: number;
    reasons: string[];
  } {
    const result = {
      shouldEnter: false,
      direction: 'NEUTRAL' as 'LONG' | 'SHORT' | 'NEUTRAL',
      smcScore: 0,
      mlPrediction: 0.5,
      ictScore: 0,
      reasons: [] as string[],
    };

    const dailyTf = this.state.timeframes.get('1d');
    const tf4h = this.state.timeframes.get('4h');
    const tf1h = this.state.timeframes.get('1h');
    const tf15m = this.state.timeframes.get('15m');

    let dailyAligns = false;
    let dailyBias = 'neutral' as 'bullish' | 'bearish' | 'neutral';
    let dailyScore = 0;
    
    if (dailyTf && dailyTf.candles.length >= CONFIG.minCandlesRequired) {
      dailyBias = dailyTf.bias;
      dailyScore = dailyTf.smcScore;
      
      if (dailyBias !== 'neutral') {
        result.reasons.push(`1d: ${dailyBias[0].toUpperCase()}${dailyScore}`);
      }
    }

    if (!tf4h || tf4h.candles.length < CONFIG.minCandlesRequired || tf4h.bias === 'neutral') {
      result.reasons.push('No clear 4h trend');
      return result;
    }
    
    const tf4hBias = tf4h.bias;
    const tf4hScore = tf4h.smcScore;
    result.reasons.push(`4h: ${tf4hBias[0].toUpperCase()}${tf4hScore} ✓`);
    
    if (dailyBias !== 'neutral' && tf4hBias !== dailyBias) {
      result.reasons.push(`4h (${tf4hBias}) doesn't align with Daily (${dailyBias}) - waiting`);
      return result;
    }
    
    if (dailyBias !== 'neutral' && tf4hBias === dailyBias) {
      dailyAligns = true;
    }

    if (!tf1h || tf1h.candles.length < CONFIG.minCandlesRequired || tf1h.bias === 'neutral') {
      result.reasons.push('No clear 1h trend');
      return result;
    }
    
    if (tf1h.bias !== tf4hBias) {
      result.reasons.push(`1h (${tf1h.bias}) doesn't align with 4h (${tf4hBias})`);
      return result;
    }
    
    const tf1hBias = tf1h.bias;
    const tf1hScore = tf1h.smcScore;
    result.reasons.push(`1h: ${tf1hBias[0].toUpperCase()}${tf1hScore} ✓`);

    let tf15mAligns = true;
    let tf15mScore = 0;
    if (tf15m && tf15m.candles.length >= CONFIG.minCandlesRequired && tf15m.bias !== 'neutral') {
      if (CONFIG.require15MAlignment && tf15m.bias !== tf1hBias) {
        tf15mAligns = false;
        result.reasons.push(`15m (${tf15m.bias}) doesn't align with 1h (${tf1hBias})`);
        return result;
      }
      tf15mScore = tf15m.smcScore;
      result.reasons.push(`15m: ${tf15m.bias[0].toUpperCase()}${tf15mScore} ${tf15mAligns ? '✓' : '⚠'}`);
    }

    const direction = tf4hBias === 'bullish' ? 'LONG' : 'SHORT';
    result.direction = direction;

    const dailyWeight = dailyAligns ? 1.3 : 0.5;
    result.smcScore = Math.round(
      (dailyScore * dailyWeight) +
      (tf4hScore * 1.5) +
      (tf1hScore * 1.0) +
      (tf15mScore * 0.5)
    );

    const ictAnalysis = tf4h.ictAnalysis;
    result.ictScore = ictAnalysis?.entryScore || 0;

    const currentPrice = tf4h.candles[tf4h.candles.length - 1].close;
    const features = FeatureExtractor.extractFeatures(
      tf4h.candles,
      tf4h.candles.length - 1,
      tf4h.smcAnalysis,
      result.smcScore,
      direction === 'LONG' ? 'long' : 'short',
      ictAnalysis
    );

    const prediction = this.useLightGBM
      ? this.lgbmPredictor.predict(features as TradeFeatures)
      : this.mlModel.predict(features as TradeFeatures);
    result.mlPrediction = prediction.winProbability;

    // Use walk-forward optimized threshold if available, else fall back to config
    const mlApproved = this.useLightGBM && 'shouldTakeTrade' in prediction
      ? prediction.shouldTakeTrade  // Uses optimal threshold from walk-forward
      : result.mlPrediction >= CONFIG.minWinProbability;

    if (mlApproved && result.smcScore >= CONFIG.minSMCScore) {
      result.shouldEnter = true;
      const thresholdInfo = this.useLightGBM && 'threshold' in prediction
        ? (() => {
            const raw = (prediction as any).threshold;
            const threshold = (typeof raw === 'number' && Number.isFinite(raw)) ? raw : 0.5;
            return `ML: ${(result.mlPrediction * 100).toFixed(0)}% >= ${(threshold * 100).toFixed(0)}%`;
          })()
        : `ML: ${(result.mlPrediction * 100).toFixed(0)}%`;
      result.reasons.push(thresholdInfo);
      result.reasons.push(`SMC: ${result.smcScore}`);
      result.reasons.push(`ICT: ${result.ictScore}`);
      result.reasons.push('HTF aligned: Daily + 4h + 1h');
    }

    return result;
  }

  /**
   * Calculate trade quality score (0-100) based on SMC score and ML prediction
   * Higher score = better quality trade = larger position, wider TPs
   */
  private calculateTradeQuality(smcScore: number, mlPrediction: number): {
    quality: 'poor' | 'low' | 'medium' | 'high' | 'excellent';
    qualityScore: number;
    riskMultiplier: number;
    tpBaseMultiplier: number;
    riskPct: number;
  } {
    // Normalize SMC score (0-150 range) to 0-50
    const smcComponent = Math.min(smcScore / CONFIG.maxSMCScore, 1) * 50;
    
    // Normalize ML prediction (0-1) to 0-50
    const mlComponent = mlPrediction * 50;
    
    // Combined quality score (0-100)
    const qualityScore = smcComponent + mlComponent;
    
    // Determine quality level
    let quality: 'poor' | 'low' | 'medium' | 'high' | 'excellent';
    let riskMultiplier: number;
    let tpBaseMultiplier: number;
    
    if (qualityScore < 30) {
      quality = 'poor';
      riskMultiplier = CONFIG.qualityRiskMultipliers.poor;
      tpBaseMultiplier = CONFIG.qualityTPMultipliers.poor;
    } else if (qualityScore < 45) {
      quality = 'low';
      riskMultiplier = CONFIG.qualityRiskMultipliers.low;
      tpBaseMultiplier = CONFIG.qualityTPMultipliers.low;
    } else if (qualityScore < 60) {
      quality = 'medium';
      riskMultiplier = CONFIG.qualityRiskMultipliers.medium;
      tpBaseMultiplier = CONFIG.qualityTPMultipliers.medium;
    } else if (qualityScore < 80) {
      quality = 'high';
      riskMultiplier = CONFIG.qualityRiskMultipliers.high;
      tpBaseMultiplier = CONFIG.qualityTPMultipliers.high;
    } else {
      quality = 'excellent';
      riskMultiplier = CONFIG.qualityRiskMultipliers.excellent;
      tpBaseMultiplier = CONFIG.qualityTPMultipliers.excellent;
    }
    
    // Calculate actual risk percentage based on quality
    const riskPct = CONFIG.maxRiskPct * riskMultiplier;
    
    return {
      quality,
      qualityScore: Math.round(qualityScore),
      riskMultiplier,
      tpBaseMultiplier,
      riskPct
    };
  }

  private calculateTPLevels(stopDistance: number, qualityInfo: ReturnType<typeof CoinTrader.prototype.calculateTradeQuality>): {
    tp1: number;
    tp2: number;
    tp3: number;
  } {
    // Dynamic TPs based on quality
    // Poor: 1.0x, 1.25x, 1.5x risk (conservative)
    // Low: 1.25x, 1.5x, 1.75x risk
    // Medium: 1.5x, 1.75x, 2.0x risk
    // High: 1.75x, 2.0x, 2.25x risk
    // Excellent: 2.0x, 2.25x, 2.5x risk (aggressive)
    const tp1Multiplier = qualityInfo.tpBaseMultiplier;
    const tp2Multiplier = qualityInfo.tpBaseMultiplier + 0.5;
    const tp3Multiplier = qualityInfo.tpBaseMultiplier + 1.0;
    
    return {
      tp1: stopDistance * tp1Multiplier,
      tp2: stopDistance * tp2Multiplier,
      tp3: stopDistance * tp3Multiplier,
    };
  }

  private async enterTrade(analysis: any, currentPrice: number): Promise<void> {
    const primaryTf = this.state.timeframes.get(CONFIG.primaryInterval)!;
    const atr = SMCIndicators.atr(primaryTf.candles, 14);
    const currentATR = atr[atr.length - 1];
    const isLong = analysis.direction === 'LONG';

    // Calculate trade quality (0-100) based on SMC score + ML prediction
    const qualityInfo = this.calculateTradeQuality(analysis.smcScore, analysis.mlPrediction);
    
    // Dynamic risk amount based on quality
    const riskAmount = this.state.balance * (qualityInfo.riskPct / 100);
    const stopDistance = currentATR * CONFIG.stopLossATRMultiple;
    const positionSize = riskAmount / stopDistance;

    // Dynamic TPs based on quality
    const tpLevels = this.calculateTPLevels(stopDistance, qualityInfo);
    
    const stopLoss = isLong
      ? currentPrice - stopDistance
      : currentPrice + stopDistance;
    const takeProfit1 = isLong
      ? currentPrice + tpLevels.tp1
      : currentPrice - tpLevels.tp1;
    const takeProfit2 = isLong
      ? currentPrice + tpLevels.tp2
      : currentPrice - tpLevels.tp2;
    const takeProfit3 = isLong
      ? currentPrice + tpLevels.tp3
      : currentPrice - tpLevels.tp3;

    const trade: PaperTrade = {
      id: `${this.state.symbol}-${Date.now()}`,
      symbol: this.state.symbol,
      direction: analysis.direction,
      entryPrice: currentPrice,
      entryTime: Date.now(),
      stopLoss,
      originalStopLoss: stopLoss,
      takeProfit1,
      takeProfit2,
      takeProfit3,
      originalPositionSize: positionSize,
      currentPositionSize: positionSize,
      tp1Hit: false,
      tp2Hit: false,
      tp3Hit: false,
      stopLossMovedToBreakeven: false,
      status: 'OPEN',
      mlPrediction: analysis.mlPrediction,
      smcScore: analysis.smcScore,
      ictScore: analysis.ictScore,
    };

    this.state.openTrade = trade;
    this.state.trades.push(trade);
    this.saveState();

    const tfInfo = this.getTFInfo();
    const qualityEmoji = qualityInfo.quality === 'excellent' ? '⭐' : 
                       qualityInfo.quality === 'high' ? '🔥' :
                       qualityInfo.quality === 'medium' ? '✅' :
                       qualityInfo.quality === 'low' ? '⚠️' : '❌';
    
    console.log(`\n🔔 ${this.state.symbol}: ENTERED ${trade.direction}!`);
    console.log(`   Entry: $${trade.entryPrice.toFixed(2)} | SL: $${trade.stopLoss.toFixed(2)}`);
    console.log(`   TP1: $${trade.takeProfit1.toFixed(2)} | TP2: $${trade.takeProfit2.toFixed(2)} | TP3: $${trade.takeProfit3.toFixed(2)}`);
    console.log(`   Quality: ${qualityEmoji} ${qualityInfo.quality.toUpperCase()} (${qualityInfo.qualityScore}/100) | Risk: ${qualityInfo.riskPct.toFixed(1)}%`);
    console.log(`   ML: ${(trade.mlPrediction * 100).toFixed(0)}% | SMC: ${trade.smcScore} | Size: ${trade.currentPositionSize.toFixed(6)}`);
    console.log(`   MTF: ${tfInfo}\n`);
  }

  private getTFInfo(): string {
    const parts: string[] = [];
    for (const interval of CONFIG.intervals) {
      const tf = this.state.timeframes.get(interval);
      if (tf && tf.bias !== 'neutral') {
        parts.push(`${interval}:${tf.bias[0].toUpperCase()}${Math.round(tf.smcScore)}`);
      }
    }
    return parts.join(' | ') || 'No confluence';
  }

  private async checkOpenTrade(currentPrice: number): Promise<{
    closed: boolean;
    message: string;
    pnl: number;
    pnlPercent: number;
    pnlSign: string;
  }> {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    const priceDiff = isLong
      ? currentPrice - trade.entryPrice
      : trade.entryPrice - currentPrice;
    const unrealizedPnl = priceDiff * trade.currentPositionSize;
    const pnlPercent = (priceDiff / trade.entryPrice) * 100;

    // Check Stop Loss first
    if ((isLong && currentPrice <= trade.stopLoss) ||
        (!isLong && currentPrice >= trade.stopLoss)) {
      await this.closeTrade(trade.stopLoss, 'SL');
      return {
        closed: true,
        message: 'STOP LOSS',
        pnl: unrealizedPnl,
        pnlPercent,
        pnlSign: '',
      };
    }

    // Check TP1 FIRST (closest target at 1.0x risk) - close 50%
    if (!trade.tp1Hit) {
      if ((isLong && currentPrice >= trade.takeProfit1) ||
          (!isLong && currentPrice <= trade.takeProfit1)) {
        await this.closePartialTrade(currentPrice, 0.50, 'TP1');
        trade.tp1Hit = true;
        return {
          closed: false,
          message: `TP1 HIT (+50% closed)`,
          pnl: unrealizedPnl,
          pnlPercent,
          pnlSign: '+',
        };
      }
    }

    // Check TP2 NEXT (at 1.5x risk) - close 25%
    if (!trade.tp2Hit && trade.tp1Hit) {
      if ((isLong && currentPrice >= trade.takeProfit2) ||
          (!isLong && currentPrice <= trade.takeProfit2)) {
        await this.closePartialTrade(currentPrice, 0.25, 'TP2');
        trade.tp2Hit = true;
        return {
          closed: false,
          message: `TP2 HIT (+25% closed)`,
          pnl: unrealizedPnl,
          pnlPercent,
          pnlSign: '+',
        };
      }
    }

    // Check TP3 LAST (farthest target at 2.0x risk) - close remaining 25%
    if (!trade.tp3Hit && trade.tp2Hit) {
      if ((isLong && currentPrice >= trade.takeProfit3) ||
          (!isLong && currentPrice <= trade.takeProfit3)) {
        await this.closePartialTrade(currentPrice, 0.25, 'TP3');
        trade.tp3Hit = true;
        return {
          closed: false,
          message: `TP3 HIT (+25% closed)`,
          pnl: unrealizedPnl,
          pnlPercent,
          pnlSign: '+',
        };
      }
    }

    const pnlSign = unrealizedPnl >= 0 ? '+' : '';
    return {
      closed: false,
      message: `${trade.direction} @ $${trade.entryPrice.toFixed(0)} | Now: $${currentPrice.toFixed(0)}`,
      pnl: unrealizedPnl,
      pnlPercent,
      pnlSign,
    };
  }

  private async closePartialTrade(exitPrice: number, closeFraction: number, reason: 'TP1' | 'TP2' | 'TP3'): Promise<void> {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    const closeAmount = trade.originalPositionSize * closeFraction;
    trade.currentPositionSize -= closeAmount;

    const priceDiff = isLong
      ? exitPrice - trade.entryPrice
      : trade.entryPrice - exitPrice;
    const pnl = priceDiff * closeAmount;
    const pnlPercent = (priceDiff / trade.entryPrice) * 100;

    this.state.balance += pnl;

    // Create a closed trade record for the partial closure
    const closedTrade: PaperTrade = {
      id: `${this.state.symbol}-${Date.now()}-${reason}`,
      symbol: trade.symbol,
      direction: trade.direction,
      entryPrice: trade.entryPrice,
      entryTime: trade.entryTime,
      stopLoss: trade.stopLoss,
      originalStopLoss: trade.originalStopLoss,
      takeProfit1: trade.takeProfit1,
      takeProfit2: trade.takeProfit2,
      takeProfit3: trade.takeProfit3,
      originalPositionSize: closeAmount,  // Only the closed portion
      currentPositionSize: 0,
      tp1Hit: reason === 'TP1' ? true : trade.tp1Hit,
      tp2Hit: reason === 'TP2' ? true : trade.tp2Hit,
      tp3Hit: reason === 'TP3' ? true : trade.tp3Hit,
      stopLossMovedToBreakeven: trade.stopLossMovedToBreakeven,
      status: 'CLOSED',
      exitPrice,
      exitTime: Date.now(),
      exitReason: reason,
      pnl,
      pnlPercent,
      mlPrediction: trade.mlPrediction,
      smcScore: trade.smcScore,
      ictScore: trade.ictScore,
    };
    
    // Add to trades array to preserve history
    this.state.trades.push(closedTrade);

    const emoji = pnl > 0 ? '✅' : '❌';
    const pnlSign = pnl >= 0 ? '+' : '';

    console.log(`${emoji} ${trade.symbol}: ${reason} HIT!`);
    console.log(`   Entry: $${trade.entryPrice.toFixed(2)} | Exit: $${exitPrice.toFixed(2)}`);
    console.log(`   Closed ${(closeFraction * 100).toFixed(0)}% (${closeAmount.toFixed(6)}) | P&L: ${pnlSign}$${pnl.toFixed(2)} (${pnlSign}${pnlPercent.toFixed(2)}%)`);
    console.log(`   Remaining: ${trade.currentPositionSize.toFixed(6)}`);

    // Move SL to breakeven after TP1 is hit
    if (reason === 'TP1' && !trade.stopLossMovedToBreakeven) {
      trade.stopLoss = trade.entryPrice;
      trade.stopLossMovedToBreakeven = true;
      console.log(`   🛡️  SL moved to breakeven ($${trade.entryPrice.toFixed(2)})\n`);
    } else {
      console.log();
    }

    this.state.stats.totalTrades++;
    this.state.stats.totalPnl += pnl;
    if (pnl > 0) {
      this.state.stats.wins++;
    } else {
      this.state.stats.losses++;
    }
    this.state.stats.winRate = this.state.stats.wins / this.state.stats.totalTrades;

    this.saveState();
  }

  private async closeTrade(exitPrice: number, reason: 'TP1' | 'TP2' | 'TP3' | 'SL' | 'MANUAL'): Promise<void> {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    const priceDiff = isLong
      ? exitPrice - trade.entryPrice
      : trade.entryPrice - exitPrice;
    const pnl = priceDiff * trade.currentPositionSize;
    const pnlPercent = (priceDiff / trade.entryPrice) * 100;

    trade.exitPrice = exitPrice;
    trade.exitTime = Date.now();
    trade.exitReason = reason;
    trade.pnl = pnl;
    trade.pnlPercent = pnlPercent;
    trade.status = 'CLOSED';

    this.state.balance += pnl;

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

    console.log(`${emoji} ${trade.symbol}: CLOSED ${trade.direction} (${reason})`);
    console.log(`   Entry: $${trade.entryPrice.toFixed(2)} | Exit: $${exitPrice.toFixed(2)}`);
    console.log(`   P&L: ${pnlSign}$${pnl.toFixed(2)} (${pnlSign}${pnlPercent.toFixed(2)}%)\n`);
  }
}

class MultiCoinOrchestrator {
  private client: any;
  private traders: Map<string, CoinTrader> = new Map();
  private modelWeights: any = null;
  public running: boolean = false;
  private cycleCount: number = 0;

  constructor() {
    this.client = Binance();
  }

  async initialize(): Promise<void> {
    console.log('\n╔═════════════════════════════════════════════════════════╗');
    console.log('║      MULTI-COIN PAPER TRADING - Multi-Timeframe Analysis     ║');
    console.log('╚═════════════════════════════════════════════════════════╝\n');

    const weightsFile = path.join(process.cwd(), 'data', 'models', 'model-weights.json');
    if (fs.existsSync(weightsFile)) {
      this.modelWeights = JSON.parse(fs.readFileSync(weightsFile, 'utf-8'));
      console.log('✓ ML model loaded');
      console.log(`✓ Multi-Timeframe: ${CONFIG.intervals.join(', ')} (Primary: ${CONFIG.primaryInterval})\n`);
    } else {
      console.log('⚠ No model weights found. Run npm run learn-loop first.\n');
    }

    console.log('Initializing traders for all coins...\n');
    for (const symbol of SYMBOLS) {
      const trader = new CoinTrader(symbol);
      trader.loadState(new Map());
      await trader.initialize(this.client, this.modelWeights);
      this.traders.set(symbol, trader);
    }

    this.printSummary();
  }

  private printSummary(): void {
    let totalBalance = 0;
    let totalTrades = 0;
    let totalWins = 0;
    let totalLosses = 0;
    let totalPnl = 0;
    let unrealizedPnl = 0;
    let openTrades = 0;
    let modelType = 'None';
    let modelAccuracy = 'N/A';

    for (const trader of this.traders.values()) {
      totalBalance += trader.state.balance;
      totalTrades += trader.state.stats.totalTrades;
      totalWins += trader.state.stats.wins;
      totalLosses += trader.state.stats.losses;
      totalPnl += trader.state.stats.totalPnl;
      
      if (modelType === 'None') {
        if (trader.useLightGBM && trader.lgbmPredictor.isLoaded()) {
          modelType = 'LightGBM';
          const metadata = trader.lgbmPredictor.getMetadata();
          modelAccuracy = metadata?.validation_accuracy 
            ? `${(metadata.validation_accuracy * 100).toFixed(1)}%` 
            : '73%';
        } else if (this.modelWeights) {
          modelType = 'Gradient Descent';
          modelAccuracy = '59%';
        }
      }
      
      if (trader.state.openTrade) {
        openTrades++;
        const trade = trader.state.openTrade;
        const primaryTf = trader.state.timeframes.get(CONFIG.primaryInterval);
        if (primaryTf && primaryTf.candles.length > 0) {
          const currentPrice = primaryTf.candles[primaryTf.candles.length - 1].close;
          const isLong = trade.direction === 'LONG';
          const priceDiff = isLong
            ? currentPrice - trade.entryPrice
            : trade.entryPrice - currentPrice;
          unrealizedPnl += priceDiff * trade.currentPositionSize;
        }
      }
    }

    const winRate = totalTrades > 0 ? (totalWins / totalTrades) : 0;
    const startingBalance = SYMBOLS.length * CONFIG.virtualBalancePerCoin;
    const realizedPnl = totalBalance - startingBalance;
    const equity = totalBalance + unrealizedPnl;
    const totalPnlAll = equity - startingBalance;
    const realizedReturn = (realizedPnl / startingBalance) * 100;
    const totalReturn = (totalPnlAll / startingBalance) * 100;
    const realizedSign = realizedPnl >= 0 ? '+' : '';
    const unrealizedSign = unrealizedPnl >= 0 ? '+' : '';
    const totalSign = totalPnlAll >= 0 ? '+' : '';

    console.log('═════════════════════════════════════════════════════════════');
    console.log(`📊 PORTFOLIO SUMMARY`);
    console.log(`   Balance: $${totalBalance.toFixed(2)} (Started: $${startingBalance})`);
    console.log(`   Realized P&L: ${realizedSign}$${realizedPnl.toFixed(2)} (${realizedReturn >= 0 ? '+' : ''}${realizedReturn.toFixed(2)}%)`);
    if (openTrades > 0) {
      console.log(`   Unrealized: ${unrealizedSign}$${unrealizedPnl.toFixed(2)} (${openTrades} open)`);
    }
    console.log(`   Equity: $${equity.toFixed(2)} | Total P&L: ${totalSign}$${totalPnlAll.toFixed(2)} (${totalReturn >= 0 ? '+' : ''}${totalReturn.toFixed(2)}%)`);
    console.log(`   Trades: ${totalTrades} (${totalWins}W/${totalLosses}L) | Win Rate: ${(winRate * 100).toFixed(1)}%`);
    console.log(`   Open Positions: ${openTrades}`);
    console.log(`   ML Model: ${modelType} (${modelAccuracy})`);
    console.log('═════════════════════════════════════════════════════════════\n');
  }

  async run(): Promise<void> {
    this.running = true;
    console.log('🚀 Starting multi-coin paper trading loop...');
    console.log(`   Checking every ${CONFIG.checkIntervalMs / 1000}s`);
    console.log('   Press Ctrl+C to stop\n\n');

    while (this.running) {
      this.cycleCount++;
      const timestamp = new Date().toLocaleTimeString();
      const cycleStart = Date.now();

      process.stdout.write('\x1B[2J\x1B[0f');
      console.log('\n╔═════════════════════════════════════════════════════════════╗');
      console.log(`║  MULTI-COIN PAPER TRADER - Cycle: ${this.cycleCount} | ${timestamp}      ║`);
      console.log('╚═════════════════════════════════════════════════════════════╝\n');

      const results: Array<{ symbol: string; price: number; result: any; tfData: any }> = [];
      for (const symbol of SYMBOLS) {
        const trader = this.traders.get(symbol)!;
        const primaryTf = trader.state.timeframes.get(CONFIG.primaryInterval);
        const currentPrice = (primaryTf && primaryTf.candles && primaryTf.candles.length > 0) 
          ? primaryTf.candles[primaryTf.candles.length - 1].close 
          : 0;
        
        const result = await trader.tick(this.client, timestamp);
        results.push({ symbol, price: currentPrice, result, tfData: result.tfData });

        trader.saveState();
      }

      for (const { symbol, price, result } of results) {
        const trader = this.traders.get(symbol)!;
        const priceDisplay = price > 0 ? `$${price.toFixed(2)}` : 'N/A';
        
        let statusLine = `${symbol.padEnd(8)}: ${priceDisplay.padEnd(10)} | `;
        
        if (trader.state.openTrade) {
          const trade = trader.state.openTrade;
          const isLong = trade.direction === 'LONG';
          const priceDiff = isLong
            ? price - trade.entryPrice
            : trade.entryPrice - price;
          const unrealizedPnl = priceDiff * trade.currentPositionSize;
          const pnlPercent = (priceDiff / trade.entryPrice) * 100;
          const pnlSign = unrealizedPnl >= 0 ? '+' : '';
          const pnlPrecision = trade.entryPrice < 1 ?4 : 2;
          const tpInfo = [trade.tp1Hit ? 'TP1' : '', trade.tp2Hit ? 'TP2' : '', trade.tp3Hit ? 'TP3' : ''].filter(Boolean).join(' ') || 'TP1/TP2/TP3';
          statusLine += `OPEN ${trade.direction.padEnd(6)} | ${pnlSign}$${unrealizedPnl.toFixed(pnlPrecision)} (${pnlSign}${pnlPercent.toFixed(1)}%) | SL:$${trade.stopLoss.toFixed(2)} | TP1:$${trade.takeProfit1.toFixed(2)} TP2:$${trade.takeProfit2.toFixed(2)} TP3:$${trade.takeProfit3.toFixed(2)}`;
        } else {
          statusLine += `${result.status.padEnd(10)} | ${result.details}`;
        }

        console.log(statusLine);
      }

      if (this.cycleCount % 5 === 0) {
        console.log();
        this.printSummary();
      }

      const elapsed = Date.now() - cycleStart;
      const waitTime = Math.max(0, CONFIG.checkIntervalMs - elapsed);
      await new Promise(resolve => setTimeout(resolve, waitTime));
    }

    console.log('\n🔄 Trading loop stopped. Performing final save...');
    this.stop();
  }

  stop(): void {
    this.running = false;
    console.log('\n\nStopping multi-coin paper trader...');
    
    for (const trader of this.traders.values()) {
      trader.saveState();
    }
    
    this.printSummary();
  }
}

async function main() {
  const orchestrator = new MultiCoinOrchestrator();

  process.on('SIGINT', () => {
    console.log('\n\n⚠  SIGINT received. Stopping gracefully...');
    orchestrator.running = false;
  });

  await orchestrator.initialize();
  await orchestrator.run();
  console.log('\n✅ Multi-coin paper trader stopped successfully.\n');
  process.exit(0);
}

main().catch(console.error);