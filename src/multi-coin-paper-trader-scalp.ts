#!/usr/bin/env node
/**
 * Multi-Coin Paper Trading - MOMENTUM SCALPER (5m Primary)
 *
 * TRUE SCALPING: Momentum-based entries, NOT structure-based.
 * - Volume spikes + price breakouts
 * - RSI momentum crossovers
 * - Bollinger Band breakouts
 * - EMA crossovers (9/21)
 * - Quick fixed % targets (0.3-0.5%)
 *
 * This is DIFFERENT from the day trader which uses SMC/ICT.
 *
 * Usage: npm run paper-trade-multi-scalp
 */

// Load environment variables FIRST
import dotenv from 'dotenv';
dotenv.config();

import { createRequire } from 'module';
import fs from 'fs';
import path from 'path';

const require = createRequire(import.meta.url);
const Binance = require('binance-api-node').default;
import { Candle, SMCIndicators } from './smc-indicators.js';
import { FeatureExtractor, TradeFeatures } from './trade-features.js';
import { LightGBMPredictor } from './lightgbm-predictor.js';
import { detectLiquiditySignals, SCALP_CONFIG, LiquiditySignals } from './liquidity-signals.js';
import { RegimeDetector, SCALP_REGIME_CONFIG, extractRegimeSignals, PortfolioRegime, CoinRegimeSignals } from './regime-detector.js';
import { ParameterAnalyzer, ClosedTrade, AnalysisResult, formatSuggestions, DEFAULT_CONFIG } from './parameter-analyzer.js';
import { LLMReporter, LLMReport, formatRecommendations, DEFAULT_LLM_CONFIG } from './llm-reporter.js';

// ═══════════════════════════════════════════════════════════════
// PORTFOLIO-LEVEL REGIME (shared across all CoinTraders)
// ═══════════════════════════════════════════════════════════════
const regimeDetector = new RegimeDetector(SCALP_REGIME_CONFIG);
let currentPortfolioRegime: PortfolioRegime = {
  regime: 'TRANSITION',
  confidence: 50,
  sizeMultiplier: 0.75,
  metrics: {
    avgAdx: 0,
    avgVolumeRatio: 0,
    bbExpansionRatio: 0,
    trendingCoinCount: 0,
    totalCoinCount: 0,
  },
  reasons: ['Not yet classified'],
};

// ═══════════════════════════════════════════════════════════════
// PARAMETER ANALYZER (shared across all CoinTraders)
// ═══════════════════════════════════════════════════════════════
const parameterAnalyzer = new ParameterAnalyzer(DEFAULT_CONFIG);
let lastAnalysisResult: AnalysisResult | null = null;

// ═══════════════════════════════════════════════════════════════
// LLM REPORTER (sends analysis to GLM 5 for recommendations)
// ═══════════════════════════════════════════════════════════════
const llmReporter = new LLMReporter(DEFAULT_LLM_CONFIG);
let lastLLMReport: LLMReport | null = null;

// Top 20 coins for scalping
const SYMBOLS = [
  'BTCUSDT',  'ETHUSDT',
  'BNBUSDT',  'SOLUSDT',
  'XRPUSDT',  'DOGEUSDT',
  'ADAUSDT',  'AVAXUSDT',
  'LINKUSDT', 'DOTUSDT',
  'MATICUSDT', 'ATOMUSDT',
  'NEARUSDT', 'ARBUSDT',
  'OPUSDT',   'INJUSDT',
  'LDOUSDT',  'SUIUSDT',
  'UNIUSDT',  'TONUSDT',
  'APTUSDT',  'PEPEUSDT',
  'FETUSDT',  'RNDRUSDT',
  'WIFUSDT',  'TIAUSDT',
  'SEIUSDT',  'MINAUSDT',
  'IMXUSDT',  'GMTUSDT',
] as const;

// ═══════════════════════════════════════════════════════════════
// SCALPER CONFIGURATION - Momentum-based
// ═══════════════════════════════════════════════════════════════
const CONFIG = {
  mode: 'MOMENTUM_SCALP',
  intervals: ['1m', '5m', '15m', '4h'] as const,  // Added 4h for trend filter
  primaryInterval: '5m' as const,  // 5m for scalping momentum
  checkIntervalMs: 5000,           // Check every 5s for quick entries
  minCandlesRequired: 50,          // Need less history for momentum
  // maxOpenPositions: 5,          // DISABLED - paper trader needs data

  // ═══════════════════════════════════════════════════════════════
  // REPORTING MODE
  // ═══════════════════════════════════════════════════════════════
  reporting: {
    mode: 'verbose',              // 'verbose' = all coins (default), 'summary' = clean output
    showOpenTrades: true,          // Always show open trade details
    showStreaks: true,             // Show current win/loss streak
    showRecent: true,              // Show last 10 trades performance
    summaryFile: path.join(process.cwd(), 'data', 'paper-trades-summary-scalp-live.json'),  // Live summary for OpenClaw
  },

  // ═══════════════════════════════════════════════════════════════
  // MOMENTUM THRESHOLDS
  // ═══════════════════════════════════════════════════════════════
  momentum: {
    // Volume spike detection (MOMENTUM needs 2x, RANGE needs 1.2x)
    volumeSpikeMultipleMomentum: 2.0,   // MOMENTUM mode: Volume > 2x average
    volumeSpikeMultipleRange: 1.2,      // RANGE mode: Volume > 1.2x average (was 1.5)
    volumeSpikeMultiple: 1.2,           // Default (used for signal counting)
    volumeAvgPeriod: 20,                // 20-candle average for comparison

    // RSI settings
    rsiPeriod: 14,
    rsiBullishCross: 50,           // RSI crossing above 50 = bullish momentum
    rsiBearishCross: 50,           // RSI crossing below 50 = bearish momentum
    rsiOverbought: 75,             // was 70 - at BB extremes RSI 70 is common
    rsiOversold: 25,               // was 30

    // EMA crossover
    emaFast: 9,
    emaSlow: 21,

    // Bollinger Bands
    bbPeriod: 20,
    bbStdDev: 2,

    // Price breakout
    breakoutLookback: 10,          // Look for break of 10-candle high/low

    // Candle momentum
    minBodyRatio: 0.5,             // Body must be > 50% of candle range (was 0.6 - range reversals have smaller bodies)

    // Confluence required (increased for quality entries)
    minSignals: 2,                 // Need at least 2 momentum signals

    // Pullback entry thresholds (for RANGE mode)
    pullbackToEmaPct: 0.004,        // Enter when price within 0.4% of EMA
    pullbackToVwapPct: 0.004,      // Enter when price within 0.4% of VWAP
    maxDistanceFromEma: 0.015,     // Skip if price > 1.5% from EMA

    // MACD settings
    macdFast: 12,
    macdSlow: 26,
    macdSignal: 9,
  },

  // ═══════════════════════════════════════════════════════════════
  // REGIME DETECTION: MOMENTUM vs RANGE vs NONE
  // ═══════════════════════════════════════════════════════════════
  regime: {
    atrPeriod: 14,

    // ADX thresholds for regime classification
    adxMomentumMin: 30,            // ADX > 30 = strong trend (was 25 - less MOMENTUM, more RANGE)
    adxRangeMax: 25,               // ADX < 25 = weak/range-bound (was 20 - wider RANGE window)

    // BB width thresholds (as multiple of average)
    bbExpandingMultiple: 1.3,      // BB width > 1.3x avg = expanding/volatile (was 1.2)
    bbNormalMaxMultiple: 1.8,      // BB width < 1.8x avg = normal (was 1.5 - more tolerance)

    // BB mean reversion thresholds (for RANGE mode)
    rangeLongThreshold: 0.15,      // BB < 15% for LONG entries (was 25% - TRUE extreme only)
    rangeShortThreshold: 0.85,     // BB > 85% for SHORT entries (was 75%)

    // MOMENTUM mode: breakout entries, trail at -1R (no fixed TPs)
    // RANGE mode: mean reversion with TP1/TP2
    // NONE mode: transition/squeeze - skip trading
  },

  // ═══════════════════════════════════════════════════════════════
  // AUTO-LEARNING
  // ═══════════════════════════════════════════════════════════════
  autoLearn: {
    enabled: false,                // DISABLED - Collecting data with new quant features (Williams %R, OFI, 1.1x vol, 25% Kelly)
    triggerEveryNTrades: 100,      // Retrain after every 100 closed trades
    minTradesForTraining: 50,     // Need at least 50 trades to train
  },

  // ML filtering (set to 0 for data collection, 0.30+ to filter trades)
  minWinProbability: 0,           // Disabled - ML model AUC 0.49 is worse than random! Collecting data first.

  // ═══════════════════════════════════════════════════════════════
  // ENTRY/EXIT - DUAL MODE TARGETS
  // ═══════════════════════════════════════════════════════════════
  targets: {
    stopLossPct: 0.40,             // Fallback SL % (was 0.50 - tighter stops)
    // RANGE MODE: BB-BASED TARGETS (Mean Reversion)
    tp1ClosePct: 0.60,             // Close 60% at Middle BB (was 70% - leave more for TP2)
    tp2ClosePct: 0.40,             // Close 40% at Opposite BB (was 30%)
    protectedProfitR: 0.2,         // After TP1, SL moves to Entry + 0.2R (protected profit, not breakeven)
    phase2TriggerR: 0.6,           // When profit reaches TP2 - 0.4R (was 0.7)
    phase2TrailR: 0.4,             // Trail at 0.4R distance (was 0.5)
    // MOMENTUM MODE: Breakout trail (no fixed TPs)
    momentumTrailR: 1.0,           // Trail at -1R from entry (let winners run)
    momentumTrailAfterR: 0.5,      // Start trailing after 0.5R profit
    // Time stop for runners
    timeStopCandles: 3,            // Exit remaining position after 3 candles (15 min)
  },

  // ═══════════════════════════════════════════════════════════════
  // SCRAPER TIMEOUT - PREVENT TILT/OVERTRADING
  // ═══════════════════════════════════════════════════════════════
  consecutiveLossCooldownMs: 30 * 60 * 1000,  // 30 min break after 3 losses
  maxConsecutiveLosses: 3,                    // Trigger cooldown after this many
  consecutiveWinsCooldownMs: 10 * 60 * 1000,  // 10 min break after 5 wins (overconfidence check)
  maxConsecutiveWins: 5,                      // Trigger cooldown after this many

  // ═══════════════════════════════════════════════════════════════
  // TIMING - TRUE SCALPING (Quality over Quantity)
  // ═══════════════════════════════════════════════════════════════
  maxHoldMinutes: 15,              // Max 15 mins - consistent with 3 candles
  cooldownMs: 90_000,              // 90s cooldown (was 60s - range setups need time)
  onlyEnterOnCandleClose: false,   // Scalper can enter mid-candle
  maxDailyTrades: 9999,            // No limit - need all data for paper trading
  // Overtrading = 34% less profitable. Frequency is enemy of edge.

  // ═══════════════════════════════════════════════════════════════
  // RISK & FEES
  // ═══════════════════════════════════════════════════════════════
  virtualBalancePerCoin: 10000,
  riskPerTradePct: 1.0,            // Risk 1% per scalp
  leverage: 1,
  takerFeeRate: 0.00025,           // 2.5 bps (BNB discount / VIP tier realistic)
  slippageBps: 1,                  // 1 bp (scalper uses limit orders where possible)

  // Data refresh
  refreshMsByInterval: {
    '1m': 5_000,
    '5m': 5_000,
    '15m': 15_000,
  } as Record<string, number>,
};

// ═══════════════════════════════════════════════════════════════
// MOMENTUM INDICATORS
// ═══════════════════════════════════════════════════════════════

interface MomentumSignals {
  volumeSpike: boolean;
  volumeRatio: number;
  rsiValue: number;
  rsiBullishCross: boolean;
  rsiBearishCross: boolean;
  rsiOverbought: boolean;
  rsiOversold: boolean;
  williamsR: number;              // Williams %R: -100 to 0, <-80 oversold, >-20 overbought
  williamsROversold: boolean;
  williamsROverbought: boolean;
  emaFast: number;
  emaSlow: number;
  emaBullishCross: boolean;
  emaBearishCross: boolean;
  emaAligned: 'bullish' | 'bearish' | 'neutral';
  bbUpper: number;
  bbLower: number;
  bbMiddle: number;
  bbPosition: number;  // 0 = at lower band, 0.5 = middle, 1 = at upper band
  bbWidth: number;     // Band width as % of price (for squeeze detection)
  bbWidthAvg: number;  // Average BB width over lookback period
  bbExpanding: boolean; // BB width > 1.2x avg = expanding
  bbBreakoutUp: boolean;
  bbBreakoutDown: boolean;
  priceBreakoutUp: boolean;
  priceBreakoutDown: boolean;
  candleMomentum: 'bullish' | 'bearish' | 'neutral';

  // MACD
  macdLine: number;
  macdSignal: number;
  macdHistogram: number;
  macdBullishCross: boolean;
  macdBearishCross: boolean;

  // Volatility / Regime
  atr: number;
  atrPercent: number;  // ATR as % of price
  regime: 'MOMENTUM' | 'RANGE';  // NO NONE zone - hysteresis handles transition
  adx: number;         // ADX trend strength (0-100)
  plusDI: number;      // +DI for bullish strength
  minusDI: number;     // -DI for bearish strength

  // VWAP (Institutional anchor)
  vwap: number;
  vwapDeviation: number;      // (price - vwap) / vwap as %
  vwapDeviationStd: number;   // How many std devs from VWAP
  priceAboveVwap: boolean;
  // Multi-session VWAP bands
  sessionVwapAsia: number;
  sessionVwapLondon: number;
  sessionVwapNy: number;
  sessionVwapUpper: number;
  sessionVwapLower: number;
  currentSession: 'asia' | 'london' | 'ny';

  // Kill Zone (Session timing)
  killZone: 'LONDON' | 'NY_OPEN' | 'NY_AFTERNOON' | 'ASIA' | 'OFF_HOURS';
  isKillZone: boolean;        // True if in high-probability session

  // Structure-based stops (swing points)
  swingHigh: number | null;   // Most recent swing high for SHORT stops
  swingLow: number | null;    // Most recent swing low for LONG stops

  // Order Flow Imbalance (OFI) - Market microstructure edge
  ofi: number;                // Order Flow Imbalance: -1 (bearish) to +1 (bullish)
  ofiStrongBullish: boolean;  // OFI > 0.3 = strong buying pressure
  ofiStrongBearish: boolean;  // OFI < -0.3 = strong selling pressure

  // Aggregated
  bullishSignals: number;
  bearishSignals: number;
  direction: 'LONG' | 'SHORT' | 'NEUTRAL';
  strength: number;  // 0-1

  // Liquidity signals (stop hunts, fakeouts, QML, flip zones)
  liquiditySignals?: LiquiditySignals;
}

function calculateRSI(candles: Candle[], period: number): number[] {
  const rsi: number[] = [];
  let gainsSum = 0;   // Cumulative sum for initial average only
  let lossesSum = 0;  // Cumulative sum for initial average only
  let avgGain = 0;    // Running Wilder's average
  let avgLoss = 0;    // Running Wilder's average

  for (let i = 1; i < candles.length; i++) {
    const change = candles[i].close - candles[i - 1].close;
    const gain = change > 0 ? change : 0;
    const loss = change < 0 ? -change : 0;

    if (i <= period) {
      // Build cumulative sums for initial average
      gainsSum += gain;
      lossesSum += loss;

      if (i === period) {
        // Initialize running averages
        avgGain = gainsSum / period;
        avgLoss = lossesSum / period;
        const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
        rsi.push(100 - (100 / (1 + rs)));
      }
    } else {
      // Wilder's smoothing: use previous average, not cumulative sum
      avgGain = ((avgGain * (period - 1)) + gain) / period;
      avgLoss = ((avgLoss * (period - 1)) + loss) / period;
      const rs = avgLoss === 0 ? 100 : avgGain / avgLoss;
      rsi.push(100 - (100 / (1 + rs)));
    }
  }

  return rsi;
}

function calculateWilliamsR(candles: Candle[], period: number): number[] {
  const williamsR: number[] = [];

  if (candles.length < period) {
    return williamsR;
  }

  for (let i = period - 1; i < candles.length; i++) {
    const slice = candles.slice(i - period + 1, i + 1);
    const highestHigh = Math.max(...slice.map(c => c.high));
    const lowestLow = Math.min(...slice.map(c => c.low));
    const currentClose = candles[i].close;

    // Williams %R: -100 * (highestHigh - currentClose) / (highestHigh - lowestLow)
    // Range: -100 (most oversold) to 0 (most overbought)
    const range = highestHigh - lowestLow;
    const wr = range === 0 ? -50 : -100 * (highestHigh - currentClose) / range;
    williamsR.push(wr);
  }

  return williamsR;
}

// ═══════════════════════════════════════════════════════════════
// ORDER FLOW IMBALANCE (OFI) - Market Microstructure Edge
// ═══════════════════════════════════════════════════════════════

interface OrderBookDepth {
  bids: [price: number, quantity: number][];
  asks: [price: number, quantity: number][];
}

interface OFISignal {
  ofi: number;              // -1 to +1, positive = bullish
  ofiStrongBullish: boolean; // OFI > 0.3
  ofiStrongBearish: boolean; // OFI < -0.3
}

/**
 * Calculate Order Flow Imbalance from order book depth
 * OFI = (bid_volume - ask_volume) / (bid_volume + ask_volume)
 * Uses top N levels for calculation (default 5)
 */
function calculateOFI(depth: OrderBookDepth, levels: number = 10): OFISignal {
  const bids = depth.bids.slice(0, levels);
  const asks = depth.asks.slice(0, levels);

  // Calculate total bid and ask volumes
  const bidVolume = bids.reduce((sum, [price, qty]) => sum + price * qty, 0);
  const askVolume = asks.reduce((sum, [price, qty]) => sum + price * qty, 0);

  const totalVolume = bidVolume + askVolume;

  // OFI: -1 (all asks) to +1 (all bids)
  const ofi = totalVolume > 0 ? (bidVolume - askVolume) / totalVolume : 0;

  return {
    ofi,
    ofiStrongBullish: ofi > 0.3,
    ofiStrongBearish: ofi < -0.3,
  };
}

/**
 * Fetch order book depth from Binance REST API
 * Returns top 20 levels (bids and asks)
 */
async function fetchOrderBookDepth(_client: any, symbol: string): Promise<OrderBookDepth | null> {
  try {
    // Use Binance REST API directly for order book depth
    const response = await fetch(`https://api.binance.com/api/v3/depth?symbol=${symbol}&limit=20`);
    if (!response.ok) {
      throw new Error(`HTTP ${response.status}`);
    }
    const data = await response.json() as { bids: [string, string][]; asks: [string, string][] };
    return {
      bids: data.bids.map((b: [string, string]) => [parseFloat(b[0]), parseFloat(b[1])]),
      asks: data.asks.map((a: [string, string]) => [parseFloat(a[0]), parseFloat(a[1])]),
    };
  } catch (error: any) {
    // Silently fail - OFI is optional enhancement
    return null;
  }
}

function calculateEMA(candles: Candle[], period: number): number[] {
  const ema: number[] = [];
  const multiplier = 2 / (period + 1);

  // Start with SMA
  let sum = 0;
  for (let i = 0; i < period && i < candles.length; i++) {
    sum += candles[i].close;
  }
  ema.push(sum / Math.min(period, candles.length));

  // Calculate EMA
  for (let i = period; i < candles.length; i++) {
    const value = (candles[i].close - ema[ema.length - 1]) * multiplier + ema[ema.length - 1];
    ema.push(value);
  }

  return ema;
}

function calculateBollingerBands(candles: Candle[], period: number, stdDev: number): {
  upper: number[];
  middle: number[];
  lower: number[];
} {
  const upper: number[] = [];
  const middle: number[] = [];
  const lower: number[] = [];

  for (let i = period - 1; i < candles.length; i++) {
    const slice = candles.slice(i - period + 1, i + 1);
    const mean = slice.reduce((sum, c) => sum + c.close, 0) / period;
    const variance = slice.reduce((sum, c) => sum + Math.pow(c.close - mean, 2), 0) / period;
    const std = Math.sqrt(variance);

    middle.push(mean);
    upper.push(mean + stdDev * std);
    lower.push(mean - stdDev * std);
  }

  return { upper, middle, lower };
}

function calculateMACD(candles: Candle[], fastPeriod: number, slowPeriod: number, signalPeriod: number): {
  macdLine: number[];
  signalLine: number[];
  histogram: number[];
} {
  const emaFast = calculateEMA(candles, fastPeriod);
  const emaSlow = calculateEMA(candles, slowPeriod);

  // MACD Line = Fast EMA - Slow EMA
  const macdLine: number[] = [];
  const startIdx = slowPeriod - fastPeriod;

  for (let i = 0; i < emaSlow.length; i++) {
    const fastIdx = i + startIdx;
    if (fastIdx >= 0 && fastIdx < emaFast.length) {
      macdLine.push(emaFast[fastIdx] - emaSlow[i]);
    }
  }

  // Signal Line = EMA of MACD Line
  const signalLine: number[] = [];
  if (macdLine.length >= signalPeriod) {
    const multiplier = 2 / (signalPeriod + 1);
    let sum = 0;
    for (let i = 0; i < signalPeriod; i++) {
      sum += macdLine[i];
    }
    signalLine.push(sum / signalPeriod);

    for (let i = signalPeriod; i < macdLine.length; i++) {
      const value = (macdLine[i] - signalLine[signalLine.length - 1]) * multiplier + signalLine[signalLine.length - 1];
      signalLine.push(value);
    }
  }

  // Histogram = MACD Line - Signal Line
  const histogram: number[] = [];
  const offset = macdLine.length - signalLine.length;
  for (let i = 0; i < signalLine.length; i++) {
    histogram.push(macdLine[i + offset] - signalLine[i]);
  }

  return { macdLine, signalLine, histogram };
}

function calculateVWAP(candles: Candle[]): { vwap: number; stdDev: number } {
  // VWAP = Cumulative(Price * Volume) / Cumulative(Volume)
  // Use typical price: (High + Low + Close) / 3
  let cumulativePV = 0;
  let cumulativeVolume = 0;
  const typicalPrices: number[] = [];

  for (const candle of candles) {
    const typicalPrice = (candle.high + candle.low + candle.close) / 3;
    typicalPrices.push(typicalPrice);
    cumulativePV += typicalPrice * candle.volume;
    cumulativeVolume += candle.volume;
  }

  const vwap = cumulativeVolume > 0 ? cumulativePV / cumulativeVolume : candles[candles.length - 1]?.close || 0;

  // Calculate standard deviation of price from VWAP
  const deviations = typicalPrices.map(p => Math.pow(p - vwap, 2));
  const variance = deviations.reduce((a, b) => a + b, 0) / deviations.length;
  const stdDev = Math.sqrt(variance);

  return { vwap, stdDev };
}

interface MultiSessionVWAP {
  asia: { vwap: number; upper: number; lower: number };
  london: { vwap: number; upper: number; lower: number };
  ny: { vwap: number; upper: number; lower: number };
  currentSession: 'asia' | 'london' | 'ny';
}

function calculateMultiSessionVWAP(candles: Candle[], numStdDev: number = 1.5): MultiSessionVWAP {
  const asia: { vwap: number; upper: number; lower: number } = { vwap: 0, upper: 0, lower: 0 };
  const london: { vwap: number; upper: number; lower: number } = { vwap: 0, upper: 0, lower: 0 };
  const ny: { vwap: number; upper: number; lower: number } = { vwap: 0, upper: 0, lower: 0 };

  // Group candles by session
  const asiaCandles: Candle[] = [];
  const londonCandles: Candle[] = [];
  const nyCandles: Candle[] = [];

  for (const candle of candles) {
    const hour = new Date(candle.timestamp).getUTCHours();

    // Asia: 00:00-08:00 UTC (inclusive overlap)
    if (hour >= 0 && hour < 8) {
      asiaCandles.push(candle);
    }
    // London: 07:00-16:00 UTC
    if (hour >= 7 && hour < 16) {
      londonCandles.push(candle);
    }
    // NY: 13:00-22:00 UTC
    if (hour >= 13 && hour < 22) {
      nyCandles.push(candle);
    }
  }

  // Calculate VWAP for each session
  const asiaData = asiaCandles.length > 0 ? calculateVWAPWithBands(asiaCandles, numStdDev) : { vwap: 0, upper: 0, lower: 0 };
  const londonData = londonCandles.length > 0 ? calculateVWAPWithBands(londonCandles, numStdDev) : { vwap: 0, upper: 0, lower: 0 };
  const nyData = nyCandles.length > 0 ? calculateVWAPWithBands(nyCandles, numStdDev) : { vwap: 0, upper: 0, lower: 0 };

  // Determine current session
  const currentHour = new Date(candles[candles.length - 1].timestamp).getUTCHours();
  let currentSession: 'asia' | 'london' | 'ny' = 'asia';
  if (currentHour >= 13 && currentHour < 22) {
    currentSession = 'ny';
  } else if (currentHour >= 7 && currentHour < 16) {
    currentSession = 'london';
  }

  return {
    asia: asiaData,
    london: londonData,
    ny: nyData,
    currentSession,
  };
}

function calculateVWAPWithBands(candles: Candle[], numStdDev: number): { vwap: number; upper: number; lower: number } {
  let cumulativePV = 0;
  let cumulativeVolume = 0;
  const typicalPrices: number[] = [];

  for (const candle of candles) {
    const typicalPrice = (candle.high + candle.low + candle.close) / 3;
    typicalPrices.push(typicalPrice);
    cumulativePV += typicalPrice * candle.volume;
    cumulativeVolume += candle.volume;
  }

  const vwap = cumulativeVolume > 0 ? cumulativePV / cumulativeVolume : candles[candles.length - 1]?.close || 0;

  const deviations = typicalPrices.map(p => Math.pow(p - vwap, 2));
  const variance = deviations.reduce((a, b) => a + b, 0) / deviations.length;
  const stdDev = Math.sqrt(variance);

  return {
    vwap,
    upper: vwap + numStdDev * stdDev,
    lower: vwap - numStdDev * stdDev,
  };
}

function getKillZone(timestamp: number): { zone: 'LONDON' | 'NY_OPEN' | 'NY_AFTERNOON' | 'ASIA' | 'OFF_HOURS'; isActive: boolean } {
  // Kill zones based on UTC hours (high-probability trading windows)
  // London: 07:00-10:00 UTC (8-11 AM London)
  // NY Open: 13:00-16:00 UTC (8-11 AM EST)
  // NY Afternoon: 18:00-20:00 UTC (2-4 PM EST)
  // Asia: 00:00-03:00 UTC (Tokyo/Sydney overlap)

  const date = new Date(timestamp);
  const hour = date.getUTCHours();

  if (hour >= 7 && hour < 10) {
    return { zone: 'LONDON', isActive: true };
  } else if (hour >= 13 && hour < 16) {
    return { zone: 'NY_OPEN', isActive: true };
  } else if (hour >= 18 && hour < 20) {
    return { zone: 'NY_AFTERNOON', isActive: true };
  } else if (hour >= 0 && hour < 3) {
    return { zone: 'ASIA', isActive: true };
  } else {
    return { zone: 'OFF_HOURS', isActive: false };
  }
}

function calculateATR(candles: Candle[], period: number): number[] {
  const atr: number[] = [];
  const trueRanges: number[] = [];

  for (let i = 1; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1].close;

    const tr = Math.max(
      high - low,
      Math.abs(high - prevClose),
      Math.abs(low - prevClose)
    );
    trueRanges.push(tr);

    if (trueRanges.length >= period) {
      if (atr.length === 0) {
        // First ATR is simple average
        const sum = trueRanges.slice(-period).reduce((a, b) => a + b, 0);
        atr.push(sum / period);
      } else {
        // Subsequent ATR uses smoothing
        const prevATR = atr[atr.length - 1];
        atr.push((prevATR * (period - 1) + tr) / period);
      }
    }
  }

  return atr;
}

function calculateADX(candles: Candle[], period: number = 14): { adx: number[]; plusDI: number[]; minusDI: number[] } {
  const adx: number[] = [];
  const plusDI: number[] = [];
  const minusDI: number[] = [];

  if (candles.length < period * 2) {
    return { adx: [0], plusDI: [0], minusDI: [0] };
  }

  // Calculate TR, +DM, -DM
  const tr: number[] = [];
  const plusDM: number[] = [];
  const minusDM: number[] = [];

  for (let i = 1; i < candles.length; i++) {
    const high = candles[i].high;
    const low = candles[i].low;
    const prevClose = candles[i - 1].close;
    const prevHigh = candles[i - 1].high;
    const prevLow = candles[i - 1].low;

    const trueRange = Math.max(high - low, Math.abs(high - prevClose), Math.abs(low - prevClose));
    const upMove = high - prevHigh;
    const downMove = prevLow - low;

    const positiveDM = (upMove > downMove && upMove > 0) ? upMove : 0;
    const negativeDM = (downMove > upMove && downMove > 0) ? downMove : 0;

    tr.push(trueRange);
    plusDM.push(positiveDM);
    minusDM.push(negativeDM);
  }

  // Calculate smoothed TR, +DM, -DM for first period
  let smoothedTR = 0;
  let smoothedPlusDM = 0;
  let smoothedMinusDM = 0;

  for (let i = 0; i < period; i++) {
    smoothedTR += tr[i];
    smoothedPlusDM += plusDM[i];
    smoothedMinusDM += minusDM[i];
  }

  // Calculate +DI and -DI
  const di: number[] = [];
  for (let i = period; i < tr.length; i++) {
    smoothedPlusDM = smoothedPlusDM - (smoothedPlusDM / period) + plusDM[i];
    smoothedMinusDM = smoothedMinusDM - (smoothedMinusDM / period) + minusDM[i];
    smoothedTR = smoothedTR - (smoothedTR / period) + tr[i];

    const plusDI_val = 100 * (smoothedPlusDM / smoothedTR);
    const minusDI_val = 100 * (smoothedMinusDM / smoothedTR);

    plusDI.push(plusDI_val);
    minusDI.push(minusDI_val);

    const dx = 100 * Math.abs(plusDI_val - minusDI_val) / (plusDI_val + minusDI_val || 1);
    di.push(dx);
  }

  // Calculate ADX as smoothed DX
  if (di.length >= period) {
    let adxSum = 0;
    for (let i = 0; i < period; i++) {
      adxSum += di[i];
    }
    adx.push(adxSum / period);

    for (let i = period; i < di.length; i++) {
      const newAdx = (adx[adx.length - 1] * (period - 1) + di[i]) / period;
      adx.push(newAdx);
    }
  }

  return { adx, plusDI, minusDI };
}

// ═══════════════════════════════════════════════════════════════
// STRUCTURE-BASED STOPS - Find swing highs/lows
// ═══════════════════════════════════════════════════════════════

interface SwingPoints {
  recentSwingHigh: number | null;
  recentSwingLow: number | null;
  swingHighIdx: number;
  swingLowIdx: number;
}

function findSwingPoints(candles: Candle[], lookback: number = 5): SwingPoints {
  /**
   * Find the most recent swing high and swing low.
   * A swing high is a candle whose high is higher than the surrounding candles.
   * A swing low is a candle whose low is lower than the surrounding candles.
   */
  let recentSwingHigh: number | null = null;
  let recentSwingLow: number | null = null;
  let swingHighIdx = -1;
  let swingLowIdx = -1;

  const len = candles.length;
  if (len < lookback * 2 + 1) {
    return { recentSwingHigh: null, recentSwingLow: null, swingHighIdx: -1, swingLowIdx: -1 };
  }

  // Search backwards from most recent candle (skip last few as they can't be confirmed yet)
  for (let i = len - lookback - 1; i >= lookback; i--) {
    const high = candles[i].high;
    const low = candles[i].low;

    // Check if this is a swing high (higher than all candles in lookback range)
    let isSwingHigh = true;
    let isSwingLow = true;

    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (candles[j].high >= high) isSwingHigh = false;
      if (candles[j].low <= low) isSwingLow = false;
    }

    if (isSwingHigh && recentSwingHigh === null) {
      recentSwingHigh = high;
      swingHighIdx = i;
    }
    if (isSwingLow && recentSwingLow === null) {
      recentSwingLow = low;
      swingLowIdx = i;
    }

    // Once we have both, we can stop
    if (recentSwingHigh !== null && recentSwingLow !== null) {
      break;
    }
  }

  return { recentSwingHigh, recentSwingLow, swingHighIdx, swingLowIdx };
}

function analyzeMomentum(candles: Candle[], ofiSignal?: OFISignal): MomentumSignals {
  const cfg = CONFIG.momentum;
  const len = candles.length;

  if (len < 30) {
    return {
      volumeSpike: false, volumeRatio: 1,
      rsiValue: 50, rsiBullishCross: false, rsiBearishCross: false,
      rsiOverbought: false, rsiOversold: false,
      williamsR: -50, williamsROversold: false, williamsROverbought: false,
      emaFast: 0, emaSlow: 0, emaBullishCross: false, emaBearishCross: false,
      emaAligned: 'neutral',
      bbUpper: 0, bbLower: 0, bbMiddle: 0, bbPosition: 0.5, bbWidth: 0, bbWidthAvg: 0, bbExpanding: false,
      bbBreakoutUp: false, bbBreakoutDown: false,
      priceBreakoutUp: false, priceBreakoutDown: false,
      candleMomentum: 'neutral',
      macdLine: 0, macdSignal: 0, macdHistogram: 0,
      macdBullishCross: false, macdBearishCross: false,
      atr: 0, atrPercent: 0, regime: 'RANGE' as 'MOMENTUM' | 'RANGE',  // Default to RANGE (safer)
      adx: 0, plusDI: 0, minusDI: 0,
      vwap: 0, vwapDeviation: 0, vwapDeviationStd: 0, priceAboveVwap: false,
      sessionVwapAsia: 0, sessionVwapLondon: 0, sessionVwapNy: 0,
      sessionVwapUpper: 0, sessionVwapLower: 0,
      currentSession: 'asia' as 'asia' | 'london' | 'ny',
      killZone: 'OFF_HOURS' as const, isKillZone: false,
      swingHigh: null, swingLow: null,
      ofi: 0, ofiStrongBullish: false, ofiStrongBearish: false,
      bullishSignals: 0, bearishSignals: 0,
      direction: 'NEUTRAL', strength: 0,
    };
  }

  const current = candles[len - 1];
  const prev = candles[len - 2];

  // Volume spike
  const volumeSlice = candles.slice(-cfg.volumeAvgPeriod - 1, -1);
  const avgVolume = volumeSlice.reduce((sum, c) => sum + c.volume, 0) / volumeSlice.length;
  const volumeRatio = current.volume / avgVolume;
  const volumeSpike = volumeRatio >= cfg.volumeSpikeMultiple;

  // RSI
  const rsiValues = calculateRSI(candles, cfg.rsiPeriod);
  const rsiValue = rsiValues[rsiValues.length - 1] || 50;
  const prevRsi = rsiValues[rsiValues.length - 2] || 50;
  const rsiBullishCross = prevRsi < cfg.rsiBullishCross && rsiValue >= cfg.rsiBullishCross;
  const rsiBearishCross = prevRsi > cfg.rsiBearishCross && rsiValue <= cfg.rsiBearishCross;
  const rsiOverbought = rsiValue >= cfg.rsiOverbought;
  const rsiOversold = rsiValue <= cfg.rsiOversold;

  // Williams %R (momentum oscillator: -100 to 0, <-80 oversold, >-20 overbought)
  const williamsRValues = calculateWilliamsR(candles, cfg.rsiPeriod);
  const williamsR = williamsRValues[williamsRValues.length - 1] || -50;
  const prevWilliamsR = williamsRValues[williamsRValues.length - 2] || -50;
  const williamsROverbought = williamsR > -20;
  const williamsROversold = williamsR < -80;

  // EMA crossover
  const emaFastValues = calculateEMA(candles, cfg.emaFast);
  const emaSlowValues = calculateEMA(candles, cfg.emaSlow);
  const emaFast = emaFastValues[emaFastValues.length - 1] || current.close;
  const emaSlow = emaSlowValues[emaSlowValues.length - 1] || current.close;
  const prevEmaFast = emaFastValues[emaFastValues.length - 2] || emaFast;
  const prevEmaSlow = emaSlowValues[emaSlowValues.length - 2] || emaSlow;
  const emaBullishCross = prevEmaFast <= prevEmaSlow && emaFast > emaSlow;
  const emaBearishCross = prevEmaFast >= prevEmaSlow && emaFast < emaSlow;
  const emaAligned: 'bullish' | 'bearish' | 'neutral' =
    emaFast > emaSlow ? 'bullish' :
    emaFast < emaSlow ? 'bearish' : 'neutral';

  // Bollinger Bands
  const bb = calculateBollingerBands(candles, cfg.bbPeriod, cfg.bbStdDev);
  const bbUpper = bb.upper[bb.upper.length - 1] || current.close * 1.02;
  const bbLower = bb.lower[bb.lower.length - 1] || current.close * 0.98;
  const bbMiddle = bb.middle[bb.middle.length - 1] || current.close;
  const bbBreakoutUp = current.close > bbUpper && prev.close <= bb.upper[bb.upper.length - 2];
  const bbBreakoutDown = current.close < bbLower && prev.close >= bb.lower[bb.lower.length - 2];
  // BB Position: 0 = at lower band, 0.5 = middle, 1 = at upper band
  const bbRange = bbUpper - bbLower;
  const bbPosition = bbRange > 0 ? Math.max(0, Math.min(1, (current.close - bbLower) / bbRange)) : 0.5;
  // BB Width: as % of price (for squeeze detection - narrow bands = explosive move coming)
  const bbWidth = current.close > 0 ? (bbRange / current.close) * 100 : 0;

  // Price breakout (new high/low)
  const lookbackCandles = candles.slice(-cfg.breakoutLookback - 1, -1);
  const recentHigh = Math.max(...lookbackCandles.map(c => c.high));
  const recentLow = Math.min(...lookbackCandles.map(c => c.low));
  const priceBreakoutUp = current.close > recentHigh;
  const priceBreakoutDown = current.close < recentLow;

  // Candle momentum (big body in direction)
  const candleRange = current.high - current.low;
  const candleBody = Math.abs(current.close - current.open);
  const bodyRatio = candleRange > 0 ? candleBody / candleRange : 0;
  const isBullishCandle = current.close > current.open;
  const isBearishCandle = current.close < current.open;
  const candleMomentum: 'bullish' | 'bearish' | 'neutral' =
    bodyRatio >= cfg.minBodyRatio
      ? (isBullishCandle ? 'bullish' : isBearishCandle ? 'bearish' : 'neutral')
      : 'neutral';

  // MACD
  const macd = calculateMACD(candles, cfg.macdFast, cfg.macdSlow, cfg.macdSignal);
  const macdLine = macd.macdLine[macd.macdLine.length - 1] || 0;
  const macdSignalLine = macd.signalLine[macd.signalLine.length - 1] || 0;
  const macdHistogram = macd.histogram[macd.histogram.length - 1] || 0;
  const prevMacdLine = macd.macdLine[macd.macdLine.length - 2] || macdLine;
  const prevMacdSignal = macd.signalLine[macd.signalLine.length - 2] || macdSignalLine;
  const macdBullishCross = prevMacdLine <= prevMacdSignal && macdLine > macdSignalLine;
  const macdBearishCross = prevMacdLine >= prevMacdSignal && macdLine < macdSignalLine;

  // ATR and ADX-based Regime Detection
  const atrValues = calculateATR(candles, CONFIG.regime.atrPeriod);
  const atr = atrValues[atrValues.length - 1] || 0;
  const atrPercent = current.close > 0 ? (atr / current.close) : 0;

  const adxResult = calculateADX(candles, 14);
  const adx = adxResult.adx[adxResult.adx.length - 1] || 0;
  const plusDI = adxResult.plusDI[adxResult.plusDI.length - 1] || 0;
  const minusDI = adxResult.minusDI[adxResult.minusDI.length - 1] || 0;

  // BB Width average calculation (for regime detection)
  // Calculate BB widths for last 20 candles, find average
  const bbWidthLookback = 20;
  const bbWidths: number[] = [];
  for (let i = Math.max(0, bb.upper.length - bbWidthLookback); i < bb.upper.length; i++) {
    const width = bb.middle[i] > 0 ? ((bb.upper[i] - bb.lower[i]) / bb.middle[i]) * 100 : 0;
    bbWidths.push(width);
  }
  const bbWidthAvg = bbWidths.length > 0 ? bbWidths.reduce((a, b) => a + b, 0) / bbWidths.length : bbWidth;
  const bbExpanding = bbWidth > bbWidthAvg * CONFIG.regime.bbExpandingMultiple;

  // NEW REGIME DETECTION: MOMENTUM / RANGE (NO NONE ZONE)
  // Hysteresis with overlapping thresholds to prevent flickering
  // - If was MOMENTUM, stay MOMENTUM until ADX drops below 25
  // - If was RANGE, stay RANGE until ADX exceeds 28
  // - Default to RANGE (safer start)
  let regime: 'MOMENTUM' | 'RANGE';

  // Note: previousRegime parameter would need to be passed in from caller
  // For now, use simpler single-threshold approach: ADX >= 28 = MOMENTUM, else RANGE
  // This eliminates the dead zone while still providing separation
  if (adx >= 28) {
    regime = 'MOMENTUM';
  } else {
    regime = 'RANGE';
  }

  // VWAP (Institutional anchor) - Multi-session
  const multiSessionVwap = calculateMultiSessionVWAP(candles, 1.5);
  const vwap = multiSessionVwap[multiSessionVwap.currentSession]?.vwap || current.close;
  const vwapDeviation = vwap > 0 ? ((current.close - vwap) / vwap) * 100 : 0;
  const vwapDeviationStd = 1.0;  // Simplified for multi-session

  // Use current session's VWAP bands
  const currentSessionVwap = multiSessionVwap[multiSessionVwap.currentSession];
  const sessionVwapUpper = currentSessionVwap?.upper || current.close * 1.01;
  const sessionVwapLower = currentSessionVwap?.lower || current.close * 0.99;
  const priceAboveVwap = current.close > vwap;

  // Kill Zone (Session timing)
  const killZoneData = getKillZone(current.timestamp);
  const killZone = killZoneData.zone;
  const isKillZone = killZoneData.isActive;

  // Count signals
  let bullishSignals = 0;
  let bearishSignals = 0;

  // Volume spike is directional based on candle
  if (volumeSpike && candleMomentum === 'bullish') bullishSignals++;
  if (volumeSpike && candleMomentum === 'bearish') bearishSignals++;

  // RSI
  if (rsiBullishCross || rsiOversold) bullishSignals++;
  if (rsiBearishCross || rsiOverbought) bearishSignals++;

  // Williams %R
  if (williamsROversold) bullishSignals++;        // Williams %R < -80 = oversold
  if (williamsROverbought) bearishSignals++;      // Williams %R > -20 = overbought

  // Order Flow Imbalance (OFI) - market microstructure edge
  if (ofiSignal?.ofiStrongBullish) bullishSignals++;   // OFI > 0.3 = strong buying
  if (ofiSignal?.ofiStrongBearish) bearishSignals++;   // OFI < -0.3 = strong selling

  // EMA
  if (emaBullishCross || emaAligned === 'bullish') bullishSignals++;
  if (emaBearishCross || emaAligned === 'bearish') bearishSignals++;

  // Bollinger
  if (bbBreakoutUp) bullishSignals++;
  if (bbBreakoutDown) bearishSignals++;

  // Price breakout
  if (priceBreakoutUp) bullishSignals++;
  if (priceBreakoutDown) bearishSignals++;

  // Candle momentum
  if (candleMomentum === 'bullish') bullishSignals++;
  if (candleMomentum === 'bearish') bearishSignals++;

  // MACD
  if (macdBullishCross || macdHistogram > 0) bullishSignals++;
  if (macdBearishCross || macdHistogram < 0) bearishSignals++;

  // Direction
  let direction: 'LONG' | 'SHORT' | 'NEUTRAL' = 'NEUTRAL';
  const maxSignals = Math.max(bullishSignals, bearishSignals);

  if (bullishSignals >= cfg.minSignals && bullishSignals > bearishSignals) {
    direction = 'LONG';
  } else if (bearishSignals >= cfg.minSignals && bearishSignals > bullishSignals) {
    direction = 'SHORT';
  }

  const strength = maxSignals / 7;  // 7 possible signals now (added MACD)

  // Find swing points for structure-based stops (lookback=10 for 5m = 50min of structure)
  const swingPoints = findSwingPoints(candles, 10);  // Increased from 5 to 10

  return {
    volumeSpike, volumeRatio,
    rsiValue, rsiBullishCross, rsiBearishCross, rsiOverbought, rsiOversold,
    williamsR, williamsROversold, williamsROverbought,
    emaFast, emaSlow, emaBullishCross, emaBearishCross, emaAligned,
    bbUpper, bbLower, bbMiddle, bbPosition, bbWidth, bbWidthAvg, bbExpanding, bbBreakoutUp, bbBreakoutDown,
    priceBreakoutUp, priceBreakoutDown,
    candleMomentum,
    macdLine, macdSignal: macdSignalLine, macdHistogram,
    macdBullishCross, macdBearishCross,
    atr, atrPercent, regime,
    adx, plusDI, minusDI,
    vwap, vwapDeviation, vwapDeviationStd, priceAboveVwap,
    sessionVwapAsia: multiSessionVwap.asia.vwap,
    sessionVwapLondon: multiSessionVwap.london.vwap,
    sessionVwapNy: multiSessionVwap.ny.vwap,
    sessionVwapUpper,
    sessionVwapLower,
    currentSession: multiSessionVwap.currentSession,
    killZone, isKillZone,
    swingHigh: swingPoints.recentSwingHigh,
    swingLow: swingPoints.recentSwingLow,
    ofi: ofiSignal?.ofi ?? 0,
    ofiStrongBullish: ofiSignal?.ofiStrongBullish ?? false,
    ofiStrongBearish: ofiSignal?.ofiStrongBearish ?? false,
    bullishSignals, bearishSignals,
    direction, strength,
  };
}

// ═══════════════════════════════════════════════════════════════
// PAPER TRADE TYPES
// ═══════════════════════════════════════════════════════════════

// Snapshot of market state at a point in time (for ML training)
interface MarketSnapshot {
  price: number;
  timestamp: number;
  regime: 'MOMENTUM' | 'RANGE' | 'NONE';
  atrPercent: number;
  bbPosition: number;
  bbWidth: number;
  rsiValue: number;
  emaFast: number;
  emaSlow: number;
  emaAligned: string;
  macdLine: number;
  macdSignal: number;
  macdHistogram: number;
  vwap: number;
  vwapDeviation: number;
  vwapDeviationStd: number;
  priceAboveVwap: boolean;
  volumeRatio: number;
  volumeSpike: boolean;
  killZone: string;
  isKillZone: boolean;
  direction: string;
  strength: number;
}

interface PaperTrade {
  id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  entryTime: number;
  stopLoss: number;
  originalStopLoss: number;
  tp1DistanceR: number;            // R = TP1 distance (tp1 - entry), used for all R calculations
  takeProfit1: number;              // Middle BB (for RANGE mode)
  takeProfit2: number;              // Opposite BB band (for RANGE mode)
  isMomentumMode: boolean;          // true = MOMENTUM (trail only), false = RANGE (TP1/TP2)
  trailingStop: number | null;
  originalPositionSize: number;
  currentPositionSize: number;
  candlesHeld: number;              // Track for time stop
  lastCandleTime: number;           // Track for incrementing candlesHeld on candle close
  tp1Hit: boolean;
  tp2Hit: boolean;
  phase2Active: boolean;            // Phase 2 trailing active (RANGE mode) or trailing active (MOMENTUM mode)
  status: 'OPEN' | 'CLOSED';
  pnl?: number;
  pnlPercent?: number;
  exitPrice?: number;
  exitTime?: number;
  exitReason?: 'TP1' | 'TP2' | 'SL' | 'TRAILING' | 'TIMEOUT' | 'MANUAL' | 'TREND_REVERSAL';
  feesPaid: number;
  momentumStrength: number;
  signals: string[];
  // ML training data - captured at entry and exit
  entryFeatures?: MarketSnapshot;
  exitFeatures?: MarketSnapshot;
}

function captureSnapshot(m: MomentumSignals, price: number): MarketSnapshot {
  return {
    price,
    timestamp: Date.now(),
    regime: m.regime,
    atrPercent: m.atrPercent,
    bbPosition: m.bbPosition,
    bbWidth: m.bbUpper && m.bbLower && price > 0 ? ((m.bbUpper - m.bbLower) / price) * 100 : 0,
    rsiValue: m.rsiValue,
    emaFast: m.emaFast,
    emaSlow: m.emaSlow,
    emaAligned: m.emaAligned,
    macdLine: m.macdLine,
    macdSignal: m.macdSignal,
    macdHistogram: m.macdHistogram,
    vwap: m.vwap,
    vwapDeviation: m.vwapDeviation,
    vwapDeviationStd: m.vwapDeviationStd,
    priceAboveVwap: m.priceAboveVwap,
    volumeRatio: m.volumeRatio,
    volumeSpike: m.volumeSpike,
    killZone: m.killZone,
    isKillZone: m.isKillZone,
    direction: m.direction,
    strength: m.strength,
  };
}

interface TimeframeData {
  candles: Candle[];
  momentum: MomentumSignals;
  lastUpdate: number;
  // Order book depth for OFI calculation (updated periodically)
  orderBookDepth: OrderBookDepth | null;
  lastOFIUpdate: number;
}

interface CoinTradingState {
  symbol: string;
  balance: number;
  timeframes: Map<string, TimeframeData>;
  openTrade: PaperTrade | null;
  trades: PaperTrade[];
  cooldownUntil: number;
  consecutiveLosses: number;        // Track for scraper timeout
  consecutiveWins: number;          // Track for overconfidence timeout
  previousRegime: 'MOMENTUM' | 'RANGE';  // For hysteresis - no NONE zone
  stats: {
    totalTrades: number;
    wins: number;
    losses: number;
    totalPnl: number;
    winRate: number;
  };
  orderBookDepth: OrderBookDepth | null;
  lastOFIUpdate: number;
}

// ═══════════════════════════════════════════════════════════════
// COIN TRADER CLASS
// ═══════════════════════════════════════════════════════════════

class CoinTrader {
  public state: CoinTradingState;
  public lgbmPredictor: LightGBMPredictor;
  public useLightGBM: boolean = false;
  private stateFile: string;

  constructor(symbol: typeof SYMBOLS[number]) {
    this.stateFile = path.join(process.cwd(), 'data', 'paper-trades-scalp', `${symbol}.json`);
    this.lgbmPredictor = new LightGBMPredictor();

    this.state = {
      symbol,
      balance: CONFIG.virtualBalancePerCoin,
      timeframes: new Map(),
      openTrade: null,
      trades: [],
      cooldownUntil: 0,
      consecutiveLosses: 0,
      consecutiveWins: 0,
      previousRegime: 'RANGE',  // Default to RANGE (safer start)
      stats: {
        totalTrades: 0,
        wins: 0,
        losses: 0,
        totalPnl: 0,
        winRate: 0,
      },
      orderBookDepth: null,
      lastOFIUpdate: 0,
    };

    // Initialize timeframes
    for (const interval of CONFIG.intervals) {
      this.state.timeframes.set(interval, {
        candles: [],
        momentum: analyzeMomentum([]),
        lastUpdate: 0,
        orderBookDepth: null,
        lastOFIUpdate: 0,
      });
    }
  }

  loadState(): void {
    try {
      if (fs.existsSync(this.stateFile)) {
        const saved = JSON.parse(fs.readFileSync(this.stateFile, 'utf-8'));
        this.state.balance = saved.balance || CONFIG.virtualBalancePerCoin;
        this.state.openTrade = saved.openTrade || null;
        this.state.trades = saved.trades || [];
        this.state.cooldownUntil = saved.cooldownUntil || 0;
        this.state.consecutiveLosses = saved.consecutiveLosses || 0;
        this.state.consecutiveWins = saved.consecutiveWins || 0;
        this.state.previousRegime = saved.previousRegime || 'RANGE';  // Backward compat
        this.state.stats = saved.stats || this.state.stats;

        // 🔧 FIX: Migrate MOMENTUM mode trades with wrong BB-based TPs
        if (this.state.openTrade?.isMomentumMode) {
          const trade = this.state.openTrade;
          const isLong = trade.direction === 'LONG';
          const stopDistance = Math.abs(trade.entryPrice - trade.originalStopLoss);
          // Recalculate R-based TPs for MOMENTUM mode
          trade.takeProfit1 = isLong
            ? trade.entryPrice + stopDistance * 1.5
            : trade.entryPrice - stopDistance * 1.5;
          trade.takeProfit2 = isLong
            ? trade.entryPrice + stopDistance * 3.0
            : trade.entryPrice - stopDistance * 3.0;
          trade.tp1DistanceR = stopDistance * 1.5;
          console.log(`   🔧 ${this.state.symbol}: Fixed MOMENTUM TPs (was using old BB values)`);
        }
      }
    } catch (e) {
      console.log(`  ${this.state.symbol}: Fresh start (no saved state)`);
    }
  }

  saveState(): void {
    try {
      const dir = path.dirname(this.stateFile);
      if (!fs.existsSync(dir)) {
        fs.mkdirSync(dir, { recursive: true });
      }

      const toSave = {
        symbol: this.state.symbol,
        balance: this.state.balance,
        openTrade: this.state.openTrade,
        trades: this.state.trades.slice(-100),  // Keep last 100
        cooldownUntil: this.state.cooldownUntil,
        consecutiveLosses: this.state.consecutiveLosses,
        consecutiveWins: this.state.consecutiveWins,
        previousRegime: this.state.previousRegime,  // Save for hysteresis
        stats: this.state.stats,
        savedAt: new Date().toISOString(),
      };

      fs.writeFileSync(this.stateFile, JSON.stringify(toSave, null, 2));
    } catch (e) {
      console.error(`  ${this.state.symbol}: Failed to save state:`, e);
    }
  }

  resetState(): void {
    // Backup current state before reset
    if (fs.existsSync(this.stateFile)) {
      const backupFile = this.stateFile.replace('.json', '-backup.json');
      fs.copyFileSync(this.stateFile, backupFile);
      console.log(`  ${this.state.symbol}: State backed up to ${path.basename(backupFile)}`);
    }

    // Initialize new timeframes Map with all intervals
    const newTimeframes = new Map();
    for (const interval of CONFIG.intervals) {
      newTimeframes.set(interval, {
        candles: [],
        momentum: analyzeMomentum([]),
        lastUpdate: 0,
        orderBookDepth: null,
        lastOFIUpdate: 0,
      });
    }

    // Reset to initial state
    this.state = {
      symbol: this.state.symbol,
      balance: CONFIG.virtualBalancePerCoin,
      openTrade: null,
      trades: [],
      cooldownUntil: 0,
      consecutiveLosses: 0,
      consecutiveWins: 0,
      previousRegime: 'RANGE',  // Reset to RANGE
      stats: {
        totalTrades: 0,
        wins: 0,
        losses: 0,
        totalPnl: 0,
        winRate: 0,
      },
      timeframes: newTimeframes,
      orderBookDepth: null,
      lastOFIUpdate: 0,
    };
    this.saveState();
    console.log(`  ${this.state.symbol}: State reset to $${CONFIG.virtualBalancePerCoin} balance`);
  }

  async initialize(client: any): Promise<void> {
    this.useLightGBM = this.lgbmPredictor.load();
    this.loadState();

    // Fetch initial candles
    for (const interval of CONFIG.intervals) {
      await this.fetchCandles(client, interval);
    }
  }

  async fetchCandles(client: any, interval: string): Promise<void> {
    const tf = this.state.timeframes.get(interval);
    if (!tf) return;

    const now = Date.now();
    const refreshMs = CONFIG.refreshMsByInterval[interval] || 15000;

    if (now - tf.lastUpdate < refreshMs) return;

    try {
      const rawCandles = await client.candles({
        symbol: this.state.symbol,
        interval,
        limit: 100,
      });

      tf.candles = rawCandles.map((c: any) => ({
        timestamp: c.openTime,
        open: parseFloat(c.open),
        high: parseFloat(c.high),
        low: parseFloat(c.low),
        close: parseFloat(c.close),
        volume: parseFloat(c.volume),
      }));

      // Calculate OFI if we have order book depth
      const ofiSignal = tf.orderBookDepth ? calculateOFI(tf.orderBookDepth) : undefined;
      tf.momentum = analyzeMomentum(tf.candles, ofiSignal);

      // Run liquidity detection on primary timeframe only
      if (interval === CONFIG.primaryInterval && tf.candles.length >= 20) {
        tf.momentum.liquiditySignals = detectLiquiditySignals(tf.candles as any[], SCALP_CONFIG);
      }

      tf.lastUpdate = now;
    } catch (e: any) {
      if (!e.message?.includes('ENOTFOUND')) {
        console.error(`  ${this.state.symbol}/${interval}: Fetch error:`, e.message);
      }
    }
  }

  // Fetch and update Order Flow Imbalance (every 15 seconds to avoid rate limits)
  async updateOFI(client: any): Promise<OFISignal | null> {
    const now = Date.now();
    const ofiUpdateInterval = 15000; // 15 seconds
    const primary = this.state.timeframes.get(CONFIG.primaryInterval);

    if (!primary) return null;
    if (now - primary.lastOFIUpdate < ofiUpdateInterval) {
      // Use cached OFI if available
      if (primary.orderBookDepth) {
        return calculateOFI(primary.orderBookDepth);
      }
      return null;
    }

    try {
      const depth = await fetchOrderBookDepth(client, this.state.symbol);
      if (depth) {
        primary.orderBookDepth = depth;
        primary.lastOFIUpdate = now;
        return calculateOFI(depth);
      }
    } catch (e: any) {
      // Silently fail - OFI is optional enhancement
    }

    return null;
  }

  async tick(client: any): Promise<{ status: string; details: string }> {
    // Fetch candles for all timeframes
    for (const interval of CONFIG.intervals) {
      await this.fetchCandles(client, interval);
    }

    const primary = this.state.timeframes.get(CONFIG.primaryInterval);
    if (!primary || primary.candles.length < CONFIG.minCandlesRequired) {
      return { status: 'WAIT', details: 'Insufficient data' };
    }

    const lastCandle = primary.candles[primary.candles.length - 1];
    const currentPrice = lastCandle.close;
    const candleLow = lastCandle.low;
    const candleHigh = lastCandle.high;
    const candleTime = lastCandle.timestamp;

    // Check open trade
    if (this.state.openTrade) {
      const result = this.checkOpenTrade(currentPrice, candleLow, candleHigh, candleTime);
      if (result.closed) {
        return { status: 'CLOSED', details: result.message };
      }
      return { status: 'HOLDING', details: result.message };
    }

    // Check cooldown
    if (Date.now() < this.state.cooldownUntil) {
      const remaining = Math.round((this.state.cooldownUntil - Date.now()) / 1000);
      return { status: 'COOLDOWN', details: `${remaining}s remaining` };
    }

    // Update Order Flow Imbalance (market microstructure edge)
    await this.updateOFI(client);
    // Recalculate momentum with fresh OFI data
    const ofiSignal = primary.orderBookDepth ? calculateOFI(primary.orderBookDepth) : undefined;
    primary.momentum = analyzeMomentum(primary.candles, ofiSignal);

    // Analyze for entry
    const analysis = this.analyzeForEntry();

    if (analysis.shouldEnter && analysis.direction !== 'NEUTRAL') {
      this.enterTrade({ ...analysis, direction: analysis.direction as 'LONG' | 'SHORT' }, currentPrice);
      return { status: 'ENTERED', details: analysis.reason };
    }

    // Show rejection reason for debugging (first 80 chars)
    const rejectionReason = analysis.reason.length > 80 ? analysis.reason.slice(0, 77) + '...' : analysis.reason;
    return { status: 'SCAN', details: `${analysis.direction} (${analysis.signals.join(', ')}) [${rejectionReason}]` };
  }

  private analyzeForEntry(): {
    shouldEnter: boolean;
    direction: 'LONG' | 'SHORT' | 'NEUTRAL';
    strength: number;
    signals: string[];
    reason: string;
    mlPrediction: number;
    liquidityScore?: number;
  } {
    const tf5m = this.state.timeframes.get('5m');
    const tf1m = this.state.timeframes.get('1m');
    const tf15m = this.state.timeframes.get('15m');
    const tf4h = this.state.timeframes.get('4h');

    if (!tf5m || tf5m.candles.length < CONFIG.minCandlesRequired) {
      return { shouldEnter: false, direction: 'NEUTRAL', strength: 0, signals: [], reason: 'No data', mlPrediction: 0.5 };
    }

    const m5 = tf5m.momentum;
    const m1 = tf1m?.momentum;
    const m15 = tf15m?.momentum;
    const m4h = tf4h?.momentum;

    const signals: string[] = [];

    // Collect active signals
    if (m5.volumeSpike) signals.push(`VOL:${m5.volumeRatio.toFixed(1)}x`);
    if (m5.rsiBullishCross) signals.push('RSI↑50');
    if (m5.rsiBearishCross) signals.push('RSI↓50');
    if (m5.rsiOversold) signals.push('RSI<30');
    if (m5.rsiOverbought) signals.push('RSI>70');
    if (m5.emaBullishCross) signals.push('EMA↑');
    if (m5.emaBearishCross) signals.push('EMA↓');
    if (m5.bbBreakoutUp) signals.push('BB↑');
    if (m5.bbBreakoutDown) signals.push('BB↓');
    if (m5.priceBreakoutUp) signals.push('BRK↑');
    if (m5.priceBreakoutDown) signals.push('BRK↓');
    if (m5.candleMomentum !== 'neutral') signals.push(`CANDLE:${m5.candleMomentum[0].toUpperCase()}`);

    // Check direction (RANGE mode can override later via BB position)
    const direction = m5.direction;
    const regime = m5.regime;

    // Only block NEUTRAL in TREND mode - RANGE derives direction from BB position
    if (direction === 'NEUTRAL' && regime !== 'RANGE') {
      return { shouldEnter: false, direction: 'NEUTRAL', strength: 0, signals, reason: 'No clear direction', mlPrediction: 0.5 };
    }

    // ═══════════════════════════════════════════════════════════════
    // 4H TREND FILTER - SOFT FILTER (reduces size, doesn't block)
    // Counter-trend trades allowed at 50% position size for data collection
    // ═══════════════════════════════════════════════════════════════
    const isLong = direction === 'LONG';
    let trendMultiplier = 1.0;  // Position size multiplier

    if (m4h) {
      const trend4h = m4h.emaAligned; // 'bullish', 'bearish', or 'neutral'

      // Counter-trend: Allow but reduce position size by 50%
      const counterTrend = (trend4h === 'bullish' && !isLong) || (trend4h === 'bearish' && isLong);

      if (counterTrend) {
        trendMultiplier = 0.5;  // Half size for counter-trend
        signals.push('4h↔');    // Counter-trend warning
      } else if (trend4h === 'bullish') {
        signals.push('4h↑');
      } else if (trend4h === 'bearish') {
        signals.push('4h↓');
      } else {
        signals.push('4h~');
      }
    }

    // ═══════════════════════════════════════════════════════════════
    // 15M TREND FILTER - CRITICAL FOR AVOIDING FALLING KNIVES
    // ═══════════════════════════════════════════════════════════════
    const m15Aligned = m15 && m15.emaAligned === (isLong ? 'bullish' : 'bearish');
    const m15Against = m15 && m15.emaAligned === (isLong ? 'bearish' : 'bullish');

    if (m15Against) {
      // 15m trend is AGAINST us - need STRONG reversal confirmation
      // Require: candle moving in our direction + volume spike
      const candleConfirm = m5.candleMomentum === (isLong ? 'bullish' : 'bearish');
      if (!candleConfirm) {
        return {
          shouldEnter: false,
          direction,
          strength: 0,
          signals,
          reason: `15m against (${m15.emaAligned}) - need candle confirm`,
          mlPrediction: 0.5,
        };
      }
      signals.push('15m⚠');  // Warning: counter-trend
    } else if (m15Aligned) {
      signals.push('15m✓');
    }

    // Confirm with 1m (optional boost)
    let strengthBoost = 0;
    if (m1 && m1.direction === direction) {
      strengthBoost += 0.1;
      signals.push('1m✓');
    }

    // Apply trend multiplier for counter-trend trades (50% size reduction)
    let strength = Math.min(1, (m5.strength + strengthBoost) * trendMultiplier);

    // Need minimum signals (skip for RANGE - direction comes from BB position, not momentum)
    if (regime !== 'RANGE') {
      const signalCount = direction === 'LONG' ? m5.bullishSignals : m5.bearishSignals;
      if (signalCount < CONFIG.momentum.minSignals) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `Only ${signalCount}/${CONFIG.momentum.minSignals} signals`,
          mlPrediction: 0.5,
        };
      }
    }

    // ═══════════════════════════════════════════════════════════════
    // VWAP & KILL ZONE CONTEXT
    // ═══════════════════════════════════════════════════════════════
    const vwapConf = direction === 'LONG' ? m5.priceAboveVwap : !m5.priceAboveVwap;
    if (vwapConf) signals.push('VWAP✓');
    if (Math.abs(m5.vwapDeviationStd) > 1.5) signals.push(`VWAP:${m5.vwapDeviationStd.toFixed(1)}σ`);
    if (m5.isKillZone) signals.push(`KZ:${m5.killZone}`);

    // ═══════════════════════════════════════════════════════════════
    // REGIME: MOMENTUM vs RANGE vs NONE
    // ═══════════════════════════════════════════════════════════════
    const bbPos = m5.bbPosition;
    signals.push(`${regime}`);
    signals.push(`BB:${(bbPos * 100).toFixed(0)}%`);
    signals.push(`ADX:${m5.adx?.toFixed(0) || '?'}`);
    signals.push(m5.bbExpanding ? 'BB↑↑' : `BW:${m5.bbWidth?.toFixed(1)}%`);

    // NO NONE ZONE - always either MOMENTUM or RANGE (hysteresis handles transition)

    if (regime === 'MOMENTUM') {
      // ═══════════════════════════════════════════════════════════════
      // MOMENTUM MODE: Breakout entries (ADX > 25 + BB expanding)
      // Entry: Price breakout or BB breakout with volume
      // Exit: Trail at -1R from entry (let winners run, no fixed TP)
      // ═══════════════════════════════════════════════════════════════
      const isLong = direction === 'LONG';
      const currentPrice = tf5m.candles[tf5m.candles.length - 1].close;

      // 1. VWAP alignment (price above VWAP for longs, below for shorts)
      const hasVwapAlign = isLong ? m5.priceAboveVwap : !m5.priceAboveVwap;
      if (!hasVwapAlign) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `MOMENTUM: VWAP not aligned (${m5.priceAboveVwap ? 'above' : 'below'} VWAP)`,
          mlPrediction: 0.5,
        };
      }
      signals.push('VWAP✓');

      // 2. Structure break required (close above/below recent swing high/low)
      // NOT just BB breakout - need actual price structure break
      const hasStructureBreak = isLong
        ? m5.priceBreakoutUp   // Close > recent swing high
        : m5.priceBreakoutDown; // Close < recent swing low

      if (!hasStructureBreak) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `MOMENTUM: No structure break (need close above swing ${isLong ? 'high' : 'low'})`,
          mlPrediction: 0.5,
        };
      }
      signals.push(isLong ? 'SWING↑' : 'SWING↓');

      // 3. Volume spike REQUIRED (2x for MOMENTUM mode)
      const volumeThreshold = CONFIG.momentum.volumeSpikeMultipleMomentum;  // 2x
      if (m5.volumeRatio < volumeThreshold) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `MOMENTUM: Need volume > ${volumeThreshold}x (current:${m5.volumeRatio.toFixed(1)}x)`,
          mlPrediction: 0.5,
        };
      }
      signals.push(`VOL:${m5.volumeRatio.toFixed(1)}x`);

      // 4. MACD confirmation (bonus, not required)
      const hasMacdConfirm = isLong
        ? (m5.macdBullishCross || m5.macdHistogram > 0)
        : (m5.macdBearishCross || m5.macdHistogram < 0);
      if (hasMacdConfirm) signals.push('MACD✓');

      // ML prediction (optional)
      let mlPrediction = 0.5;
      if (this.useLightGBM) {
        try {
          const features: Record<string, any> = {
            bb_position: m5.bbPosition,
            bb_width: m5.bbUpper && m5.bbUpper > m5.bbLower ? ((m5.bbUpper - m5.bbLower) / currentPrice) * 100 : 0,
            rsi_value: m5.rsiValue,
            ema_fast: m5.emaFast,
            ema_slow: m5.emaSlow,
            ema_aligned: m5.emaAligned,
            macd_line: m5.macdLine,
            macd_signal: m5.macdSignal,
            macd_histogram: m5.macdHistogram,
            vwap: m5.vwap,
            vwap_deviation: m5.vwapDeviation,
            vwap_deviation_std: m5.vwapDeviationStd,
            price_above_vwap: m5.priceAboveVwap,
            atr_percent: m5.atrPercent,
            volatility: m5.atrPercent,
            volume_ratio: m5.volumeRatio,
            volume_spike: m5.volumeSpike ? 1 : 0,
            direction: direction === 'LONG' ? 'long' : 'short',
            regime: m5.regime,
          };

          const prediction = this.lgbmPredictor.predict(features as TradeFeatures);
          mlPrediction = prediction.winProbability;
        } catch {
          // ML prediction failed, proceed with default 0.5
        }
      }

      // ML filter (bypass if threshold = 0)
      if (CONFIG.minWinProbability > 0 && mlPrediction < CONFIG.minWinProbability) {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          mlPrediction,
          reason: `ML reject: ${(mlPrediction * 100).toFixed(0)}% < ${(CONFIG.minWinProbability * 100).toFixed(0)}%`
        };
      }

      if (CONFIG.minWinProbability > 0) {
        signals.push(`ML:${(mlPrediction * 100).toFixed(0)}%`);
      }

      return {
        shouldEnter: true,
        direction,
        strength: Math.min(1, strength + (hasMacdConfirm ? 0.1 : 0)),
        signals,
        reason: `MOMENTUM ${direction} (breakout+VOL+EMA)`,
        mlPrediction,
      };

    } else {
      // ═══════════════════════════════════════════════════════════════
      // RANGE MODE: Mean reversion (ADX < 20 + BB normal)
      // Entry: At BB extremes, look for reversal
      // Exit: TP1 = Middle BB (70%), TP2 = Opposite BB (30%)
      // ═══════════════════════════════════════════════════════════════
      // Override direction based on BB position for mean reversion
      let rangeDirection: 'LONG' | 'SHORT' | 'SKIP' = 'SKIP';
      if (bbPos < CONFIG.regime.rangeLongThreshold) {  // < 25%
        rangeDirection = 'LONG';
      } else if (bbPos > CONFIG.regime.rangeShortThreshold) {  // > 75%
        rangeDirection = 'SHORT';
      }

      if (rangeDirection === 'SKIP') {
        return {
          shouldEnter: false,
          direction,
          strength,
          signals,
          reason: `RANGE: BB in middle zone (${(bbPos * 100).toFixed(0)}%) - need <${CONFIG.regime.rangeLongThreshold * 100}% or >${CONFIG.regime.rangeShortThreshold * 100}%`,
          mlPrediction: 0.5,
        };
      }

      // ═══════════════════════════════════════════════════════════════
      // OFI FILTER - Order Flow Imbalance at BB extremes (optional)
      // Only filter if we have actual OFI data (not default 0)
      // ═══════════════════════════════════════════════════════════════
      const ofi = m5.ofi || 0;
      const hasOfiData = m5.ofi !== undefined && m5.ofi !== 0;
      const ofiThreshold = 0.3;

      if (hasOfiData) {
        if (rangeDirection === 'LONG' && ofi < ofiThreshold) {
          return {
            shouldEnter: false,
            direction: rangeDirection,
            strength,
            signals,
            reason: `RANGE: OFI ${ofi.toFixed(2)} < ${ofiThreshold} - no buying pressure at lower BB`,
            mlPrediction: 0.5,
          };
        }
        if (rangeDirection === 'SHORT' && ofi > -ofiThreshold) {
          return {
            shouldEnter: false,
            direction: rangeDirection,
            strength,
            signals,
            reason: `RANGE: OFI ${ofi.toFixed(2)} > -${ofiThreshold} - no selling pressure at upper BB`,
            mlPrediction: 0.5,
          };
        }
        signals.push(`OFI:${ofi > 0 ? '+' : ''}${ofi.toFixed(2)}`);
      }

      // ═══════════════════════════════════════════════════════════════
      // BAND WALK DETECTION - HARD FILTER
      // If price closed in extreme zone for 3+ candles = walking, not reversing
      // Walking = momentum continuing, NOT reversing - BLOCK ENTRY
      // ═══════════════════════════════════════════════════════════════
      const candles = tf5m.candles;
      if (candles.length >= 4) {
        if (rangeDirection === 'LONG') {
          // Check if walking down (last 3 candles closed near lows)
          const last3 = candles.slice(-3);
          const allNearLow = last3.every(c => {
            const range = c.high - c.low;
            const position = range > 0 ? (c.close - c.low) / range : 0.5;
            return position < 0.3; // Closed in bottom 30% of candle
          });
          if (allNearLow) {
            return {
              shouldEnter: false,
              direction: rangeDirection,
              strength,
              signals,
              reason: `RANGE: Band walk down (3 candles) - momentum continuing, not reversing`,
              mlPrediction: 0.5,
            };
          }
        } else if (rangeDirection === 'SHORT') {
          // Check if walking up (last 3 candles closed near highs)
          const last3 = candles.slice(-3);
          const allNearHigh = last3.every(c => {
            const range = c.high - c.low;
            const position = range > 0 ? (c.close - c.low) / range : 0.5;
            return position > 0.7; // Closed in top 30% of candle
          });
          if (allNearHigh) {
            return {
              shouldEnter: false,
              direction: rangeDirection,
              strength,
              signals,
              reason: `RANGE: Band walk up (3 candles) - momentum continuing, not reversing`,
              mlPrediction: 0.5,
            };
          }
        }
      }
      signals.push('NO_WALK✓');

      // Use mean reversion direction instead of momentum direction
      const direction_override = rangeDirection as 'LONG' | 'SHORT';
      const isLong = direction_override === 'LONG';

      // ═══════════════════════════════════════════════════════════════
      // CANDLE CONFIRMATION - HARD FILTER
      // Mean reversion needs reversal candle (engulfing/bullish/bearish)
      // No confirmation = no entry
      // ═══════════════════════════════════════════════════════════════
      const candleTurning = isLong
        ? (m5.candleMomentum === 'bullish')
        : (m5.candleMomentum === 'bearish');

      if (!candleTurning) {
        return {
          shouldEnter: false,
          direction: direction_override,
          strength,
          signals,
          reason: `RANGE: No candle confirmation - need ${isLong ? 'bullish' : 'bearish'} candle`,
          mlPrediction: 0.5,
        };
      }
      signals.push('CANDLE✓');
      signals.push(`BB:${(bbPos * 100).toFixed(0)}%`);

      // ═══════════════════════════════════════════════════════════════
      // VOLUME - HARD FILTER
      // No volume = no conviction = no entry
      // ═══════════════════════════════════════════════════════════════
      const volumeThreshold = CONFIG.momentum.volumeSpikeMultipleRange;  // 1.2x now
      const isExhaustion = m5.volumeRatio > 4.0;  // 4x+ = exhaustion spike (was 3x - 3x is often the reversal we want)

      if (isExhaustion) {
        return {
          shouldEnter: false,
          direction: direction_override,
          strength,
          signals,
          reason: `RANGE: Volume exhaustion (${m5.volumeRatio.toFixed(1)}x > 3x) - skip`,
          mlPrediction: 0.5,
        };
      }

      if (m5.volumeRatio < volumeThreshold) {
        return {
          shouldEnter: false,
          direction: direction_override,
          strength,
          signals,
          reason: `RANGE: Volume ${m5.volumeRatio.toFixed(1)}x < ${volumeThreshold}x required - no conviction`,
          mlPrediction: 0.5,
        };
      }
      signals.push(`VOL:${m5.volumeRatio.toFixed(1)}x✓`);

      // ═══════════════════════════════════════════════════════════════
      // VWAP EXTENSION CHECK - HARD FILTER
      // Price must be extended from VWAP for mean reversion
      // ═══════════════════════════════════════════════════════════════
      const vwapExtension = Math.abs(m5.vwapDeviation);
      if (vwapExtension < 0.2) {  // Less than 0.2% from VWAP = not extended (was 0.1)
        return {
          shouldEnter: false,
          direction: direction_override,
          strength,
          signals,
          reason: `RANGE: VWAP not extended (${vwapExtension.toFixed(2)}%) - need > 0.1%`,
          mlPrediction: 0.5,
        };
      }
      signals.push(`VWAP:${vwapExtension.toFixed(2)}%✓`);

      // Williams %R confirmation (bonus signal)
      const williamsAtExtreme = isLong
        ? (m5.williamsROversold || m5.williamsR < -80)
        : (m5.williamsROverbought || m5.williamsR > -20);
      if (williamsAtExtreme) signals.push('W%R✓');

      // ═══════════════════════════════════════════════════════════════
      // R:R FILTER for RANGE mode - HARD FILTER
      // Stop too far = bad R:R = skip
      // ═══════════════════════════════════════════════════════════════
      const currentPrice = tf5m.candles[tf5m.candles.length - 1].close;
      const atrStopDistance = (m5.atr || currentPrice * 0.005) * 1.5;
      const maxStopDistance = currentPrice * 0.01;  // 1% max (matches TP1)
      const estimatedStopDistance = Math.min(atrStopDistance, maxStopDistance);
      const stopPct = (estimatedStopDistance / currentPrice) * 100;

      if (stopPct > 1.0) {
        return {
          shouldEnter: false,
          direction: direction_override,
          strength,
          signals,
          reason: `RANGE: Stop too far (${stopPct.toFixed(2)}% > 1%) - bad R:R`,
          mlPrediction: 0.5,
        };
      }

      // ═══════════════════════════════════════════════════════════════
      // LIQUIDITY CONFIRMATION - Stop hunt / fakeout detection
      // Swept level + closed back = much higher probability entry
      // ═══════════════════════════════════════════════════════════════
      const liquidity = m5.liquiditySignals;
      let liquidityScore = 0;

      if (liquidity) {
        // If signals conflict with direction → skip
        if (liquidity.bestDirection !== null && liquidity.bestDirection !== direction_override) {
          return {
            shouldEnter: false,
            direction: direction_override,
            strength,
            signals,
            reason: `RANGE: Liquidity signals oppose direction (${liquidity.signalTags.join(',')})`,
            mlPrediction: 0.5,
          };
        }

        // Push all signal tags for logging
        if (liquidity.signalTags.length > 0) {
          signals.push(...liquidity.signalTags);
        }

        liquidityScore = liquidity.liquidityScore;

        // Optional hard gate - start at 0 to collect data
        const MIN_LIQUIDITY_SCORE = 0;
        if (liquidityScore < MIN_LIQUIDITY_SCORE) {
          return {
            shouldEnter: false,
            direction: direction_override,
            strength,
            signals,
            reason: `RANGE: Liquidity score too low (${liquidityScore} < ${MIN_LIQUIDITY_SCORE})`,
            mlPrediction: 0.5,
          };
        }
      }

      // ALL HARD FILTERS PASSED - Full strength entry
      const rangeStrength = Math.min(1, strength + (williamsAtExtreme ? 0.15 : 0));

      const bbZone = bbPos < CONFIG.regime.rangeLongThreshold ? 'lower' : 'upper';
      return {
        shouldEnter: true,
        direction: direction_override,
        strength: rangeStrength,
        signals,
        reason: `RANGE ${direction_override} @ BB ${bbZone}+VOL${williamsAtExtreme ? '+W%R' : ''}${candleTurning ? '+CANDLE' : ''}`,
        mlPrediction: 0.5,
        liquidityScore,
      };
    }
  }

  private enterTrade(analysis: { direction: 'LONG' | 'SHORT'; strength: number; signals: string[]; liquidityScore?: number }, currentPrice: number): void {
    const isLong = analysis.direction === 'LONG';
    const tf5m = this.state.timeframes.get('5m');
    const tf15m = this.state.timeframes.get('15m');
    const momentum5m = tf5m?.momentum;
    const momentum15m = tf15m?.momentum;

    // ═══════════════════════════════════════════════════════════════
    // ATR-BASED STOPS: Adaptive to volatility, with structure fallback
    // ATR automatically adjusts per coin and market conditions
    // ═══════════════════════════════════════════════════════════════
    const atr5m = momentum5m?.atr || 0;
    const atrPercent = momentum5m?.atrPercent || 0;

    // Primary: ATR-based stop (1.5x ATR gives room for noise)
    // Example: If ATR is 0.4%, SL = 0.6% (adapts to volatility)
    let stopDistance = atr5m * 1.5;
    let stopLoss = isLong ? currentPrice - stopDistance : currentPrice + stopDistance;

    // Safety: Cap ATR stop at max 1.0% (matches TP1 for 1:1 R:R minimum)
    const maxStopPct = 1.0;
    const stopPct = (stopDistance / currentPrice) * 100;
    if (stopPct > maxStopPct) {
      stopDistance = currentPrice * (maxStopPct / 100);
      stopLoss = isLong ? currentPrice - stopDistance : currentPrice + stopDistance;
    }

    // Fallback: Use structure-based stop if available and reasonable
    const swingHigh = (momentum15m?.swingHigh && momentum15m.swingHigh > currentPrice)
      ? momentum15m.swingHigh
      : (momentum5m?.swingHigh && momentum5m.swingHigh > currentPrice)
        ? momentum5m.swingHigh
        : null;

    const swingLow = (momentum15m?.swingLow && momentum15m.swingLow < currentPrice)
      ? momentum15m.swingLow
      : (momentum5m?.swingLow && momentum5m.swingLow < currentPrice)
        ? momentum5m.swingLow
        : null;

    // Try structure-based stop if swing point available
    let structureStopDistance: number | undefined;
    if (isLong && swingLow) {
      // LONG: Stop below recent swing low (0.3% buffer for 5m volatility)
      const structureStop = swingLow * 0.997;
      structureStopDistance = currentPrice - structureStop;
    } else if (!isLong && swingHigh) {
      // SHORT: Stop above recent swing high (0.3% buffer)
      const structureStop = swingHigh * 1.003;
      structureStopDistance = structureStop - currentPrice;
    }

    // Use structure stop if it's tighter than ATR stop (more conservative)
    if (structureStopDistance !== undefined && structureStopDistance < stopDistance) {
      const riskPct = (structureStopDistance / currentPrice) * 100;
      if (riskPct >= 0.2 && riskPct <= 2.0) {
        // Structure stop is tighter and reasonable - use it
        stopDistance = structureStopDistance;
        stopLoss = isLong ? currentPrice - stopDistance : currentPrice + stopDistance;
      }
    }

    // Log stop type for debugging
    const finalStopPct = (stopDistance / currentPrice) * 100;
    const stopType = (structureStopDistance !== undefined && structureStopDistance < (atr5m * 1.5))
      ? 'STRUCTURE'
      : `ATR ${atrPercent.toFixed(2)}%`;
    console.log(`   ${this.state.symbol}: SL = ${finalStopPct.toFixed(2)}% (${stopType})`);


    // ═══════════════════════════════════════════════════════════════
    // REGIME-SPECIFIC TARGETS
    // RANGE mode: BB-based (mean reversion)
    // MOMENTUM mode: R-based (let winners run)
    // ═══════════════════════════════════════════════════════════════

    // Get BB values from 5m momentum (for RANGE mode)
    const bbMiddle = momentum5m?.bbMiddle || currentPrice;
    const bbUpper = momentum5m?.bbUpper || currentPrice * 1.01;
    const bbLower = momentum5m?.bbLower || currentPrice * 0.99;

    // Determine regime for TP calculation
    const isMomentumMode = momentum5m?.regime === 'MOMENTUM';

    let takeProfit1: number;
    let takeProfit2: number;
    let tp1DistanceR: number;

    if (isMomentumMode) {
      // MOMENTUM mode: R-based targets (breakout trade)
      // TP1 = 1.5R, TP2 = 3R - let trends run
      takeProfit1 = isLong ? currentPrice + stopDistance * 1.5 : currentPrice - stopDistance * 1.5;
      takeProfit2 = isLong ? currentPrice + stopDistance * 3.0 : currentPrice - stopDistance * 3.0;
      tp1DistanceR = stopDistance * 1.5;  // R = 1.5 * stop distance
    } else {
      // RANGE mode: BB-based targets (mean reversion)
      // TP1 = Middle BB, TP2 = Opposite BB
      takeProfit1 = bbMiddle;
      takeProfit2 = isLong ? bbUpper : bbLower;
      tp1DistanceR = Math.abs(takeProfit1 - currentPrice);
    }

    // ═══════════════════════════════════════════════════════════════
    // KELLY CRITERION POSITION SIZING
    // f* = (bp - q) / b where b = R:R, p = win prob, q = 1-p
    // ═══════════════════════════════════════════════════════════════

    // Calculate R:R based on actual stop and TP1 distance
    const rRatio = tp1DistanceR / stopDistance;  // R:R ratio

    // Use ML prediction as win probability, fallback to conservative 0.4
    const mlWinProb = 0.4;  // Conservative fallback since we don't have ML in this context
    const p = mlWinProb;
    const q = 1 - p;

    // Kelly fraction: (bp - q) / b
    // If b = 0.75 (TP1 = 0.75R), p = 0.4, q = 0.6:
    // f* = (0.75*0.4 - 0.6) / 0.75 = (0.3 - 0.6) / 0.75 = -0.3/0.75 = -0.4 (negative!)
    // Negative Kelly means don't trade. We need a minimum R:R of 1.5:1 for p=0.4
    // Let's use a simpler approach: Kelly-based risk multiplier

    // Simplified Kelly: risk = base_risk * kelly_multiplier
    // where kelly_multiplier is based on R:R and confidence
    let kellyMultiplier = 1.0;

    if (rRatio >= 1.5) {
      kellyMultiplier = 1.5;  // Good R:R, increase size
    } else if (rRatio >= 1.0) {
      kellyMultiplier = 1.0;  // Standard R:R
    } else if (rRatio >= 0.75) {
      kellyMultiplier = 0.7;  // Poor R:R, reduce size
    } else {
      kellyMultiplier = 0.5;  // Bad R:R, minimum size
    }

    // Combine with signal strength
    const baseRiskPct = 0.5;  // 0.5% base risk
    const strengthMultiplier = 0.5 + analysis.strength * 1.0;  // 0.5x to 1.5x based on strength
    let kellyRiskPct = baseRiskPct * kellyMultiplier * strengthMultiplier;

    // ─────────────────────────────────────────────────────────────────
    // FIX: Liquidity → Priority, NOT Size (prevent concentration risk)
    // High liquidity = take trade, Low/No liquidity = still trade, just standard size
    // NEVER size UP - cap at 1.0x
    // NOTE: Removed the hard gate - liquidity is optional bonus, not requirement
    // ─────────────────────────────────────────────────────────────────
    const liqScore = analysis.liquidityScore ?? 0;
    const liqMultiplier =
      liqScore >= 75 ? 1.0 :    // Best setups = standard size (was 1.5x)
      liqScore >= 50 ? 1.0 :   // Good setups = standard size (was 1.25x)
      liqScore >= 25 ? 0.9 :   // Basic confirmation = slight reduction
      0.85;                     // No confirmation = slightly reduced (not blocked!)

    // Log liquidity score but don't skip
    if (liqScore > 0) {
      console.log(`   💧 ${this.state.symbol}: Liquidity score ${liqScore}/100 (${liqMultiplier}x size)`);
    }

    kellyRiskPct *= liqMultiplier;

    // ─────────────────────────────────────────────────────────────────
    // FIX: Portfolio regime GATES weak setups, doesn't just shrink size
    // ─────────────────────────────────────────────────────────────────
    const coinAdx = tf5m?.momentum?.adx || 0;
    const coinRegime = tf5m?.momentum?.regime || 'RANGE';  // Default to RANGE (no NONE zone)

    // Gate weak setups in hostile markets
    if (regimeDetector.shouldGateCoin(coinAdx, coinRegime)) {
      console.log(`   ⏭️ ${this.state.symbol}: Gated by portfolio regime (${currentPortfolioRegime.regime}) - coin ADX=${coinAdx.toFixed(1)}, regime=${coinRegime}`);
      return;
    }

    // Apply modest size reduction for strong setups in weak markets (not punitive)
    kellyRiskPct *= currentPortfolioRegime.sizeMultiplier;

    const riskAmount = this.state.balance * (kellyRiskPct / 100);
    let positionSize = riskAmount / stopDistance;

    // Dynamic leverage based on signal strength: 1x (weak) to 4x (strong)
    const dynamicLeverage = 1 + Math.floor(analysis.strength * 3);  // 0-33%=1x, 34-66%=2x, 67-100%=3x, 100%=4x
    const maxNotional = this.state.balance * dynamicLeverage;
    const uncappedNotional = currentPrice * positionSize;

    // Debug: Log position sizing (helpful for troubleshooting leverage issues)
    const effectiveLeverage = uncappedNotional / this.state.balance;
    if (effectiveLeverage > 5) {
      console.log(`  ⚠️ ${this.state.symbol}: HIGH LEVERAGE WARNING!`);
      console.log(`     Balance: $${this.state.balance.toFixed(2)}, Strength: ${(analysis.strength * 100).toFixed(0)}%`);
      console.log(`     Uncapped: $${uncappedNotional.toFixed(2)} (${effectiveLeverage.toFixed(1)}x), Max: $${maxNotional.toFixed(2)} (${dynamicLeverage}x)`);
    }

    if (uncappedNotional > maxNotional) {
      positionSize = maxNotional / currentPrice;
      console.log(`  ${this.state.symbol}: Position capped to ${dynamicLeverage}x leverage (strength: ${(analysis.strength * 100).toFixed(0)}%)`);
    }

    // Apply slippage
    const slippage = currentPrice * (CONFIG.slippageBps / 10000);
    const fillPrice = isLong ? currentPrice + slippage : currentPrice - slippage;

    // Track entry fee (will be deducted from PnL at exit, NOT from balance now)
    const notional = fillPrice * positionSize;
    const entryFee = notional * CONFIG.takerFeeRate;

    // Determine if MOMENTUM mode (breakout trade with trailing only)
    // Note: isMomentumMode already defined above for TP calculation

    const trade: PaperTrade = {
      id: `${this.state.symbol}-${Date.now()}`,
      symbol: this.state.symbol,
      direction: analysis.direction,
      entryPrice: fillPrice,
      entryTime: Date.now(),
      stopLoss,
      originalStopLoss: stopLoss,
      tp1DistanceR,  // Calculated above based on regime
      takeProfit1,
      takeProfit2,
      isMomentumMode,              // MOMENTUM = trail only, RANGE = TP1/TP2
      trailingStop: null,
      originalPositionSize: positionSize,
      currentPositionSize: positionSize,
      candlesHeld: 0,            // Track for time stop
      lastCandleTime: 0,         // Track for incrementing candlesHeld on candle close
      tp1Hit: isMomentumMode,    // MOMENTUM: skip TP1 logic entirely
      tp2Hit: isMomentumMode,    // MOMENTUM: skip TP2 logic entirely
      phase2Active: false,       // Phase 2 trailing active
      status: 'OPEN',
      feesPaid: entryFee,
      momentumStrength: analysis.strength,
      signals: analysis.signals,
      entryFeatures: momentum5m ? captureSnapshot(momentum5m, currentPrice) : undefined,
    };

    this.state.openTrade = trade;
    // DON'T push to trades array here - only push when closed
    this.saveState();

    const usedSwing = isLong ? swingLow : swingHigh;
    const swingSource = usedSwing
      ? `STRUCTURE ${momentum15m?.swingHigh || momentum15m?.swingLow ? '15m' : '5m'} (swing ${isLong ? 'low' : 'high'})`
      : 'FIXED %';

    // Calculate final risk % for logging
    const riskPct = (stopDistance / currentPrice) * 100;

    if (isMomentumMode) {
      // MOMENTUM mode: Trail only, no fixed TPs
      console.log(`\n🚀 ${this.state.symbol}: MOMENTUM ${trade.direction} [${swingSource}]`);
      console.log(`   Entry: $${fillPrice.toFixed(4)} | SL: $${stopLoss.toFixed(4)} (${riskPct.toFixed(2)}% risk)`);
      console.log(`   Mode: TRAIL ONLY - Trail at -${CONFIG.targets.momentumTrailR}R after ${CONFIG.targets.momentumTrailAfterR}R profit`);
      console.log(`   Target: +${trade.tp1DistanceR.toFixed(4)}R (1.5x stop) | Time stop: ${CONFIG.targets.timeStopCandles} candles (0.5R movement)`);
    } else {
      // RANGE mode: Mean reversion with TP1/TP2
      const tp1Pct = ((takeProfit1 - fillPrice) / fillPrice * 100).toFixed(2);
      const tp2Pct = Math.abs(((takeProfit2 - fillPrice) / fillPrice * 100)).toFixed(2);
      console.log(`\n⚡ ${this.state.symbol}: RANGE ${trade.direction} [${swingSource}]`);
      console.log(`   Entry: $${fillPrice.toFixed(4)} | SL: $${stopLoss.toFixed(4)} (${riskPct.toFixed(2)}% risk)`);
      console.log(`   TP1: $${takeProfit1.toFixed(4)} (Middle BB ${tp1Pct}%) | TP2: $${takeProfit2.toFixed(4)} (Opposite BB ${tp2Pct}%)`);
      console.log(`   Split: ${(CONFIG.targets.tp1ClosePct * 100).toFixed(0)}% at TP1, ${(CONFIG.targets.tp2ClosePct * 100).toFixed(0)}% at TP2 | Time stop: ${CONFIG.targets.timeStopCandles} candles (0.5R movement required)`);
    }
    console.log(`   Signals: ${analysis.signals.join(', ')}`);
    console.log(`   Strength: ${(analysis.strength * 100).toFixed(0)}% | Size: ${positionSize.toFixed(4)}`);
    if (liqScore > 0) {
      const liqTag = liqScore >= 75 ? '🔥' : liqScore >= 50 ? '✓' : liqScore >= 25 ? '·' : '⚠';
      console.log(`   Liquidity: ${liqScore}/100 ${liqTag} (${liqMultiplier}x size, capped)`);
    }
  }

  private checkOpenTrade(currentPrice: number, candleLow: number = currentPrice, candleHigh: number = currentPrice, currentCandleTime: number = 0): { closed: boolean; message: string } {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    // Increment candle counter only when candle actually changes (not every check cycle)
    // currentCandleTime is the timestamp of the current 5m candle
    if (currentCandleTime > 0 && trade.lastCandleTime !== currentCandleTime) {
      trade.candlesHeld++;
      trade.lastCandleTime = currentCandleTime;
    }

    const priceDiff = isLong ? currentPrice - trade.entryPrice : trade.entryPrice - currentPrice;
    const pnlPercent = (priceDiff / trade.entryPrice) * 100;
    const pnl = priceDiff * trade.currentPositionSize;

    // Check if position is flat (all TPs hit)
    if (trade.currentPositionSize <= 0.0000001) {
      trade.status = 'CLOSED';
      this.state.openTrade = null;
      this.saveState();
      return { closed: true, message: 'Position closed (flat)' };
    }

    // ═══════════════════════════════════════════════════════════════
    // TREND REVERSAL FLASH CLOSE - Lock profits when trend turns
    // ═══════════════════════════════════════════════════════════════
    const tf4h = this.state.timeframes.get('4h');
    const m4h = tf4h?.momentum;

    if (m4h && pnlPercent > 0.2) {  // Only flash close if profitable > 0.2%
      const trendReversed = isLong
        ? m4h.emaAligned === 'bearish'  // LONG but 4h turned bearish
        : m4h.emaAligned === 'bullish'; // SHORT but 4h turned bullish

      if (trendReversed) {
        return this.closeTrade(currentPrice, 'TREND_REVERSAL', pnl, pnlPercent);
      }
    }

    // ═══════════════════════════════════════════════════════════════
    // TIME STOP - Exit after N candles IF no movement
    // "No movement" = price within 0.5R of entry
    // ═══════════════════════════════════════════════════════════════
    if (trade.candlesHeld >= CONFIG.targets.timeStopCandles) {
      // Calculate R-based movement: how far from entry in R units
      const priceMovement = Math.abs(currentPrice - trade.entryPrice);
      const movementInR = priceMovement / trade.tp1DistanceR;  // R = tp1DistanceR

      // Only time out if "no movement" (less than 0.5R from entry)
      if (movementInR < 0.5) {
        return this.closeTrade(currentPrice, 'TIMEOUT', pnl, pnlPercent);
      }
      // If there's meaningful movement, let it run
    }

    // Also check minute-based timeout for very long holds
    const holdTime = Date.now() - trade.entryTime;
    const maxHoldMs = CONFIG.maxHoldMinutes * 60 * 1000;
    if (holdTime > maxHoldMs) {
      return this.closeTrade(currentPrice, 'TIMEOUT', pnl, pnlPercent);
    }

    // Use candle extremes for SL/TP checks to catch wicks
    // For LONG: SL triggers on low, TP triggers on high
    // For SHORT: SL triggers on high, TP triggers on low
    const slCheckPrice = isLong ? candleLow : candleHigh;
    const tpCheckPrice = isLong ? candleHigh : candleLow;

    // Check stop loss (use candle extreme)
    if ((isLong && slCheckPrice <= trade.stopLoss) || (!isLong && slCheckPrice >= trade.stopLoss)) {
      return this.closeTrade(trade.stopLoss, 'SL', pnl, pnlPercent);
    }

    // Check trailing stop (use candle extreme)
    if (trade.trailingStop !== null) {
      if ((isLong && slCheckPrice <= trade.trailingStop) || (!isLong && slCheckPrice >= trade.trailingStop)) {
        return this.closeTrade(trade.trailingStop, 'TRAILING', pnl, pnlPercent);
      }
    }

    // ═══════════════════════════════════════════════════════════════
    // MOMENTUM MODE: Trail at -1R from entry (no fixed TPs)
    // ═══════════════════════════════════════════════════════════════
    if (trade.isMomentumMode) {
      // Calculate profit in R (R = stop distance for momentum mode)
      const profitR = priceDiff / trade.tp1DistanceR;

      // Start trailing after momentumTrailAfterR (0.5R) profit
      if (profitR >= CONFIG.targets.momentumTrailAfterR) {
        if (!trade.phase2Active) {
          trade.phase2Active = true;
          console.log(`   ${trade.symbol}: MOMENTUM trailing ACTIVE at ${profitR.toFixed(2)}R profit`);
        }

        // Trail at momentumTrailR (1.0R) from current price
        const trailDistance = CONFIG.targets.momentumTrailR * trade.tp1DistanceR;
        const newTrailing = isLong ? currentPrice - trailDistance : currentPrice + trailDistance;

        // Only tighten (never loosen), AND must be better than current SL
        if (trade.trailingStop === null ||
            (isLong && newTrailing > trade.trailingStop) ||
            (!isLong && newTrailing < trade.trailingStop)) {
          trade.trailingStop = newTrailing;
        }
      }

      const trailInfo = trade.trailingStop ? ` TRAIL:$${trade.trailingStop.toFixed(4)}` : '';
      const candleInfo = ` C${trade.candlesHeld}/${CONFIG.targets.timeStopCandles}`;
      const profitRInfo = ` (${profitR.toFixed(2)}R)`;
      return {
        closed: false,
        message: `MOMENTUM${trade.phase2Active ? ' [TRAILING]' : ' [HOLD]'}${trailInfo}${candleInfo}${profitRInfo} | PnL: ${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%`
      };
    }

    // ═══════════════════════════════════════════════════════════════
    // RANGE MODE: TP1/TP2 mean reversion
    // TP1: Middle BB - Close 70%, Move SL to Entry + 0.2R
    // ═══════════════════════════════════════════════════════════════
    if (!trade.tp1Hit) {
      if ((isLong && tpCheckPrice >= trade.takeProfit1) || (!isLong && tpCheckPrice <= trade.takeProfit1)) {
        trade.tp1Hit = true;
        const closeAmount = trade.originalPositionSize * CONFIG.targets.tp1ClosePct;  // 60%
        // IMPORTANT: Close at TP1 price, not current price (wick may have retracted)
        this.closePartialTrade(trade.takeProfit1, closeAmount, 'TP1');

        // Move SL to Entry + 0.2R where R = (tp1 - entry)
        const protectedSlDistance = trade.tp1DistanceR * CONFIG.targets.protectedProfitR;  // 0.2R
        trade.stopLoss = isLong
          ? trade.entryPrice + protectedSlDistance
          : trade.entryPrice - protectedSlDistance;

        // Recalculate PnL after partial close for accurate display
        const newPriceDiff = isLong ? currentPrice - trade.entryPrice : trade.entryPrice - currentPrice;
        const newPnl = newPriceDiff * trade.currentPositionSize;
        const newPnlPercent = (newPriceDiff / trade.entryPrice) * 100;

        return { closed: false, message: `TP1 HIT (+${(CONFIG.targets.tp1ClosePct * 100).toFixed(0)}%) | SL→+${CONFIG.targets.protectedProfitR}R | PnL: ${newPnlPercent >= 0 ? '+' : ''}${newPnlPercent.toFixed(2)}%` };
      }
    }

    // ═══════════════════════════════════════════════════════════════
    // TP2: Opposite BB - Close remaining 30%
    // ═══════════════════════════════════════════════════════════════
    if (!trade.tp2Hit && trade.tp1Hit) {
      if ((isLong && tpCheckPrice >= trade.takeProfit2) || (!isLong && tpCheckPrice <= trade.takeProfit2)) {
        trade.tp2Hit = true;
        const closeAmount = trade.currentPositionSize; // Close remaining
        this.closePartialTrade(trade.takeProfit2, closeAmount, 'TP2');
        trade.status = 'CLOSED';
        this.state.openTrade = null;
        this.saveState();
        return { closed: true, message: `TP2 HIT (+30%) - Full exit` };
      }
    }

    // ═══════════════════════════════════════════════════════════════
    // PHASE 2 TRAILING - Activate when price near TP2
    // Trigger: current >= tp2 - 0.3 * (tp2 - entry) = at 70% of TP2 distance
    // Trail: current - 0.5 * (tp1 - entry) = -0.5R from current
    // ═══════════════════════════════════════════════════════════════
    if (trade.tp1Hit && !trade.tp2Hit) {
      // Calculate trigger price: tp2 - 0.3 * (tp2 - entry)
      const tp2Distance = Math.abs(trade.takeProfit2 - trade.entryPrice);
      const phase2TriggerPrice = isLong
        ? trade.takeProfit2 - 0.3 * tp2Distance
        : trade.takeProfit2 + 0.3 * tp2Distance;

      // Check if we should activate phase 2
      const shouldActivatePhase2 = isLong
        ? currentPrice >= phase2TriggerPrice
        : currentPrice <= phase2TriggerPrice;

      if (shouldActivatePhase2 && !trade.phase2Active) {
        trade.phase2Active = true;
        const tp2R = tp2Distance / trade.tp1DistanceR;
        console.log(`   ${trade.symbol}: Phase 2 trailing ACTIVE (TP2 at ${tp2R.toFixed(2)}R)`);
      }

      // Apply trailing when phase 2 is active
      if (trade.phase2Active) {
        // Trail at -0.5R from current price where R = (tp1 - entry)
        const trailDistance = CONFIG.targets.phase2TrailR * trade.tp1DistanceR;  // 0.5R
        const newTrailing = isLong ? currentPrice - trailDistance : currentPrice + trailDistance;

        // Only tighten (never loosen), AND must be better than current SL
        if (trade.trailingStop === null ||
            (isLong && newTrailing > trade.trailingStop) ||
            (!isLong && newTrailing < trade.trailingStop)) {
          trade.trailingStop = newTrailing;
        }
      }
    }

    const tpStatus = [trade.tp1Hit ? 'TP1' : '', trade.tp2Hit ? 'TP2' : ''].filter(Boolean).join('+') || 'HOLD';
    const phaseInfo = trade.phase2Active ? ' [P2-TRAIL]' : '';
    const trailInfo = trade.trailingStop ? ` TRAIL:$${trade.trailingStop.toFixed(4)}` : '';
    const candleInfo = ` C${trade.candlesHeld}/${CONFIG.targets.timeStopCandles}`;
    return {
      closed: false,
      message: `${tpStatus}${phaseInfo}${trailInfo}${candleInfo} | PnL: ${pnlPercent >= 0 ? '+' : ''}${pnlPercent.toFixed(2)}%`
    };
  }

  private closePartialTrade(exitPrice: number, closeAmount: number, reason: 'TP1' | 'TP2'): void {
    const trade = this.state.openTrade!;
    const isLong = trade.direction === 'LONG';

    // Apply slippage
    const slippage = exitPrice * (CONFIG.slippageBps / 10000);
    const fillPrice = isLong ? exitPrice - slippage : exitPrice + slippage;

    // Exit fee for partial
    const notional = fillPrice * closeAmount;
    const exitFee = notional * CONFIG.takerFeeRate;

    // Calculate PnL for this partial (gross profit/loss before fees)
    const priceDiff = isLong ? fillPrice - trade.entryPrice : trade.entryPrice - fillPrice;
    const grossPnl = priceDiff * closeAmount;
    const grossPnlPercent = (priceDiff / trade.entryPrice) * 100;

    // Account for entry fee proportionally based on % of position closing
    const entryFeeAllocation = (closeAmount / trade.originalPositionSize) *
      (trade.entryPrice * trade.originalPositionSize * CONFIG.takerFeeRate);

    // Net PnL = gross profit - entry fee portion - exit fee
    const netPnl = grossPnl - entryFeeAllocation - exitFee;

    // Update position
    trade.currentPositionSize -= closeAmount;
    trade.feesPaid += entryFeeAllocation + exitFee;  // Track both entry and exit fees

    // Update stats with net PnL
    this.state.balance += netPnl;
    this.state.stats.totalPnl += netPnl;
    this.state.stats.totalTrades++;
    if (netPnl > 0) {
      this.state.stats.wins++;
    } else {
      this.state.stats.losses++;
    }
    this.state.stats.winRate = this.state.stats.wins / Math.max(1, this.state.stats.totalTrades);

    const pnlSign = netPnl >= 0 ? '+' : '';
    console.log(`✅ ${trade.symbol}: ${reason} HIT | ${pnlSign}$${netPnl.toFixed(2)} (${pnlSign}${grossPnlPercent.toFixed(2)}%) | Remaining: ${(trade.currentPositionSize / trade.originalPositionSize * 100).toFixed(0)}%`);

    this.saveState();
  }

  private closeTrade(exitPrice: number, reason: PaperTrade['exitReason'], pnl: number, pnlPercent: number): { closed: boolean; message: string } {
    const trade = this.state.openTrade!;

    // Apply slippage on exit
    const slippage = exitPrice * (CONFIG.slippageBps / 10000);
    const fillPrice = trade.direction === 'LONG' ? exitPrice - slippage : exitPrice + slippage;

    // Exit fee
    const notional = fillPrice * trade.currentPositionSize;
    const exitFee = notional * CONFIG.takerFeeRate;

    // Final PnL calculation
    const isLong = trade.direction === 'LONG';
    const priceDiff = isLong ? fillPrice - trade.entryPrice : trade.entryPrice - fillPrice;
    const grossPnl = priceDiff * trade.currentPositionSize;

    // Account for remaining entry fee portion
    const remainingEntryFee = (trade.currentPositionSize / trade.originalPositionSize) *
      (trade.entryPrice * trade.originalPositionSize * CONFIG.takerFeeRate);

    // Net PnL = gross - remaining entry fee - exit fee
    const finalPnl = grossPnl - remainingEntryFee - exitFee;
    const finalPnlPercent = (priceDiff / trade.entryPrice) * 100;

    trade.exitPrice = fillPrice;
    trade.exitTime = Date.now();
    trade.exitReason = reason;
    trade.pnl = finalPnl;
    trade.pnlPercent = finalPnlPercent;
    trade.feesPaid += remainingEntryFee + exitFee;  // Track all fees
    trade.status = 'CLOSED';

    // Capture exit features for ML training
    const tf5m = this.state.timeframes.get('5m');
    if (tf5m?.momentum) {
      trade.exitFeatures = captureSnapshot(tf5m.momentum, fillPrice);
    }

    // Update balance and stats
    this.state.balance += finalPnl;
    this.state.stats.totalTrades++;
    this.state.stats.totalPnl += finalPnl;

    if (finalPnl > 0) {
      this.state.stats.wins++;
      // Track consecutive wins (for overconfidence check)
      this.state.consecutiveWins++;
      this.state.consecutiveLosses = 0;  // Reset loss streak
    } else {
      this.state.stats.losses++;
      // Track consecutive losses (for scraper timeout)
      this.state.consecutiveLosses++;
      this.state.consecutiveWins = 0;  // Reset win streak
    }

    this.state.stats.winRate = this.state.stats.totalTrades > 0
      ? (this.state.stats.wins / this.state.stats.totalTrades) * 100
      : 0;

    // ═══════════════════════════════════════════════════════════════
    // SCRAPER TIMEOUT - Prevent tilt and overconfidence
    // ═══════════════════════════════════════════════════════════════
    let cooldownExtra = 0;
    let timeoutReason = '';

    // 3 consecutive losses = 30 min break (prevent tilt)
    if (this.state.consecutiveLosses >= CONFIG.maxConsecutiveLosses) {
      cooldownExtra = CONFIG.consecutiveLossCooldownMs - CONFIG.cooldownMs;  // Add extra time
      timeoutReason = ` | ⚠️ ${this.state.consecutiveLosses} losses = 30min break`;
      console.log(`   🛑 SCRAPER TIMEOUT: ${this.state.consecutiveLosses} consecutive losses - taking 30 min break`);
    }

    // 5 consecutive wins = 10 min break (prevent overconfidence)
    if (this.state.consecutiveWins >= CONFIG.maxConsecutiveWins) {
      cooldownExtra = CONFIG.consecutiveWinsCooldownMs - CONFIG.cooldownMs;  // 10 min
      timeoutReason = ` | ⏸️ ${this.state.consecutiveWins} wins = 10min cooloff`;
      console.log(`   ⏸️ OVERCONFIDENCE CHECK: ${this.state.consecutiveWins} consecutive wins - taking 10 min break`);
    }

    // Set cooldown (normal + any extra)
    this.state.cooldownUntil = Date.now() + CONFIG.cooldownMs + cooldownExtra;

    this.state.openTrade = null;
    this.saveState();

    const emoji = finalPnl > 0 ? '✅' : '❌';
    const pnlSign = finalPnl >= 0 ? '+' : '';
    const holdMins = Math.round((trade.exitTime - trade.entryTime) / 60000);
    const candles = trade.candlesHeld || 0;

    console.log(`\n${emoji} ${this.state.symbol}: CLOSED ${reason}`);
    console.log(`   PnL: ${pnlSign}$${finalPnl.toFixed(2)} (${pnlSign}${finalPnlPercent.toFixed(2)}%)`);
    console.log(`   Held: ${holdMins}m (${candles} candles) | Fees: $${trade.feesPaid.toFixed(2)}`);
    console.log(`   Stats: ${this.state.stats.wins}W/${this.state.stats.losses}L (${this.state.stats.winRate.toFixed(1)}%)`);
    console.log(`   Balance: $${this.state.balance.toFixed(2)}${timeoutReason}\n`);

    return { closed: true, message: `${reason} ${pnlSign}${finalPnlPercent.toFixed(2)}%` };
  }
}

// ═══════════════════════════════════════════════════════════════
// MAIN TRADING LOOP
// ═══════════════════════════════════════════════════════════════

async function main() {
  console.log('═══════════════════════════════════════════════════════════════');
  console.log('     STRUCTURE-BASED SCALPER - Paper Trading');
  console.log('═══════════════════════════════════════════════════════════════');
  console.log(`Mode: ${CONFIG.mode}`);
  console.log(`Regime Mode: MOMENTUM (breakouts+trail) / RANGE (BB reversion) / NONE (skip)`);
  console.log(`Primary TF: ${CONFIG.primaryInterval}`);
  console.log(`STOPS: STRUCTURE-BASED (swing high/low, max 2% risk)`);
  console.log(`TARGET: Middle Bollinger Band (min 1.5:1 R:R)`);
  console.log(`Cooldown: ${CONFIG.cooldownMs / 1000}s`);
  console.log(`Auto-Learn: Every ${CONFIG.autoLearn.triggerEveryNTrades} trades`);
  console.log(`Symbols: ${SYMBOLS.length}`);
  console.log('═══════════════════════════════════════════════════════════════\n');

  // Check for --reset flag
  const shouldReset = process.argv.includes('--reset');
  if (shouldReset) {
    console.log('🔄 RESET MODE: Backing up and resetting all trader states...\n');
  }

  const client = Binance();

  // Initialize traders
  const traders: CoinTrader[] = [];
  for (const symbol of SYMBOLS) {
    const trader = new CoinTrader(symbol);
    await trader.initialize(client);
    if (shouldReset) {
      trader.resetState();
    }
    traders.push(trader);
  }

  console.log(`Initialized ${traders.length} traders\n`);
  console.log('Starting momentum scalper loop...\n');

/**
 * Write live summary report for OpenClaw to read and report back
 * Separate file: data/paper-trades-summary-scalp-live.json
 */
interface LiveSummary {
  timestamp: string;
  cycle: number;
  trader: 'scalp';
  openTrades: number;
  totalTrades: number;
  totalWins: number;
  totalLosses: number;
  winRate: number;
  totalPnl: number;
  currentStreak: {
    type: 'WIN' | 'LOSS' | 'NONE';
    count: number;
  };
  recentPerformance: {
    trades: number;
    wins: number;
    winRate: number;
    pnl: number;
  };
  openPositions: Array<{
    symbol: string;
    direction: 'LONG' | 'SHORT';
    entryPrice: number;
    currentPrice: number;
    unrealizedPnl: number;
    pnlPercent: number;
  }>;
}

function writeLiveSummary(
  traders: CoinTrader[],
  iteration: number,
  results: Array<{ symbol: string; price: number; trader: CoinTrader }>
): void {
  // Calculate summary metrics
  const openCount = traders.filter(t => t.state.openTrade).length;
  const totalPnl = traders.reduce((sum, t) => sum + t.state.stats.totalPnl, 0);
  const totalTrades = traders.reduce((sum, t) => sum + t.state.stats.totalTrades, 0);
  const totalWins = traders.reduce((sum, t) => sum + t.state.stats.wins, 0);
  const totalLosses = traders.reduce((sum, t) => sum + t.state.stats.losses, 0);
  const winRate = totalTrades > 0 ? (totalWins / totalTrades) * 100 : 0;

  // Calculate current streak
  const allTrades = traders.flatMap(t => t.state.trades).filter(t => t.status === 'CLOSED');
  let currentStreak = 0;
  let currentStreakType: 'WIN' | 'LOSS' | 'NONE' = 'NONE';
  for (let i = allTrades.length - 1; i >= 0; i--) {
    const pnl = allTrades[i].pnl || 0;
    if (currentStreak === 0) {
      currentStreakType = pnl >= 0 ? 'WIN' : 'LOSS';
      currentStreak++;
    } else if ((currentStreakType === 'WIN' && pnl >= 0) || (currentStreakType === 'LOSS' && pnl < 0)) {
      currentStreak++;
    } else {
      break;
    }
  }

  // Calculate recent performance (last 10 trades)
  const recentTrades = allTrades.slice(-10);
  const recentWins = recentTrades.filter(t => (t.pnl || 0) > 0).length;
  const recentWinRate = recentTrades.length > 0 ? (recentWins / recentTrades.length) * 100 : 0;
  const recentPnl = recentTrades.reduce((sum, t) => sum + (t.pnl || 0), 0);

  // Open positions
  const openPositions: LiveSummary['openPositions'] = [];
  for (const { symbol, price, trader } of results) {
    if (trader.state.openTrade) {
      const trade = trader.state.openTrade;
      const isLong = trade.direction === 'LONG';
      const currentPrice = price > 0 ? price : trade.entryPrice;
      const priceDiff = isLong ? currentPrice - trade.entryPrice : trade.entryPrice - currentPrice;
      const unrealizedPnl = priceDiff * (trade.currentPositionSize || 1);
      const pnlPercent = trade.entryPrice > 0 ? (priceDiff / trade.entryPrice) * 100 : 0;
      openPositions.push({
        symbol,
        direction: trade.direction,
        entryPrice: trade.entryPrice,
        currentPrice,
        unrealizedPnl: Math.round(unrealizedPnl * 100) / 100,
        pnlPercent: Math.round(pnlPercent * 100) / 100,
      });
    }
  }

  const summary: LiveSummary = {
    timestamp: new Date().toISOString(),
    cycle: iteration,
    trader: 'scalp',
    openTrades: openCount,
    totalTrades,
    totalWins,
    totalLosses,
    winRate: Math.round(winRate * 10) / 10,
    totalPnl: Math.round(totalPnl * 100) / 100,
    currentStreak: {
      type: currentStreakType,
      count: currentStreak,
    },
    recentPerformance: {
      trades: recentTrades.length,
      wins: recentWins,
      winRate: Math.round(recentWinRate * 10) / 10,
      pnl: Math.round(recentPnl * 100) / 100,
    },
    openPositions,
  };

  // Write to file
  try {
    const summaryDir = path.dirname(CONFIG.reporting.summaryFile);
    if (!fs.existsSync(summaryDir)) {
      fs.mkdirSync(summaryDir, { recursive: true });
    }
    fs.writeFileSync(CONFIG.reporting.summaryFile, JSON.stringify(summary, null, 2));
  } catch (e) {
    // Silently fail - this is optional reporting
  }
}

  // Helper for price display
  const getDecimalPlaces = (p: number): number => {
    if (p >= 1000) return 2;
    if (p >= 1000) return 2;
    if (p >= 100) return 2;
    if (p >= 10) return 3;
    if (p >= 1) return 4;
    return 5;
  };

  // Daily trade limit tracking (quality over quantity)
  let dailyTradeCount = 0;
  let dailyTradeDate = new Date().toDateString();

  // Main loop
  let iteration = 0;
  let headerPrinted = false;

  while (true) {
    iteration++;
    const now = new Date();
    const nowStr = now.toLocaleTimeString();
    const todayDate = now.toDateString();

    // Reset daily counter at midnight
    if (todayDate !== dailyTradeDate) {
      dailyTradeCount = 0;
      dailyTradeDate = todayDate;
      console.log(`\n📅 New day - trade counter reset`);
    }

    // Show header only once at startup
    if (!headerPrinted) {
      console.log('\n╔═══════════════════════════════════════════════════════════════╗');
      console.log(`║  REGIME SCALPER - MOMENTUM/RANGE/NONE                          ║`);
      console.log('╚═══════════════════════════════════════════════════════════════╝\n');
      headerPrinted = true;
    }

    // Small cycle marker each iteration
    console.log(`\n🔄 SCALP Cycle ${iteration} | ${nowStr} | Daily trades: ${dailyTradeCount}/${CONFIG.maxDailyTrades}`);

    // Daily trade limit check - quality over quantity
    const dailyLimitReached = dailyTradeCount >= CONFIG.maxDailyTrades;

    // Count current open positions (for display only)
    const currentOpenCount = traders.filter(t => t.state.openTrade).length;

    // Process each trader and collect results
    const results: Array<{ symbol: string; price: number; result: any; trader: typeof traders[0] }> = [];

    for (const trader of traders) {
      try {
        const tf5m = trader.state.timeframes.get('5m');
        const price = tf5m?.candles?.length ? tf5m.candles[tf5m.candles.length - 1].close : 0;

        // Skip new entries if daily limit reached
        if (dailyLimitReached && !trader.state.openTrade) {
          results.push({
            symbol: trader.state.symbol,
            price,
            result: { status: 'DAILY_LIMIT', details: `Max ${CONFIG.maxDailyTrades} trades/day reached` },
            trader
          });
          continue;
        }

        const hadOpenTrade = !!trader.state.openTrade;
        const result = await trader.tick(client);

        // Count new entries (trade opened this tick)
        if (!hadOpenTrade && trader.state.openTrade) {
          dailyTradeCount++;
        }

        results.push({ symbol: trader.state.symbol, price, result, trader });
      } catch (e: any) {
        results.push({
          symbol: trader.state.symbol,
          price: 0,
          result: { status: 'ERROR', details: e.message?.slice(0, 30) || 'Unknown error' },
          trader
        });
      }
    }

    // ═══════════════════════════════════════════════════════════════
    // PORTFOLIO REGIME: Classify market state across all coins
    // ═══════════════════════════════════════════════════════════════
    const regimeSignals: CoinRegimeSignals[] = [];
    for (const trader of traders) {
      const tf = trader.state.timeframes.get(CONFIG.primaryInterval);
      if (tf && tf.momentum) {
        regimeSignals.push(extractRegimeSignals(trader.state.symbol, tf.momentum));
      }
    }
    if (regimeSignals.length > 0) {
      currentPortfolioRegime = regimeDetector.classify(regimeSignals);
    }

    // ═══════════════════════════════════════════════════════════════
    // DISPLAY: Summary mode (clean) or Verbose mode (all coins)
    // ═══════════════════════════════════════════════════════════════
    const isVerbose = CONFIG.reporting.mode === 'verbose';

    // Show open trades (always shown)
    const openTrades = results.filter(({ trader }) => trader.state.openTrade);
    if (openTrades.length > 0) {
      console.log('⚡ OPEN TRADES:');
      for (const { symbol, price, trader } of openTrades) {
        const trade = trader.state.openTrade!;
        const isLong = trade.direction === 'LONG';
        const currentPrice = price > 0 ? price : trade.entryPrice;
        const priceDiff = isLong ? currentPrice - trade.entryPrice : trade.entryPrice - currentPrice;
        const unrealizedPnl = priceDiff * (trade.currentPositionSize || 1);
        const pnlPercent = trade.entryPrice > 0 ? (priceDiff / trade.entryPrice) * 100 : 0;
        const pnlSign = unrealizedPnl >= 0 ? '+' : '';
        const pricePrecision = getDecimalPlaces(trade.entryPrice);

        // TP hit indicators
        const tp1 = trade.tp1Hit ? '✓' : ' ';
        const tp2 = trade.tp2Hit ? '✓' : ' ';

        const currentPriceDisplay = price > 0 ? `$${price.toFixed(pricePrecision)}` : '$---';
        const phaseInfo = trade.phase2Active ? ' [P2]' : '';
        console.log(`   ${symbol}: ${trade.direction} | ${currentPriceDisplay} | ${pnlSign}$${unrealizedPnl.toFixed(2)} (${pnlSign}${pnlPercent.toFixed(2)}%) | TP: [${tp1}]$${trade.takeProfit1.toFixed(pricePrecision)} [${tp2}]$${trade.takeProfit2.toFixed(pricePrecision)}${phaseInfo} | SL:$${trade.stopLoss.toFixed(pricePrecision)}`);
      }
    }

    // Show all coins only in verbose mode
    if (isVerbose) {
      console.log('\n📡 SCANNING:');
      for (const { symbol, price, result, trader } of results) {
        if (trader.state.openTrade) continue; // Skip open trades (already shown)
        const priceDisplay = price > 0 ? `$${price.toFixed(getDecimalPlaces(price))}` : 'N/A';
        console.log(`   ${symbol.padEnd(10)}: ${priceDisplay.padEnd(14)} | ${result.status.padEnd(10)} | ${result.details}`);
      }
    } else {
      const scanCount = results.filter(({ trader }) => !trader.state.openTrade).length;
      console.log(`\n📡 SCANNING: ${scanCount} coins (use verbose mode for details)`);
    }

    // ═══════════════════════════════════════════════════════════════
    // SUMMARY: Enhanced metrics
    // ═══════════════════════════════════════════════════════════════
    const openCount = traders.filter(t => t.state.openTrade).length;
    const totalPnl = traders.reduce((sum, t) => sum + t.state.stats.totalPnl, 0);
    const totalTrades = traders.reduce((sum, t) => sum + t.state.stats.totalTrades, 0);
    const totalWins = traders.reduce((sum, t) => sum + t.state.stats.wins, 0);
    const totalLosses = traders.reduce((sum, t) => sum + t.state.stats.losses, 0);
    const winRate = totalTrades > 0 ? (totalWins / totalTrades * 100).toFixed(1) : '0.0';
    const pnlSign = totalPnl >= 0 ? '+' : '';
    const pnlColor = totalPnl >= 0 ? '🟢' : '🔴';

    // Calculate current streak
    const allTrades = traders.flatMap(t => t.state.trades).filter(t => t.status === 'CLOSED');
    let currentStreak = 0;
    let currentStreakType = 'NONE';
    for (let i = allTrades.length - 1; i >= 0; i--) {
      const pnl = allTrades[i].pnl || 0;
      if (currentStreak === 0) {
        currentStreakType = pnl >= 0 ? 'WIN' : 'LOSS';
        currentStreak++;
      } else if ((currentStreakType === 'WIN' && pnl >= 0) || (currentStreakType === 'LOSS' && pnl < 0)) {
        currentStreak++;
      } else {
        break;
      }
    }

    // Calculate unrealized P&L from open trades
    let unrealizedPnl = 0;
    for (const { price, trader } of results) {
      if (trader.state.openTrade) {
        const trade = trader.state.openTrade;
        const isLong = trade.direction === 'LONG';
        const currentPrice = price > 0 ? price : trade.entryPrice;
        const priceDiff = isLong ? currentPrice - trade.entryPrice : trade.entryPrice - currentPrice;
        unrealizedPnl += priceDiff * (trade.currentPositionSize || 1);
      }
    }
    const unrealizedSign = unrealizedPnl >= 0 ? '+' : '';
    const unrealizedColor = unrealizedPnl >= 0 ? '🟢' : '🔴';

    console.log('\n═══════════════════════════════════════════════════════════════');
    console.log(`📊 SUMMARY: Open: ${openCount} | Trades: ${totalTrades} (${totalWins}W/${totalLosses}L) | Win: ${winRate}%`);
    console.log(`           PnL: ${pnlColor} ${pnlSign}$${totalPnl.toFixed(2)} | Unrealized: ${unrealizedColor} ${unrealizedSign}$${unrealizedPnl.toFixed(2)}`);
    console.log(`           Streak: ${currentStreakType} (${currentStreak})`);
    // Portfolio regime display
    const regimeColor = currentPortfolioRegime.regime === 'RISK_ON' ? '🟢' :
                        currentPortfolioRegime.regime === 'RISK_OFF' ? '🔴' : '🟡';
    console.log(`           Regime: ${regimeColor} ${currentPortfolioRegime.regime} (${(currentPortfolioRegime.sizeMultiplier * 100).toFixed(0)}% size) | ADX: ${currentPortfolioRegime.metrics.avgAdx.toFixed(1)} | Trending: ${currentPortfolioRegime.metrics.trendingCoinCount}/${currentPortfolioRegime.metrics.totalCoinCount}`);
    console.log('═══════════════════════════════════════════════════════════════\n');

    // ═══════════════════════════════════════════════════════════════
    // LIVE SUMMARY: Write to JSON file for OpenClaw integration
    // ═══════════════════════════════════════════════════════════════
    writeLiveSummary(traders, iteration, results);

    // ═══════════════════════════════════════════════════════════════
    // PARAMETER ANALYSIS: Bucket analysis every N trades
    // ═══════════════════════════════════════════════════════════════
    if (parameterAnalyzer.shouldAnalyze(totalTrades)) {
      console.log(`\n📊 PARAMETER ANALYSIS: Analyzing ${totalTrades} trades...`);

      // Collect all closed trades from all traders
      const allClosedTrades: ClosedTrade[] = [];
      for (const trader of traders) {
        for (const t of trader.state.trades) {
          if (t.status !== 'CLOSED') continue;
          allClosedTrades.push({
            symbol: t.symbol,
            direction: t.direction,
            entryPrice: t.entryPrice,
            exitPrice: t.exitPrice || 0,
            pnl: t.pnl || 0,
            status: 'CLOSED',
            entryTime: t.entryTime,
            exitTime: t.exitTime || 0,
            regime: (t as any).entryFeatures?.regime,
            bbPosition: (t as any).entryFeatures?.bbPosition,
            volumeRatio: (t as any).entryFeatures?.volumeRatio,
            rsi: (t as any).entryFeatures?.rsiValue,
            holdCandles: t.candlesHeld,
            liquidityScore: (t as any).liquidityScore,
            signalStrength: (t as any).entryFeatures?.strength,
          });
        }
      }

      if (allClosedTrades.length >= DEFAULT_CONFIG.minTradesPerBucket) {
        lastAnalysisResult = parameterAnalyzer.analyze(allClosedTrades);
        parameterAnalyzer.markAnalyzed(totalTrades);

        console.log(`   ${lastAnalysisResult.summary}`);
        if (lastAnalysisResult.suggestions.length > 0) {
          console.log('\n   📋 SUGGESTIONS:');
          console.log('   ' + formatSuggestions(lastAnalysisResult.suggestions).replace(/\n/g, '\n   '));
        }
        console.log('');

        // ═══════════════════════════════════════════════════════════════
        // LLM REPORT: Send to GLM 5 for intelligent recommendations
        // ═══════════════════════════════════════════════════════════════
        if (llmReporter.isEnabled() && llmReporter.shouldCall()) {
          console.log('   🤖 Sending analysis to GLM 5...');
          try {
            lastLLMReport = await llmReporter.getRecommendations(lastAnalysisResult);
            if (lastLLMReport.recommendations.length > 0) {
              console.log('   ' + formatRecommendations(lastLLMReport).replace(/\n/g, '\n   '));
            } else if (lastLLMReport.error) {
              console.log(`   ⚠️ LLM: ${lastLLMReport.error}`);
            }
          } catch (e: any) {
            console.log(`   ⚠️ LLM error: ${e.message}`);
          }
          console.log('');
        }
      }
    }

    // Show last analysis suggestions in summary (if available)
    if (lastAnalysisResult && lastAnalysisResult.suggestions.length > 0) {
      const topSuggestion = lastAnalysisResult.suggestions[0];
      const impactIcon = topSuggestion.impact === 'HIGH' ? '🔴' : topSuggestion.impact === 'MEDIUM' ? '🟡' : '🟢';
      console.log(`💡 LAST ANALYSIS: ${impactIcon} ${topSuggestion.current} → ${topSuggestion.suggested}`);
    }

    // ═══════════════════════════════════════════════════════════════
    // AUTO-LEARNING: Trigger retraining every N trades
    // ═══════════════════════════════════════════════════════════════
    if (CONFIG.autoLearn.enabled && totalTrades > 0) {
      const tradesAtLastLearn = (global as any).__lastLearnedAt || 0;
      const tradesSinceLearn = totalTrades - tradesAtLastLearn;

      if (tradesSinceLearn >= CONFIG.autoLearn.triggerEveryNTrades && totalTrades >= CONFIG.autoLearn.minTradesForTraining) {
        console.log(`\n🧠 AUTO-LEARN: ${tradesSinceLearn} new trades - triggering learning loop...`);
        (global as any).__lastLearnedAt = totalTrades;

        try {
          const { execSync } = await import('child_process');
          const cwd = process.cwd();

          // Export trades
          console.log('   Exporting trades...');
          execSync('npm run export-paper-trades-scalp', { cwd, stdio: 'pipe' });

          // Find the latest export file
          const exportDir = path.join(cwd, 'data', 'h2o-training');
          const files = fs.readdirSync(exportDir)
            .filter(f => f.startsWith('paper_scalp_') && f.endsWith('.csv'))
            .sort()
            .reverse();

          if (files.length > 0) {
            const latestFile = path.join(exportDir, files[0]);
            console.log(`   Training on: ${files[0]}`);

            // Run walk-forward training to SEPARATE paper model dir (never overwrite main model)
            const paperModelDir = path.join(cwd, 'data', 'models-paper-scalp');
            if (!fs.existsSync(paperModelDir)) fs.mkdirSync(paperModelDir, { recursive: true });
            const trainOutput = execSync(`python scripts/lightgbm_walkforward.py --input "${latestFile}" --output "${paperModelDir}"`, {
              cwd,
              encoding: 'utf-8',
              timeout: 300000  // 5 min timeout
            });

            // Parse and display key metrics from training output
            const lines = trainOutput.split('\n');
            let showOutput = false;
            for (const line of lines) {
              const t = line.trim();
              if (!t || t.startsWith('Loading') || t.startsWith('Loaded') || t.startsWith('Config:')) continue;
              // Show the useful training output
              if (t.includes('WALK-FORWARD') ||
                  t.includes('RESULTS') ||
                  t.includes('FOLD') ||
                  t.includes('MODEL') ||
                  t.includes('SMALL DATA') ||
                  t.includes('Optimal threshold') ||
                  t.includes('Accuracy') ||
                  t.includes('AUC') ||
                  t.includes('Baseline') ||
                  t.includes('Filtered') ||
                  t.includes('Improvement') ||
                  t.includes('Win rate') || t.includes('Win Rate') || t.includes('WIN rate') ||
                  t.includes('PnL') ||
                  t.includes('Avg PnL') ||
                  t.includes('Top 10') ||
                  t.includes('Features Used') ||
                  t.includes('Best Fold') ||
                  t.includes('Best iteration') ||
                  t.includes('scale_pos_weight') ||
                  t.includes('Train:') || t.includes('Test:') ||
                  t.includes('Trades') ||
                  t.includes('Model saved') || t.includes('Saved') ||
                  t.includes('Not saving') ||
                  t.includes('improvement') ||
                  t.match(/^\d+\./) ||          // Numbered feature list (Top 10)
                  t.startsWith('---') || t.startsWith('===')) {
                console.log(`   ${t}`);
                showOutput = true;
              }
            }

            if (!showOutput) {
              console.log('   ✅ Training completed (no output captured)');
            }
            console.log('');
          }
        } catch (e: any) {
          console.log(`   ⚠️ Learning failed: ${e.message?.slice(0, 50)}\n`);
        }
      }
    }

    // Wait before next check
    await new Promise(resolve => setTimeout(resolve, CONFIG.checkIntervalMs));
  }
}

main().catch(console.error);
