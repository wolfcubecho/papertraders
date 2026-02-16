#!/usr/bin/env node
/**
 * Export labeled historical trades to CSV for LightGBM evaluation/training.
 *
 * This is a lightweight extractor/backtest:
 * - loads candles from Historical_Data_Lite (via LocalDataLoader)
 * - scans for setups using SMCIndicators + UnifiedScoring
 * - extracts TradeFeatures at entry
 * - simulates a simple SL/TP outcome to label WIN/LOSS
 * - writes a CSV compatible with scripts/lightgbm_trainer.py
 *
 * Usage:
 *   npm run export-historical-trades -- --symbols BTCUSDT,ETHUSDT --interval 15m --minScore 15 --maxTrades 5000
 */

import fs from 'fs';
import path from 'path';

import { LocalDataLoader } from './data-loader.js';
import { Candle, SMCIndicators } from './smc-indicators.js';
import { ICTIndicators } from './ict-indicators.js';
import { UnifiedScoring } from './unified-scoring.js';
import { FeatureExtractor, TradeFeatures, BacktestTrade } from './trade-features.js';

const DEFAULTS = {
  symbols: ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT'] as string[],
  interval: '5m',
  minScore: 15,
  lookback: 200,
  forwardCandles: 100,
  maxTrades: 5000,
  dataPath: path.join(process.cwd(), 'Historical_Data_Lite'),
  outputDir: path.join(process.cwd(), 'data', 'h2o-training'),
};

function parseArgs(argv: string[]): Record<string, string> {
  const out: Record<string, string> = {};
  for (let i = 0; i < argv.length; i++) {
    const a = argv[i];
    if (!a.startsWith('--')) continue;
    const key = a.slice(2);
    const val = argv[i + 1];
    if (!val || val.startsWith('--')) {
      out[key] = 'true';
    } else {
      out[key] = val;
      i++;
    }
  }
  return out;
}

function exportToCSV(trades: TradeFeatures[], outputPath: string): void {
  if (trades.length === 0) throw new Error('No rows to export');
  const headers = Object.keys(trades[0]).join(',');
  const rows = trades.map(trade => {
    const values = Object.values(trade).map(val => {
      if (typeof val === 'string') return `"${String(val).replaceAll('"', '""')}"`;
      if (typeof val === 'boolean') return val ? 1 : 0;
      if (typeof val === 'number') return Number.isFinite(val) ? val : 0;
      return `"${String(val)}"`;
    });
    return values.join(',');
  });
  fs.writeFileSync(outputPath, [headers, ...rows].join('\n'));
}

function parseIntervalToMs(interval: string): number {
  const match = interval.match(/^(\d+)([mhd])$/);
  if (!match) throw new Error(`Unsupported interval: ${interval}`);
  const n = parseInt(match[1], 10);
  const unit = match[2];
  if (unit === 'm') return n * 60_000;
  if (unit === 'h') return n * 60 * 60_000;
  if (unit === 'd') return n * 24 * 60 * 60_000;
  throw new Error(`Unsupported interval unit: ${unit}`);
}

function resampleCandles(base: Candle[], targetIntervalMs: number): Candle[] {
  if (base.length === 0) return [];

  const out: Candle[] = [];
  let currentBucketStart = Math.floor(base[0].timestamp / targetIntervalMs) * targetIntervalMs;
  let bucketOpen = base[0].open;
  let bucketHigh = base[0].high;
  let bucketLow = base[0].low;
  let bucketClose = base[0].close;
  let bucketVolume = base[0].volume;

  for (let i = 1; i < base.length; i++) {
    const c = base[i];
    const bucketStart = Math.floor(c.timestamp / targetIntervalMs) * targetIntervalMs;

    if (bucketStart !== currentBucketStart) {
      out.push({
        timestamp: currentBucketStart,
        open: bucketOpen,
        high: bucketHigh,
        low: bucketLow,
        close: bucketClose,
        volume: bucketVolume,
      });

      currentBucketStart = bucketStart;
      bucketOpen = c.open;
      bucketHigh = c.high;
      bucketLow = c.low;
      bucketClose = c.close;
      bucketVolume = c.volume;
      continue;
    }

    bucketHigh = Math.max(bucketHigh, c.high);
    bucketLow = Math.min(bucketLow, c.low);
    bucketClose = c.close;
    bucketVolume += c.volume;
  }

  out.push({
    timestamp: currentBucketStart,
    open: bucketOpen,
    high: bucketHigh,
    low: bucketLow,
    close: bucketClose,
    volume: bucketVolume,
  });

  return out;
}

async function loadCandlesWithFallback(
  loader: LocalDataLoader,
  symbol: string,
  interval: string,
): Promise<{ candles: Candle[]; source: string }>{
  try {
    const { candles } = await loader.loadData(symbol, interval);
    return { candles, source: interval };
  } catch (e) {
    // If the repo only has 5m data, we can still evaluate 15m by resampling.
    if (interval === '15m') {
      const { candles: base } = await loader.loadData(symbol, '5m');
      const candles = resampleCandles(base, parseIntervalToMs('15m'));
      return { candles, source: '5m->15m' };
    }
    throw e;
  }
}

function simulateOutcome(
  candles: Candle[],
  entryIndex: number,
  entryPrice: number,
  direction: 'long' | 'short',
  atrValue: number,
  forwardCandles: number
): BacktestTrade {
  const isLong = direction === 'long';

  // Simple fixed RR system: SL at 2*ATR, TP at 3R
  const stopDistance = Math.max(1e-12, atrValue * 2);
  const stopLoss = isLong ? (entryPrice - stopDistance) : (entryPrice + stopDistance);
  const riskDistance = Math.abs(entryPrice - stopLoss);
  const takeProfit = isLong ? (entryPrice + riskDistance * 3) : (entryPrice - riskDistance * 3);

  let exitPrice = candles[entryIndex].close;
  let exitReason = 'TIMEOUT';
  let exitIndex = Math.min(entryIndex + forwardCandles, candles.length - 1);

  for (let i = entryIndex + 1; i <= exitIndex; i++) {
    const c = candles[i];
    if (isLong) {
      if (c.low <= stopLoss) {
        exitPrice = stopLoss;
        exitReason = 'SL';
        exitIndex = i;
        break;
      }
      if (c.high >= takeProfit) {
        exitPrice = takeProfit;
        exitReason = 'TP';
        exitIndex = i;
        break;
      }
    } else {
      if (c.high >= stopLoss) {
        exitPrice = stopLoss;
        exitReason = 'SL';
        exitIndex = i;
        break;
      }
      if (c.low <= takeProfit) {
        exitPrice = takeProfit;
        exitReason = 'TP';
        exitIndex = i;
        break;
      }
    }
  }

  const priceDiff = isLong ? (exitPrice - entryPrice) : (entryPrice - exitPrice);

  // Normalize to a pseudo-$ PnL so downstream scripts have something consistent.
  // (The LightGBM trainer only uses outcome as target.)
  const pnl_percent = entryPrice > 0 ? (priceDiff / entryPrice) * 100 : 0;
  const pnl = 1000 * (pnl_percent / 100); // $1000 notional

  return {
    outcome: pnl > 0 ? 'WIN' : 'LOSS',
    pnl,
    pnl_percent,
    exit_reason: exitReason,
    holding_periods: Math.max(1, exitIndex - entryIndex),
  };
}

async function extractForSymbol(
  loader: LocalDataLoader,
  symbol: string,
  interval: string,
  minScore: number,
  lookback: number,
  forwardCandles: number,
  maxTrades: number,
): Promise<TradeFeatures[]> {
  const { candles, source } = await loadCandlesWithFallback(loader, symbol, interval);
  if (candles.length < lookback + 60) return [];

  if (source !== interval) {
    console.log(`  ${symbol}: using ${source} candles`);
  }

  const out: TradeFeatures[] = [];

  const weights = {
    trend_structure: 40,
    order_blocks: 30,
    fvgs: 20,
    ema_alignment: 15,
    liquidity: 10,
    mtf_bonus: 35,
    rsi_penalty: 15,
  };

  for (let i = lookback; i < candles.length - Math.max(50, forwardCandles); i++) {
    const window = candles.slice(i - lookback, i + 1);

    const analysis = SMCIndicators.analyze(window);
    const scoring = UnifiedScoring.calculateConfluence(analysis, window[window.length - 1].close, weights);

    if (scoring.score < minScore) continue;
    if (scoring.bias === 'neutral') continue;

    const direction = scoring.bias === 'bullish' ? 'long' : 'short';
    const ictAnalysis = ICTIndicators.analyzeFast(window, analysis);

    const featuresBase = FeatureExtractor.extractFeatures(
      window,
      window.length - 1,
      analysis,
      scoring.score,
      direction,
      ictAnalysis
    );

    const atrArr = SMCIndicators.atr(window, 14);
    const atrValue = atrArr.length ? atrArr[atrArr.length - 1] : (window[window.length - 1].high - window[window.length - 1].low);

    const outcome = simulateOutcome(
      candles,
      i,
      candles[i].close,
      direction,
      atrValue,
      forwardCandles
    );

    const features = FeatureExtractor.addOutcome(featuresBase, outcome);
    out.push(features);

    if (out.length >= maxTrades) break;
  }

  return out;
}

async function main(): Promise<void> {
  const args = parseArgs(process.argv.slice(2));

  const symbols = (args.symbols ? args.symbols.split(',') : DEFAULTS.symbols).map(s => s.trim()).filter(Boolean);
  const interval = args.interval ?? DEFAULTS.interval;
  const minScore = args.minScore ? parseInt(args.minScore, 10) : DEFAULTS.minScore;
  const lookback = args.lookback ? Math.max(50, parseInt(args.lookback, 10)) : DEFAULTS.lookback;
  const forwardCandles = args.forwardCandles ? Math.max(10, parseInt(args.forwardCandles, 10)) : DEFAULTS.forwardCandles;
  const maxTrades = args.maxTrades ? Math.max(100, parseInt(args.maxTrades, 10)) : DEFAULTS.maxTrades;
  const dataPath = args.dataPath ? path.resolve(args.dataPath) : DEFAULTS.dataPath;
  const outputDir = args.outputDir ? path.resolve(args.outputDir) : DEFAULTS.outputDir;

  if (!fs.existsSync(outputDir)) fs.mkdirSync(outputDir, { recursive: true });

  console.log('\n=== Export Historical Trades ===');
  console.log(`Symbols: ${symbols.join(', ')}`);
  console.log(`Interval: ${interval}`);
  console.log(`Min score: ${minScore}`);
  console.log(`Lookback: ${lookback}`);
  console.log(`Forward candles: ${forwardCandles}`);
  console.log(`Max trades (per symbol cap applies via early break): ${maxTrades}`);
  console.log(`Data path: ${dataPath}`);

  const loader = new LocalDataLoader(dataPath);

  const rows: TradeFeatures[] = [];
  for (const symbol of symbols) {
    try {
      const extracted = await extractForSymbol(loader, symbol, interval, minScore, lookback, forwardCandles, Math.max(100, Math.floor(maxTrades / symbols.length)));
      rows.push(...extracted);
      console.log(`  ${symbol}: +${extracted.length} (total ${rows.length})`);
      if (rows.length >= maxTrades) break;
    } catch (e: any) {
      console.warn(`  ${symbol}: failed (${e?.message ?? e})`);
    }
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const outCsv = path.join(outputDir, `historical_${interval}_${timestamp}.csv`);

  if (rows.length === 0) {
    console.log('❌ No rows extracted.');
    process.exit(1);
  }

  exportToCSV(rows, outCsv);
  console.log(`\n✅ Exported ${rows.length} rows`);
  console.log(`CSV: ${outCsv}`);
  console.log('\nNext:');
  console.log(`  python scripts/lightgbm_evaluate.py --input "${outCsv}"`);
  console.log('');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
