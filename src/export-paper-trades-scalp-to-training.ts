#!/usr/bin/env node
/**
 * Export SCALP paper trades to ML training CSV.
 *
 * Reads saved features directly from trade records (no Binance API calls needed).
 * Trades without saved features fall back to the old reconstruction method.
 *
 * Usage:
 *   npm run export-paper-trades-scalp
 */

import fs from 'fs';
import path from 'path';

const DEFAULTS = {
  interval: '15m',
  lookback: 200,
  outputDir: path.join(process.cwd(), 'data', 'h2o-training'),
  tradesDir: path.join(process.cwd(), 'data', 'paper-trades-scalp'),
};

function intervalToMs(interval: string): number {
  const match = interval.match(/^(\d+)([mhdw])$/);
  if (!match) throw new Error(`Unsupported interval: ${interval}`);
  const n = parseInt(match[1], 10);
  const unit = match[2];
  if (unit === 'm') return n * 60_000;
  if (unit === 'h') return n * 60 * 60_000;
  if (unit === 'd') return n * 24 * 60 * 60_000;
  if (unit === 'w') return n * 7 * 24 * 60 * 60_000;
  throw new Error(`Unsupported interval unit: ${unit}`);
}

// A trade from the scalp trader JSON files
interface SavedTrade {
  id: string;
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  entryTime: number;
  exitPrice?: number;
  exitTime?: number;
  exitReason?: string;
  pnl?: number;
  pnlPercent?: number;
  status: string;
  originalPositionSize: number;
  stopLoss: number;
  takeProfit1: number;
  takeProfit2: number;
  takeProfit3: number;
  momentumStrength: number;
  signals: string[];
  entryFeatures?: {
    price: number;
    timestamp: number;
    regime: string;
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
  };
  exitFeatures?: {
    price: number;
    timestamp: number;
    regime: string;
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
  };
}

// Group partial closes into single trade outcomes
interface GroupedTrade {
  symbol: string;
  direction: 'LONG' | 'SHORT';
  entryPrice: number;
  entryTime: number;
  exitTime: number;
  exitReason: string;
  pnl: number;
  pnlPercent: number;
  holdingPeriods: number;
  outcome: 'WIN' | 'LOSS';
  entryFeatures?: SavedTrade['entryFeatures'];
  exitFeatures?: SavedTrade['exitFeatures'];
  signals: string[];
  momentumStrength: number;
}

function groupClosedTrades(allTrades: SavedTrade[], intervalMs: number): GroupedTrade[] {
  const closed = allTrades.filter(t => t.status === 'CLOSED' && !!t.exitTime && !!t.entryTime);
  const groups = new Map<string, SavedTrade[]>();

  for (const t of closed) {
    const key = `${t.symbol}::${t.entryTime}::${t.direction}`;
    const arr = groups.get(key) ?? [];
    arr.push(t);
    groups.set(key, arr);
  }

  const result: GroupedTrade[] = [];

  for (const [, trades] of groups.entries()) {
    trades.sort((a, b) => (a.exitTime ?? 0) - (b.exitTime ?? 0));

    const first = trades[0];
    const last = trades[trades.length - 1];
    const pnl = trades.reduce((s, t) => s + (Number(t.pnl) || 0), 0);
    const entryNotional = first.entryPrice > 0 ? first.entryPrice * (first.originalPositionSize || 1) : 1;
    const pnlPercent = entryNotional > 0 ? (pnl / entryNotional) * 100 : (last.pnlPercent ?? 0);
    const exitTime = last.exitTime ?? first.entryTime;
    const holdingPeriods = Math.max(1, Math.ceil((exitTime - first.entryTime) / intervalMs));

    result.push({
      symbol: first.symbol,
      direction: first.direction,
      entryPrice: first.entryPrice,
      entryTime: first.entryTime,
      exitTime,
      exitReason: (last.exitReason ?? 'UNKNOWN').toString(),
      pnl,
      pnlPercent,
      holdingPeriods,
      outcome: pnl > 0 ? 'WIN' : 'LOSS',
      entryFeatures: first.entryFeatures,  // Features from first partial (entry)
      exitFeatures: last.exitFeatures,      // Features from last partial (final exit)
      signals: first.signals || [],
      momentumStrength: first.momentumStrength || 0,
    });
  }

  result.sort((a, b) => a.entryTime - b.entryTime);
  return result;
}

// Build a CSV row from saved features (fast path - no API calls)
function buildRowFromSavedFeatures(g: GroupedTrade): Record<string, any> | null {
  const ef = g.entryFeatures;
  if (!ef) return null;

  const session = ef.killZone || 'unknown';

  return {
    // Entry features
    entry_time: g.entryTime,
    symbol: g.symbol,
    direction: g.direction.toLowerCase(),
    regime: ef.regime,
    session,
    is_kill_zone: ef.isKillZone ? 1 : 0,
    kill_zone: ef.killZone,

    // Momentum indicators at entry
    bb_position: ef.bbPosition,
    bb_width: ef.bbWidth,
    rsi_value: ef.rsiValue,
    rsi_state: ef.rsiValue > 70 ? 'overbought' : ef.rsiValue < 30 ? 'oversold' : 'neutral',
    ema_fast: ef.emaFast,
    ema_slow: ef.emaSlow,
    ema_aligned: ef.emaAligned,
    macd_line: ef.macdLine,
    macd_signal: ef.macdSignal,
    macd_histogram: ef.macdHistogram,
    macd_bullish_cross: ef.macdLine > ef.macdSignal && ef.macdHistogram > 0 ? 1 : 0,
    macd_bearish_cross: ef.macdLine < ef.macdSignal && ef.macdHistogram < 0 ? 1 : 0,
    vwap: ef.vwap,
    vwap_deviation: ef.vwapDeviation,
    vwap_deviation_std: ef.vwapDeviationStd,
    price_above_vwap: ef.priceAboveVwap ? 1 : 0,
    atr_percent: ef.atrPercent,
    volatility: ef.atrPercent,
    volume_ratio: ef.volumeRatio,
    volume_spike: ef.volumeSpike ? 1 : 0,
    strength: ef.strength,
    body_ratio: 0, // Not available in snapshot, ML can work without it

    // Price context
    price_position: ef.bbPosition,
    confluence_score: ef.strength,

    // Trade outcome
    outcome: g.outcome,
    pnl: g.pnl,
    pnl_percent: g.pnlPercent,
    exit_reason: g.exitReason,
    holding_periods: g.holdingPeriods,

    // Exit features (if available)
    exit_bb_position: g.exitFeatures?.bbPosition ?? 0,
    exit_rsi_value: g.exitFeatures?.rsiValue ?? 0,
    exit_regime: g.exitFeatures?.regime ?? '',
    exit_volume_ratio: g.exitFeatures?.volumeRatio ?? 0,
  };
}

function exportToCSV(rows: Record<string, any>[], outputPath: string): void {
  if (rows.length === 0) throw new Error('No rows to export');

  const headers = Object.keys(rows[0]).join(',');
  const csvRows = rows.map(row => {
    return Object.values(row).map(val => {
      if (typeof val === 'string') return `"${String(val).replaceAll('"', '""')}"`;
      if (typeof val === 'boolean') return val ? 1 : 0;
      if (typeof val === 'number') return Number.isFinite(val) ? val : 0;
      return `"${String(val)}"`;
    }).join(',');
  });

  fs.writeFileSync(outputPath, [headers, ...csvRows].join('\n'));
}

async function main(): Promise<void> {
  const interval = DEFAULTS.interval;
  const intervalMs = intervalToMs(interval);
  const tradesDir = DEFAULTS.tradesDir;
  const outputDir = DEFAULTS.outputDir;

  if (!fs.existsSync(tradesDir)) {
    throw new Error(`Trades directory not found: ${tradesDir}`);
  }
  if (!fs.existsSync(outputDir)) {
    fs.mkdirSync(outputDir, { recursive: true });
  }

  const files = fs
    .readdirSync(tradesDir)
    .filter(f => f.endsWith('.json') && !f.includes('summary'))
    .map(f => path.join(tradesDir, f));

  if (files.length === 0) {
    throw new Error(`No state files found in: ${tradesDir}`);
  }

  let rawTrades: SavedTrade[] = [];
  for (const file of files) {
    try {
      const state = JSON.parse(fs.readFileSync(file, 'utf-8'));
      const trades = (state?.trades ?? []) as SavedTrade[];
      rawTrades.push(...trades);
    } catch (e: any) {
      console.warn(`[WARN] Failed to read ${file}: ${e?.message ?? e}`);
    }
  }

  const grouped = groupClosedTrades(rawTrades, intervalMs);

  console.log(`\n=== Export Paper Trades (SCALP) ===`);
  console.log(`State files: ${files.length}`);
  console.log(`Raw trades: ${rawTrades.length}`);
  console.log(`Grouped closed entries: ${grouped.length}`);

  const rows: Record<string, any>[] = [];
  let withFeatures = 0;
  let withoutFeatures = 0;

  for (const g of grouped) {
    const row = buildRowFromSavedFeatures(g);
    if (row) {
      rows.push(row);
      withFeatures++;
    } else {
      withoutFeatures++;
    }
  }

  console.log(`\nWith saved features: ${withFeatures}`);
  console.log(`Without features (old trades): ${withoutFeatures}`);

  if (rows.length === 0) {
    console.log('\nNo trades with saved features yet.');
    console.log('New trades will have features saved automatically.');
    console.log('Run the traders and wait for new trades to close.\n');
    // Don't exit with error - this is expected for old data
    return;
  }

  const timestamp = new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19);
  const outCsv = path.join(outputDir, `paper_scalp_${timestamp}.csv`);
  const outJson = path.join(outputDir, `paper_scalp_${timestamp}.json`);

  exportToCSV(rows, outCsv);
  fs.writeFileSync(outJson, JSON.stringify({
    meta: {
      groupedTrades: grouped.length,
      exportedRows: rows.length,
      withFeatures,
      withoutFeatures,
      generatedAt: new Date().toISOString(),
    },
  }, null, 2));

  console.log(`\nExported: ${rows.length} rows`);
  console.log(`CSV:  ${outCsv}`);
  console.log(`JSON: ${outJson}`);
  console.log('');
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
