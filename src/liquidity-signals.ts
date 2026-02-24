/**
 * liquidity-signals.ts
 *
 * Detects liquidity-based reversal setups:
 *   - Fakeouts / stop hunts (sweep + close back inside)
 *   - QML (Quasimodo / failed swing)
 *   - Supply/demand flip zones
 *   - Compression → expansion (BB squeeze breakout)
 *   - Two-bar reversals / pin bars / outside bars
 *
 * Used by BOTH traders. Thresholds are passed in via LiquidityConfig
 * so the scalper (5m) and swing trader (4H) use appropriate values.
 */

// ─────────────────────────────────────────────────────────────────
// CANDLE TYPE (matches both traders)
// ─────────────────────────────────────────────────────────────────

export interface Candle {
  timestamp: number;
  open: number;
  high: number;
  low: number;
  close: number;
  volume: number;
}

// ─────────────────────────────────────────────────────────────────
// CONFIG — tune per timeframe
// ─────────────────────────────────────────────────────────────────

export interface LiquidityConfig {
  minSweepDepthPct: number;
  maxSweepDepthPct: number;
  sweepLookback: number;
  qmlLookback: number;
  flipZoneProximityPct: number;
  squeezeThresholdMultiple: number;
  minSqueezeCandleCount: number;
  minWickRatio: number;
  minEngulfBodyRatio: number;
  swingLookback: number;
}

export const SCALP_CONFIG: LiquidityConfig = {
  minSweepDepthPct: 0.10,
  maxSweepDepthPct: 0.50,
  sweepLookback: 10,
  qmlLookback: 5,
  flipZoneProximityPct: 0.30,
  squeezeThresholdMultiple: 0.75,
  minSqueezeCandleCount: 3,
  minWickRatio: 0.55,
  minEngulfBodyRatio: 0.50,
  swingLookback: 3,
};

export const SWING_CONFIG: LiquidityConfig = {
  minSweepDepthPct: 0.30,
  maxSweepDepthPct: 1.50,
  sweepLookback: 20,
  qmlLookback: 10,
  flipZoneProximityPct: 0.80,
  squeezeThresholdMultiple: 0.75,
  minSqueezeCandleCount: 4,
  minWickRatio: 0.50,
  minEngulfBodyRatio: 0.50,
  swingLookback: 5,
};

// ─────────────────────────────────────────────────────────────────
// OUTPUT INTERFACES
// ─────────────────────────────────────────────────────────────────

export interface FakeoutSignal {
  detected: boolean;
  direction: 'BULLISH' | 'BEARISH' | null;
  sweptLevel: number;
  sweepDepthPct: number;
  wickRatio: number;
  closedBackInside: boolean;
  volumeConfirmed: boolean;
}

export interface QMLSignal {
  detected: boolean;
  direction: 'BULLISH_QML' | 'BEARISH_QML' | null;
  failedSwingLevel: number;
  sweepLevel: number;
  sweepDepthPct: number;
}

export interface FlipZoneSignal {
  detected: boolean;
  type: 'DEMAND_FLIP' | 'SUPPLY_FLIP' | null;
  level: number;
  distancePct: number;
  priorTestCount: number;
}

export interface SqueezeSignal {
  inSqueeze: boolean;
  squeezeCandleCount: number;
  expansionBreakout: boolean;
  expansionDirection: 'BULLISH' | 'BEARISH' | null;
  expansionVolume: boolean;
}

export interface ReversalPatternSignal {
  pinBar: boolean;
  pinBarDirection: 'BULLISH' | 'BEARISH' | null;
  engulfing: boolean;
  engulfingDirection: 'BULLISH' | 'BEARISH' | null;
  twoBarReversal: boolean;
  twoBarDirection: 'BULLISH' | 'BEARISH' | null;
  outsideBar: boolean;
  outsideBarDirection: 'BULLISH' | 'BEARISH' | null;
}

export interface LiquiditySignals {
  fakeout: FakeoutSignal;
  qml: QMLSignal;
  flipZone: FlipZoneSignal;
  squeeze: SqueezeSignal;
  reversalPattern: ReversalPatternSignal;
  liquidityScore: number;
  signalTags: string[];
  bestDirection: 'LONG' | 'SHORT' | null;
}

// ─────────────────────────────────────────────────────────────────
// HELPERS
// ─────────────────────────────────────────────────────────────────

function findSwingHighsLows(
  candles: Candle[],
  lookback: number
): { highs: Array<{ price: number; idx: number }>; lows: Array<{ price: number; idx: number }> } {
  const highs: Array<{ price: number; idx: number }> = [];
  const lows: Array<{ price: number; idx: number }> = [];
  const len = candles.length;

  const start = lookback;
  const end = len - lookback - 1;

  for (let i = start; i <= end; i++) {
    const c = candles[i];
    let isHigh = true;
    let isLow = true;

    for (let j = i - lookback; j <= i + lookback; j++) {
      if (j === i) continue;
      if (candles[j].high >= c.high) isHigh = false;
      if (candles[j].low <= c.low) isLow = false;
    }

    if (isHigh) highs.push({ price: c.high, idx: i });
    if (isLow) lows.push({ price: c.low, idx: i });
  }

  return { highs: highs.reverse(), lows: lows.reverse() };
}

function avgVolume(candles: Candle[], period: number = 20): number {
  const slice = candles.slice(-period - 1, -1);
  if (slice.length === 0) return 1;
  return slice.reduce((s, c) => s + c.volume, 0) / slice.length;
}

function bbWidth(candles: Candle[], period: number = 20, stdDev: number = 2): number[] {
  const widths: number[] = [];
  for (let i = period - 1; i < candles.length; i++) {
    const slice = candles.slice(i - period + 1, i + 1);
    const mean = slice.reduce((s, c) => s + c.close, 0) / period;
    const variance = slice.reduce((s, c) => s + Math.pow(c.close - mean, 2), 0) / period;
    const std = Math.sqrt(variance);
    widths.push(mean > 0 ? (stdDev * 2 * std / mean) * 100 : 0);
  }
  return widths;
}

// ─────────────────────────────────────────────────────────────────
// 1. FAKEOUT / STOP HUNT DETECTOR
// ─────────────────────────────────────────────────────────────────

function detectFakeout(candles: Candle[], cfg: LiquidityConfig): FakeoutSignal {
  const empty: FakeoutSignal = {
    detected: false, direction: null, sweptLevel: 0,
    sweepDepthPct: 0, wickRatio: 0, closedBackInside: false, volumeConfirmed: false,
  };

  if (candles.length < cfg.sweepLookback + 2) return empty;

  const current = candles[candles.length - 1];
  const prior = candles[candles.length - 2];
  const lookbackCandles = candles.slice(-cfg.sweepLookback - 1, -1);

  const recentHigh = Math.max(...lookbackCandles.map(c => c.high));
  const recentLow = Math.min(...lookbackCandles.map(c => c.low));

  const avgVol = avgVolume(candles);
  const volRatio = current.volume / avgVol;
  const candleRange = current.high - current.low;
  if (candleRange === 0) return empty;

  // BEARISH FAKEOUT: swept highs but closed back below
  if (current.high > recentHigh && current.close < recentHigh) {
    const sweepAmount = current.high - recentHigh;
    const sweepDepthPct = (sweepAmount / recentHigh) * 100;

    if (sweepDepthPct >= cfg.minSweepDepthPct && sweepDepthPct <= cfg.maxSweepDepthPct) {
      const wickAbove = current.high - Math.max(current.open, current.close);
      const wickRatio = wickAbove / candleRange;

      if (wickRatio >= cfg.minWickRatio) {
        return {
          detected: true,
          direction: 'BEARISH',
          sweptLevel: recentHigh,
          sweepDepthPct,
          wickRatio,
          closedBackInside: true,
          volumeConfirmed: volRatio >= 1.3,
        };
      }
    }
  }

  // BULLISH FAKEOUT: swept lows but closed back above
  if (current.low < recentLow && current.close > recentLow) {
    const sweepAmount = recentLow - current.low;
    const sweepDepthPct = (sweepAmount / recentLow) * 100;

    if (sweepDepthPct >= cfg.minSweepDepthPct && sweepDepthPct <= cfg.maxSweepDepthPct) {
      const wickBelow = Math.min(current.open, current.close) - current.low;
      const wickRatio = wickBelow / candleRange;

      if (wickRatio >= cfg.minWickRatio) {
        return {
          detected: true,
          direction: 'BULLISH',
          sweptLevel: recentLow,
          sweepDepthPct,
          wickRatio,
          closedBackInside: true,
          volumeConfirmed: volRatio >= 1.3,
        };
      }
    }
  }

  // TWO-CANDLE FAKEOUT
  const preLookback = candles.slice(-cfg.sweepLookback - 2, -2);
  if (preLookback.length > 0) {
    const preHigh = Math.max(...preLookback.map(c => c.high));
    const preLow = Math.min(...preLookback.map(c => c.low));

    if (prior.high > preHigh && current.close < preHigh && current.close < prior.open) {
      const sweepDepthPct = ((prior.high - preHigh) / preHigh) * 100;
      if (sweepDepthPct >= cfg.minSweepDepthPct && sweepDepthPct <= cfg.maxSweepDepthPct) {
        const priorRange = prior.high - prior.low;
        const wickRatio = priorRange > 0 ? (prior.high - Math.max(prior.open, prior.close)) / priorRange : 0;
        return {
          detected: true, direction: 'BEARISH', sweptLevel: preHigh,
          sweepDepthPct, wickRatio, closedBackInside: true,
          volumeConfirmed: prior.volume / avgVol >= 1.3,
        };
      }
    }

    if (prior.low < preLow && current.close > preLow && current.close > prior.open) {
      const sweepDepthPct = ((preLow - prior.low) / preLow) * 100;
      if (sweepDepthPct >= cfg.minSweepDepthPct && sweepDepthPct <= cfg.maxSweepDepthPct) {
        const priorRange = prior.high - prior.low;
        const wickRatio = priorRange > 0 ? (Math.min(prior.open, prior.close) - prior.low) / priorRange : 0;
        return {
          detected: true, direction: 'BULLISH', sweptLevel: preLow,
          sweepDepthPct, wickRatio, closedBackInside: true,
          volumeConfirmed: prior.volume / avgVol >= 1.3,
        };
      }
    }
  }

  return empty;
}

// ─────────────────────────────────────────────────────────────────
// 2. QML — QUASIMODO / FAILED SWING
// ─────────────────────────────────────────────────────────────────

function detectQML(candles: Candle[], cfg: LiquidityConfig): QMLSignal {
  const empty: QMLSignal = {
    detected: false, direction: null, failedSwingLevel: 0,
    sweepLevel: 0, sweepDepthPct: 0,
  };

  if (candles.length < cfg.qmlLookback * 3) return empty;

  const { highs, lows } = findSwingHighsLows(candles, cfg.swingLookback);
  if (highs.length < 2 || lows.length < 2) return empty;

  const current = candles[candles.length - 1];

  // BEARISH QML: HH → LH → sweeps prior HL → reverses down
  const h1 = highs[1];
  const h2 = highs[0];

  if (h2.idx > h1.idx && h2.price < h1.price) {
    const lowsBetween = lows.filter(l => l.idx > h1.idx && l.idx < h2.idx);
    if (lowsBetween.length > 0) {
      const pivotLow = lowsBetween[0];
      if (current.low < pivotLow.price) {
        const sweepDepthPct = ((pivotLow.price - current.low) / pivotLow.price) * 100;
        if (current.close > pivotLow.price &&
            sweepDepthPct >= cfg.minSweepDepthPct &&
            sweepDepthPct <= cfg.maxSweepDepthPct) {
          return {
            detected: true, direction: 'BEARISH_QML',
            failedSwingLevel: h2.price, sweepLevel: pivotLow.price, sweepDepthPct,
          };
        }
      }
    }
  }

  // BULLISH QML: LL → HL → sweeps prior LH → reverses up
  const l1 = lows[1];
  const l2 = lows[0];

  if (l2.idx > l1.idx && l2.price > l1.price) {
    const highsBetween = highs.filter(h => h.idx > l1.idx && h.idx < l2.idx);
    if (highsBetween.length > 0) {
      const pivotHigh = highsBetween[0];
      if (current.high > pivotHigh.price) {
        const sweepDepthPct = ((current.high - pivotHigh.price) / pivotHigh.price) * 100;
        if (current.close < pivotHigh.price &&
            sweepDepthPct >= cfg.minSweepDepthPct &&
            sweepDepthPct <= cfg.maxSweepDepthPct) {
          return {
            detected: true, direction: 'BULLISH_QML',
            failedSwingLevel: l2.price, sweepLevel: pivotHigh.price, sweepDepthPct,
          };
        }
      }
    }
  }

  return empty;
}

// ─────────────────────────────────────────────────────────────────
// 3. SUPPLY / DEMAND FLIP ZONES
// ─────────────────────────────────────────────────────────────────

function detectFlipZone(candles: Candle[], cfg: LiquidityConfig): FlipZoneSignal {
  const empty: FlipZoneSignal = {
    detected: false, type: null, level: 0,
    distancePct: 0, priorTestCount: 0,
  };

  if (candles.length < cfg.sweepLookback * 2) return empty;

  const current = candles[candles.length - 1];
  const { highs, lows } = findSwingHighsLows(candles, cfg.swingLookback);

  if (highs.length < 2 || lows.length < 2) return empty;

  // DEMAND FLIP: prior swing high that price has broken above
  for (const swingHigh of highs) {
    if (swingHigh.price < current.close) {
      const distancePct = Math.abs((current.close - swingHigh.price) / swingHigh.price) * 100;
      if (distancePct <= cfg.flipZoneProximityPct) {
        const testCount = candles
          .slice(0, swingHigh.idx)
          .filter(c => Math.abs(c.high - swingHigh.price) / swingHigh.price < 0.002)
          .length;
        return { detected: true, type: 'DEMAND_FLIP', level: swingHigh.price, distancePct, priorTestCount: testCount };
      }
    }
  }

  // SUPPLY FLIP: prior swing low that price has broken below
  for (const swingLow of lows) {
    if (swingLow.price > current.close) {
      const distancePct = Math.abs((swingLow.price - current.close) / swingLow.price) * 100;
      if (distancePct <= cfg.flipZoneProximityPct) {
        const testCount = candles
          .slice(0, swingLow.idx)
          .filter(c => Math.abs(c.low - swingLow.price) / swingLow.price < 0.002)
          .length;
        return { detected: true, type: 'SUPPLY_FLIP', level: swingLow.price, distancePct, priorTestCount: testCount };
      }
    }
  }

  return empty;
}

// ─────────────────────────────────────────────────────────────────
// 4. COMPRESSION → EXPANSION (BB SQUEEZE)
// ─────────────────────────────────────────────────────────────────

function detectSqueeze(candles: Candle[], cfg: LiquidityConfig): SqueezeSignal {
  const empty: SqueezeSignal = {
    inSqueeze: false, squeezeCandleCount: 0,
    expansionBreakout: false, expansionDirection: null, expansionVolume: false,
  };

  if (candles.length < 30) return empty;

  const widths = bbWidth(candles);
  if (widths.length < cfg.minSqueezeCandleCount + 2) return empty;

  const avgWidth = widths.slice(-21, -1).reduce((a, b) => a + b, 0) / 20;
  const squeezeThreshold = avgWidth * cfg.squeezeThresholdMultiple;

  let squeezeCandleCount = 0;
  for (let i = widths.length - 2; i >= 0; i--) {
    if (widths[i] < squeezeThreshold) squeezeCandleCount++;
    else break;
  }

  const currentWidth = widths[widths.length - 1];
  const inSqueeze = currentWidth < squeezeThreshold;
  const wasInSqueeze = squeezeCandleCount >= cfg.minSqueezeCandleCount;
  const nowExpanding = currentWidth > squeezeThreshold && wasInSqueeze;

  if (!nowExpanding) {
    return { inSqueeze, squeezeCandleCount, expansionBreakout: false, expansionDirection: null, expansionVolume: false };
  }

  const current = candles[candles.length - 1];
  const avgVol = avgVolume(candles);
  const expansionDirection: 'BULLISH' | 'BEARISH' = current.close > current.open ? 'BULLISH' : 'BEARISH';

  return {
    inSqueeze: false, squeezeCandleCount,
    expansionBreakout: true, expansionDirection,
    expansionVolume: current.volume / avgVol >= 1.5,
  };
}

// ─────────────────────────────────────────────────────────────────
// 5. REVERSAL PATTERNS
// ─────────────────────────────────────────────────────────────────

function detectReversalPatterns(candles: Candle[], cfg: LiquidityConfig): ReversalPatternSignal {
  const empty: ReversalPatternSignal = {
    pinBar: false, pinBarDirection: null,
    engulfing: false, engulfingDirection: null,
    twoBarReversal: false, twoBarDirection: null,
    outsideBar: false, outsideBarDirection: null,
  };

  if (candles.length < 3) return empty;

  const curr = candles[candles.length - 1];
  const prev = candles[candles.length - 2];

  const currRange = curr.high - curr.low;
  const prevRange = prev.high - prev.low;
  if (currRange === 0 || prevRange === 0) return empty;

  const currBody = Math.abs(curr.close - curr.open);
  const prevBody = Math.abs(prev.close - prev.open);
  const currBodyRatio = currBody / currRange;

  // PIN BAR
  const lowerWick = Math.min(curr.open, curr.close) - curr.low;
  const upperWick = curr.high - Math.max(curr.open, curr.close);
  const lowerWickRatio = lowerWick / currRange;
  const upperWickRatio = upperWick / currRange;

  let pinBar = false;
  let pinBarDirection: 'BULLISH' | 'BEARISH' | null = null;

  if (lowerWickRatio >= cfg.minWickRatio && currBodyRatio < 0.3) {
    pinBar = true;
    pinBarDirection = 'BULLISH';
  } else if (upperWickRatio >= cfg.minWickRatio && currBodyRatio < 0.3) {
    pinBar = true;
    pinBarDirection = 'BEARISH';
  }

  // ENGULFING
  const bullishEngulf =
    curr.close > curr.open && prev.close < prev.open &&
    curr.open <= prev.close && curr.close >= prev.open &&
    currBodyRatio >= cfg.minEngulfBodyRatio;

  const bearishEngulf =
    curr.close < curr.open && prev.close > prev.open &&
    curr.open >= prev.close && curr.close <= prev.open &&
    currBodyRatio >= cfg.minEngulfBodyRatio;

  const engulfing = bullishEngulf || bearishEngulf;
  const engulfingDirection: 'BULLISH' | 'BEARISH' | null =
    bullishEngulf ? 'BULLISH' : bearishEngulf ? 'BEARISH' : null;

  // TWO-BAR REVERSAL
  const bullishTwoBar =
    prev.close < prev.open && curr.close > curr.open &&
    curr.close > prev.open && currBodyRatio >= 0.5 && prevBody / prevRange >= 0.5;

  const bearishTwoBar =
    prev.close > prev.open && curr.close < curr.open &&
    curr.close < prev.open && currBodyRatio >= 0.5 && prevBody / prevRange >= 0.5;

  const twoBarReversal = bullishTwoBar || bearishTwoBar;
  const twoBarDirection: 'BULLISH' | 'BEARISH' | null =
    bullishTwoBar ? 'BULLISH' : bearishTwoBar ? 'BEARISH' : null;

  // OUTSIDE BAR
  const isOutsideBar = curr.high > prev.high && curr.low < prev.low;
  const outsideBar = isOutsideBar && currBodyRatio >= cfg.minEngulfBodyRatio;
  const outsideBarDirection: 'BULLISH' | 'BEARISH' | null = outsideBar
    ? (curr.close > curr.open ? 'BULLISH' : 'BEARISH')
    : null;

  return { pinBar, pinBarDirection, engulfing, engulfingDirection, twoBarReversal, twoBarDirection, outsideBar, outsideBarDirection };
}

// ─────────────────────────────────────────────────────────────────
// MAIN EXPORT
// ─────────────────────────────────────────────────────────────────

export function detectLiquiditySignals(
  candles: Candle[],
  cfg: LiquidityConfig
): LiquiditySignals {
  const emptyLiquidity: LiquiditySignals = {
    fakeout: { detected: false, direction: null, sweptLevel: 0, sweepDepthPct: 0, wickRatio: 0, closedBackInside: false, volumeConfirmed: false },
    qml: { detected: false, direction: null, failedSwingLevel: 0, sweepLevel: 0, sweepDepthPct: 0 },
    flipZone: { detected: false, type: null, level: 0, distancePct: 0, priorTestCount: 0 },
    squeeze: { inSqueeze: false, squeezeCandleCount: 0, expansionBreakout: false, expansionDirection: null, expansionVolume: false },
    reversalPattern: { pinBar: false, pinBarDirection: null, engulfing: false, engulfingDirection: null, twoBarReversal: false, twoBarDirection: null, outsideBar: false, outsideBarDirection: null },
    liquidityScore: 0, signalTags: [], bestDirection: null,
  };

  if (candles.length < 20) return emptyLiquidity;

  const fakeout = detectFakeout(candles, cfg);
  const qml = detectQML(candles, cfg);
  const flipZone = detectFlipZone(candles, cfg);
  const squeeze = detectSqueeze(candles, cfg);
  const reversalPattern = detectReversalPatterns(candles, cfg);

  // SCORING
  let score = 0;
  const tags: string[] = [];
  const directionVotes: Record<'LONG' | 'SHORT', number> = { LONG: 0, SHORT: 0 };

  // Fakeout: highest weight
  if (fakeout.detected) {
    const basePoints = 35;
    const volBonus = fakeout.volumeConfirmed ? 10 : 0;
    score += basePoints + volBonus;
    const dir = fakeout.direction === 'BULLISH' ? 'LONG' : 'SHORT';
    directionVotes[dir] += 3;
    const arrow = fakeout.direction === 'BULLISH' ? '↑' : '↓';
    tags.push(`SWEEP${arrow}(${fakeout.sweepDepthPct.toFixed(2)}%)`);
    if (fakeout.volumeConfirmed) tags.push('VOL✓');
  }

  // QML
  if (qml.detected) {
    score += 30;
    const dir = qml.direction === 'BULLISH_QML' ? 'LONG' : 'SHORT';
    directionVotes[dir] += 3;
    tags.push(`QML-${qml.direction === 'BULLISH_QML' ? '↑' : '↓'}`);
  }

  // Flip zone
  if (flipZone.detected) {
    const basePoints = flipZone.priorTestCount >= 3 ? 20 : 12;
    score += basePoints;
    const dir = flipZone.type === 'DEMAND_FLIP' ? 'LONG' : 'SHORT';
    directionVotes[dir] += 2;
    tags.push(`FLIP-${flipZone.type === 'DEMAND_FLIP' ? 'D' : 'S'}(${flipZone.distancePct.toFixed(2)}%)`);
  }

  // Squeeze expansion
  if (squeeze.expansionBreakout) {
    const basePoints = squeeze.expansionVolume ? 20 : 10;
    score += basePoints;
    const dir = squeeze.expansionDirection === 'BULLISH' ? 'LONG' : 'SHORT';
    directionVotes[dir] += 1;
    tags.push(`SQUEEZE-EXP${squeeze.expansionDirection === 'BULLISH' ? '↑' : '↓'}`);
  } else if (squeeze.inSqueeze) {
    score += 5;
    tags.push(`SQZ(${squeeze.squeezeCandleCount}c)`);
  }

  // Reversal patterns
  if (reversalPattern.pinBar) {
    score += 12;
    const dir = reversalPattern.pinBarDirection === 'BULLISH' ? 'LONG' : 'SHORT';
    directionVotes[dir] += 2;
    tags.push(`PIN${reversalPattern.pinBarDirection === 'BULLISH' ? '↑' : '↓'}`);
  }

  if (reversalPattern.engulfing) {
    score += 15;
    const dir = reversalPattern.engulfingDirection === 'BULLISH' ? 'LONG' : 'SHORT';
    directionVotes[dir] += 2;
    tags.push(`ENGULF${reversalPattern.engulfingDirection === 'BULLISH' ? '↑' : '↓'}`);
  }

  if (reversalPattern.twoBarReversal) {
    score += 10;
    const dir = reversalPattern.twoBarDirection === 'BULLISH' ? 'LONG' : 'SHORT';
    directionVotes[dir] += 1;
    tags.push(`2BAR${reversalPattern.twoBarDirection === 'BULLISH' ? '↑' : '↓'}`);
  }

  if (reversalPattern.outsideBar) {
    score += 8;
    const dir = reversalPattern.outsideBarDirection === 'BULLISH' ? 'LONG' : 'SHORT';
    directionVotes[dir] += 1;
    tags.push(`OBAR${reversalPattern.outsideBarDirection === 'BULLISH' ? '↑' : '↓'}`);
  }

  score = Math.min(100, score);

  // Consensus direction
  let bestDirection: 'LONG' | 'SHORT' | null = null;
  if (directionVotes['LONG'] > directionVotes['SHORT'] && directionVotes['LONG'] >= 2) {
    bestDirection = 'LONG';
  } else if (directionVotes['SHORT'] > directionVotes['LONG'] && directionVotes['SHORT'] >= 2) {
    bestDirection = 'SHORT';
  }

  return {
    fakeout, qml, flipZone, squeeze, reversalPattern,
    liquidityScore: score,
    signalTags: tags,
    bestDirection,
  };
}
