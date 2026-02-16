#!/usr/bin/env python3
"""
Momentum Scalper Backtester V2

IMPROVEMENTS over V1:
- ATR-based stops (not fixed %)
- Better R:R ratios (1:1.5, 1:2.5, 1:4)
- Stricter RANGE mode (BB extremes only: <15% or >85%)
- Fixed TREND mode (requires FRESH crossovers)
- Higher signal quality threshold
- Kill zone filtering (optional)

Usage:
    python scripts/momentum_backtester_v2.py --days 365
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd

try:
    import ccxt
except ImportError:
    print("ERROR: ccxt not installed. Run: pip install ccxt")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION V2
# ═══════════════════════════════════════════════════════════════

SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
]

CONFIG = {
    'timeframe': '5m',
    'lookback_candles': 100,

    # Indicators
    'rsi_period': 14,
    'ema_fast': 9,
    'ema_slow': 21,
    'bb_period': 20,
    'bb_std': 2.0,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'atr_period': 14,

    # REGIME DETECTION
    'atr_percentile_trend': 60,  # ATR above 60th percentile = TREND

    # RANGE MODE (Mean Reversion) - STRICT
    'bb_extreme_low': 0.15,      # Only LONG below 15% of BB
    'bb_extreme_high': 0.85,     # Only SHORT above 85% of BB
    'rsi_oversold': 35,          # RSI must confirm
    'rsi_overbought': 65,

    # TREND MODE (Breakout)
    'require_fresh_cross': True,  # MACD/EMA cross must be within last 3 candles
    'cross_lookback': 3,

    # SIGNAL QUALITY
    'min_signals_trend': 4,      # Need 4+ signals for TREND
    'min_signals_range': 3,      # Need 3+ signals for RANGE
    'min_body_ratio': 0.5,       # Candle body must be 50%+ of range
    'volume_spike_mult': 1.5,    # 1.5x avg volume = spike

    # RISK MANAGEMENT - ATR-Based
    'sl_atr_mult': 1.5,          # Stop loss = 1.5x ATR from entry
    'tp1_rr': 1.5,               # TP1 = 1.5R (close 40%)
    'tp2_rr': 2.5,               # TP2 = 2.5R (close 30%)
    'tp3_rr': 4.0,               # TP3 = 4R (close 30%)
    'tp1_close_pct': 0.40,
    'tp2_close_pct': 0.30,
    'tp3_close_pct': 0.30,
    'max_hold_candles': 12,      # 1 hour max hold

    # FILTERS
    'require_kill_zone': False,  # If True, only trade during kill zones
    'min_atr_pct': 0.001,        # Min 0.1% ATR to trade (avoid dead markets)

    # Costs
    'position_size': 1000,
    'taker_fee': 0.0004,
    'slippage_bps': 2,
}


# ═══════════════════════════════════════════════════════════════
# DATA CLASSES
# ═══════════════════════════════════════════════════════════════

@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


@dataclass
class Trade:
    symbol: str
    direction: str
    entry_time: int
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    position_size: float
    entry_candle_idx: int
    atr_at_entry: float

    exit_time: int = 0
    exit_price: float = 0
    exit_reason: str = ''
    pnl: float = 0
    pnl_percent: float = 0
    fees: float = 0
    holding_candles: int = 0

    features: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════

def calculate_ema(closes: np.ndarray, period: int) -> np.ndarray:
    ema = np.zeros_like(closes)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(closes[:period])
    for i in range(period, len(closes)):
        ema[i] = (closes[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema


def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    rsi = np.zeros_like(closes)
    deltas = np.diff(closes)
    gains = np.where(deltas > 0, deltas, 0)
    losses = np.where(deltas < 0, -deltas, 0)

    avg_gain = np.zeros(len(closes))
    avg_loss = np.zeros(len(closes))
    avg_gain[period] = np.mean(gains[:period])
    avg_loss[period] = np.mean(losses[:period])

    for i in range(period + 1, len(closes)):
        avg_gain[i] = (avg_gain[i-1] * (period-1) + gains[i-1]) / period
        avg_loss[i] = (avg_loss[i-1] * (period-1) + losses[i-1]) / period

    with np.errstate(divide='ignore', invalid='ignore'):
        rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bb(closes: np.ndarray, period: int = 20, std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    middle = np.zeros_like(closes)
    upper = np.zeros_like(closes)
    lower = np.zeros_like(closes)

    for i in range(period - 1, len(closes)):
        window = closes[i-period+1:i+1]
        mean = np.mean(window)
        std_dev = np.std(window)
        middle[i] = mean
        upper[i] = mean + std * std_dev
        lower[i] = mean - std * std_dev

    return upper, middle, lower


def calculate_macd(closes: np.ndarray, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    ema_fast = calculate_ema(closes, fast)
    ema_slow = calculate_ema(closes, slow)
    macd_line = ema_fast - ema_slow
    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    atr = np.zeros_like(closes)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            np.abs(highs[1:] - closes[:-1]),
            np.abs(lows[1:] - closes[:-1])
        )
    )
    tr = np.insert(tr, 0, highs[0] - lows[0])
    atr[period-1] = np.mean(tr[:period])
    for i in range(period, len(closes)):
        atr[i] = (atr[i-1] * (period-1) + tr[i]) / period
    return atr


def calculate_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    typical_price = (highs + lows + closes) / 3
    cumulative_pv = np.cumsum(typical_price * volumes)
    cumulative_vol = np.cumsum(volumes)
    return np.where(cumulative_vol > 0, cumulative_pv / cumulative_vol, closes)


def get_kill_zone(timestamp: int) -> Tuple[str, bool]:
    dt = datetime.utcfromtimestamp(timestamp / 1000)
    hour = dt.hour
    if 7 <= hour < 10:
        return 'LONDON', True
    elif 13 <= hour < 16:
        return 'NY_OPEN', True
    elif 18 <= hour < 20:
        return 'NY_AFTERNOON', True
    elif 0 <= hour < 3:
        return 'ASIA', True
    return 'OFF_HOURS', False


# ═══════════════════════════════════════════════════════════════
# SIGNAL ANALYSIS (V2 - IMPROVED)
# ═══════════════════════════════════════════════════════════════

def analyze_candles_v2(candles: List[Candle], cfg: Dict, atr_history: np.ndarray) -> Dict:
    """Improved analysis with stricter signal requirements"""

    if len(candles) < cfg['lookback_candles']:
        return {'valid': False}

    closes = np.array([c.close for c in candles])
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    volumes = np.array([c.volume for c in candles])

    current = candles[-1]
    idx = len(candles) - 1

    # Indicators
    ema_fast = calculate_ema(closes, cfg['ema_fast'])
    ema_slow = calculate_ema(closes, cfg['ema_slow'])
    rsi = calculate_rsi(closes, cfg['rsi_period'])
    bb_upper, bb_middle, bb_lower = calculate_bb(closes, cfg['bb_period'], cfg['bb_std'])
    macd_line, macd_signal, macd_hist = calculate_macd(closes, cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal'])
    atr = calculate_atr(highs, lows, closes, cfg['atr_period'])
    vwap = calculate_vwap(highs, lows, closes, volumes)

    curr_close = closes[idx]
    curr_atr = atr[idx]
    curr_rsi = rsi[idx]
    curr_vwap = vwap[idx]

    # BB position
    bb_range = bb_upper[idx] - bb_lower[idx]
    bb_position = (curr_close - bb_lower[idx]) / bb_range if bb_range > 0 else 0.5

    # ATR as percentage
    atr_pct = curr_atr / curr_close if curr_close > 0 else 0

    # Check minimum volatility
    if atr_pct < cfg['min_atr_pct']:
        return {'valid': False, 'reason': 'Low volatility'}

    # REGIME: Use ATR percentile (not fixed threshold)
    if len(atr_history) > 50:
        atr_percentile = np.percentile(atr_history[-50:], cfg['atr_percentile_trend'])
        regime = 'TREND' if curr_atr > atr_percentile else 'RANGE'
    else:
        regime = 'RANGE'  # Default to range until we have history

    # Volume analysis
    avg_volume = np.mean(volumes[idx-20:idx]) if idx >= 20 else np.mean(volumes[:idx])
    volume_ratio = volumes[idx] / avg_volume if avg_volume > 0 else 1
    volume_spike = volume_ratio >= cfg['volume_spike_mult']

    # Kill zone
    kz_name, kz_active = get_kill_zone(current.timestamp)

    # Candle analysis
    candle_range = current.high - current.low
    candle_body = abs(current.close - current.open)
    body_ratio = candle_body / candle_range if candle_range > 0 else 0
    bullish_candle = current.close > current.open and body_ratio >= cfg['min_body_ratio']
    bearish_candle = current.close < current.open and body_ratio >= cfg['min_body_ratio']

    # CROSSOVER DETECTION (within last N candles)
    def check_recent_cross(arr1, arr2, bullish=True, lookback=3):
        """Check if crossover happened within last N candles"""
        for i in range(1, min(lookback + 1, idx)):
            prev_idx = idx - i
            curr_idx = idx - i + 1
            if bullish:
                if arr1[prev_idx] <= arr2[prev_idx] and arr1[curr_idx] > arr2[curr_idx]:
                    return True
            else:
                if arr1[prev_idx] >= arr2[prev_idx] and arr1[curr_idx] < arr2[curr_idx]:
                    return True
        return False

    lookback = cfg['cross_lookback']

    # EMA crosses
    ema_bullish_cross = check_recent_cross(ema_fast, ema_slow, bullish=True, lookback=lookback)
    ema_bearish_cross = check_recent_cross(ema_fast, ema_slow, bullish=False, lookback=lookback)
    ema_aligned_bull = ema_fast[idx] > ema_slow[idx] * 1.001
    ema_aligned_bear = ema_fast[idx] < ema_slow[idx] * 0.999

    # MACD crosses
    macd_bullish_cross = check_recent_cross(macd_line, macd_signal, bullish=True, lookback=lookback)
    macd_bearish_cross = check_recent_cross(macd_line, macd_signal, bullish=False, lookback=lookback)

    # RSI states
    rsi_oversold = curr_rsi <= cfg['rsi_oversold']
    rsi_overbought = curr_rsi >= cfg['rsi_overbought']

    # VWAP
    price_above_vwap = curr_close > curr_vwap

    # ════════════════════════════════════════════════════════════
    # SIGNAL COUNTING (V2 - More granular)
    # ════════════════════════════════════════════════════════════

    bullish_signals = 0
    bearish_signals = 0
    signal_details = {'bullish': [], 'bearish': []}

    # Strong signals (worth 2 points)
    if macd_bullish_cross:
        bullish_signals += 2
        signal_details['bullish'].append('MACD_CROSS')
    if macd_bearish_cross:
        bearish_signals += 2
        signal_details['bearish'].append('MACD_CROSS')

    if ema_bullish_cross:
        bullish_signals += 2
        signal_details['bullish'].append('EMA_CROSS')
    if ema_bearish_cross:
        bearish_signals += 2
        signal_details['bearish'].append('EMA_CROSS')

    # Medium signals (worth 1 point)
    if ema_aligned_bull:
        bullish_signals += 1
        signal_details['bullish'].append('EMA_ALIGN')
    if ema_aligned_bear:
        bearish_signals += 1
        signal_details['bearish'].append('EMA_ALIGN')

    if rsi_oversold:
        bullish_signals += 1
        signal_details['bullish'].append('RSI_OS')
    if rsi_overbought:
        bearish_signals += 1
        signal_details['bearish'].append('RSI_OB')

    if bb_position < cfg['bb_extreme_low']:
        bullish_signals += 1
        signal_details['bullish'].append('BB_LOW')
    if bb_position > cfg['bb_extreme_high']:
        bearish_signals += 1
        signal_details['bearish'].append('BB_HIGH')

    if volume_spike and bullish_candle:
        bullish_signals += 1
        signal_details['bullish'].append('VOL_CANDLE')
    if volume_spike and bearish_candle:
        bearish_signals += 1
        signal_details['bearish'].append('VOL_CANDLE')

    if price_above_vwap:
        bullish_signals += 1
        signal_details['bullish'].append('ABOVE_VWAP')
    else:
        bearish_signals += 1
        signal_details['bearish'].append('BELOW_VWAP')

    if macd_hist[idx] > 0:
        bullish_signals += 1
    else:
        bearish_signals += 1

    return {
        'valid': True,
        'timestamp': current.timestamp,
        'close': curr_close,

        # Core
        'regime': regime,
        'atr': curr_atr,
        'atr_pct': atr_pct,

        # Signals
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,
        'signal_details': signal_details,

        # BB
        'bb_position': bb_position,
        'bb_extreme_low': bb_position < cfg['bb_extreme_low'],
        'bb_extreme_high': bb_position > cfg['bb_extreme_high'],

        # RSI
        'rsi': curr_rsi,
        'rsi_oversold': rsi_oversold,
        'rsi_overbought': rsi_overbought,

        # MACD
        'macd_bullish_cross': macd_bullish_cross,
        'macd_bearish_cross': macd_bearish_cross,
        'macd_hist': macd_hist[idx],

        # EMA
        'ema_bullish_cross': ema_bullish_cross,
        'ema_bearish_cross': ema_bearish_cross,
        'ema_aligned_bull': ema_aligned_bull,
        'ema_aligned_bear': ema_aligned_bear,

        # VWAP
        'vwap': curr_vwap,
        'price_above_vwap': price_above_vwap,

        # Volume
        'volume_ratio': volume_ratio,
        'volume_spike': volume_spike,

        # Candle
        'body_ratio': body_ratio,
        'bullish_candle': bullish_candle,
        'bearish_candle': bearish_candle,

        # Kill zone
        'kill_zone': kz_name,
        'is_kill_zone': kz_active,
    }


def check_entry_v2(analysis: Dict, cfg: Dict) -> Tuple[bool, str, str]:
    """
    V2 Entry logic - stricter requirements

    Returns: (should_enter, direction, reason)
    """
    if not analysis.get('valid'):
        return False, '', analysis.get('reason', 'Invalid')

    regime = analysis['regime']
    bull = analysis['bullish_signals']
    bear = analysis['bearish_signals']

    # Kill zone filter
    if cfg['require_kill_zone'] and not analysis['is_kill_zone']:
        return False, '', 'Outside kill zone'

    # ════════════════════════════════════════════════════════════
    # RANGE MODE: Mean reversion at BB extremes
    # ════════════════════════════════════════════════════════════
    if regime == 'RANGE':
        min_signals = cfg['min_signals_range']

        # LONG: Price at BB extreme low + RSI oversold + momentum starting
        if analysis['bb_extreme_low'] and analysis['rsi_oversold']:
            if bull >= min_signals and bull > bear:
                if analysis['bullish_candle'] or analysis['macd_bullish_cross']:
                    return True, 'LONG', f'RANGE: BB+RSI reversal ({bull} signals)'

        # SHORT: Price at BB extreme high + RSI overbought + momentum starting
        if analysis['bb_extreme_high'] and analysis['rsi_overbought']:
            if bear >= min_signals and bear > bull:
                if analysis['bearish_candle'] or analysis['macd_bearish_cross']:
                    return True, 'SHORT', f'RANGE: BB+RSI reversal ({bear} signals)'

        return False, '', 'RANGE: No extreme reversal setup'

    # ════════════════════════════════════════════════════════════
    # TREND MODE: Breakout with fresh crossovers
    # ════════════════════════════════════════════════════════════
    else:
        min_signals = cfg['min_signals_trend']

        # LONG: Need MACD cross + EMA aligned + above VWAP
        if bull >= min_signals and bull > bear:
            has_momentum = analysis['macd_bullish_cross'] or analysis['ema_bullish_cross']
            has_alignment = analysis['ema_aligned_bull'] and analysis['price_above_vwap']

            if has_momentum and has_alignment:
                return True, 'LONG', f'TREND: Breakout long ({bull} signals)'

        # SHORT: Need MACD cross + EMA aligned + below VWAP
        if bear >= min_signals and bear > bull:
            has_momentum = analysis['macd_bearish_cross'] or analysis['ema_bearish_cross']
            has_alignment = analysis['ema_aligned_bear'] and not analysis['price_above_vwap']

            if has_momentum and has_alignment:
                return True, 'SHORT', f'TREND: Breakout short ({bear} signals)'

        return False, '', 'TREND: No valid breakout'


# ═══════════════════════════════════════════════════════════════
# TRADE SIMULATION (V2 - ATR-based stops)
# ═══════════════════════════════════════════════════════════════

def simulate_trade_v2(trade: Trade, candles: List[Candle], start_idx: int, cfg: Dict) -> Trade:
    """Simulate with ATR-based stops and better R:R"""

    is_long = trade.direction == 'LONG'
    remaining_size = trade.position_size
    total_pnl = 0
    tp1_hit = False
    tp2_hit = False

    for i in range(start_idx + 1, min(start_idx + cfg['max_hold_candles'] + 1, len(candles))):
        candle = candles[i]
        trade.holding_candles = i - start_idx

        # Check stop loss first
        if is_long and candle.low <= trade.stop_loss:
            exit_price = trade.stop_loss * (1 - cfg['slippage_bps'] / 10000)
            pnl = (exit_price - trade.entry_price) * remaining_size / trade.entry_price
            total_pnl += pnl
            trade.exit_time = candle.timestamp
            trade.exit_price = exit_price
            trade.exit_reason = 'SL'
            break

        if not is_long and candle.high >= trade.stop_loss:
            exit_price = trade.stop_loss * (1 + cfg['slippage_bps'] / 10000)
            pnl = (trade.entry_price - exit_price) * remaining_size / trade.entry_price
            total_pnl += pnl
            trade.exit_time = candle.timestamp
            trade.exit_price = exit_price
            trade.exit_reason = 'SL'
            break

        # TP1: 1.5R - close 40%
        if not tp1_hit:
            hit = (is_long and candle.high >= trade.tp1) or (not is_long and candle.low <= trade.tp1)
            if hit:
                slip = -cfg['slippage_bps'] / 10000 if is_long else cfg['slippage_bps'] / 10000
                exit_price = trade.tp1 * (1 + slip)
                close_size = trade.position_size * cfg['tp1_close_pct']
                pnl = (exit_price - trade.entry_price) * close_size / trade.entry_price if is_long else \
                      (trade.entry_price - exit_price) * close_size / trade.entry_price
                total_pnl += pnl
                remaining_size -= close_size
                tp1_hit = True
                # Trail stop to breakeven + small profit
                trade.stop_loss = trade.entry_price * (1.001 if is_long else 0.999)

        # TP2: 2.5R - close 30%
        if tp1_hit and not tp2_hit:
            hit = (is_long and candle.high >= trade.tp2) or (not is_long and candle.low <= trade.tp2)
            if hit:
                slip = -cfg['slippage_bps'] / 10000 if is_long else cfg['slippage_bps'] / 10000
                exit_price = trade.tp2 * (1 + slip)
                close_size = trade.position_size * cfg['tp2_close_pct']
                pnl = (exit_price - trade.entry_price) * close_size / trade.entry_price if is_long else \
                      (trade.entry_price - exit_price) * close_size / trade.entry_price
                total_pnl += pnl
                remaining_size -= close_size
                tp2_hit = True
                # Trail stop to TP1 level
                trade.stop_loss = trade.tp1

        # TP3: 4R - close remaining 30%
        if tp2_hit:
            hit = (is_long and candle.high >= trade.tp3) or (not is_long and candle.low <= trade.tp3)
            if hit:
                slip = -cfg['slippage_bps'] / 10000 if is_long else cfg['slippage_bps'] / 10000
                exit_price = trade.tp3 * (1 + slip)
                pnl = (exit_price - trade.entry_price) * remaining_size / trade.entry_price if is_long else \
                      (trade.entry_price - exit_price) * remaining_size / trade.entry_price
                total_pnl += pnl
                trade.exit_time = candle.timestamp
                trade.exit_price = exit_price
                trade.exit_reason = 'TP3'
                break

    # Timeout
    if trade.exit_time == 0:
        candle = candles[min(start_idx + cfg['max_hold_candles'], len(candles) - 1)]
        slip = -cfg['slippage_bps'] / 10000 if is_long else cfg['slippage_bps'] / 10000
        exit_price = candle.close * (1 + slip)
        pnl = (exit_price - trade.entry_price) * remaining_size / trade.entry_price if is_long else \
              (trade.entry_price - exit_price) * remaining_size / trade.entry_price
        total_pnl += pnl
        trade.exit_time = candle.timestamp
        trade.exit_price = exit_price
        trade.exit_reason = 'TIMEOUT'

    # Fees
    trade.fees = trade.position_size * cfg['taker_fee'] * 2
    trade.pnl = total_pnl - trade.fees
    trade.pnl_percent = (trade.pnl / trade.position_size) * 100

    return trade


# ═══════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════

def fetch_candles(symbol: str, timeframe: str, days: int) -> List[Candle]:
    exchange = ccxt.binance({'enableRateLimit': True})
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_candles = []

    print(f"  Fetching {symbol}...", end=' ', flush=True)

    while True:
        try:
            ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
            if not ohlcv:
                break

            for row in ohlcv:
                all_candles.append(Candle(
                    timestamp=row[0], open=row[1], high=row[2],
                    low=row[3], close=row[4], volume=row[5]
                ))

            if len(ohlcv) < 1000:
                break

            since = ohlcv[-1][0] + 1
            time.sleep(0.1)

        except Exception as e:
            print(f"Error: {e}")
            break

    print(f"{len(all_candles)} candles")
    return all_candles


# ═══════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_backtest_v2(days: int, cfg: Dict) -> List[Trade]:
    all_trades = []

    print(f"\n{'='*60}")
    print(f"MOMENTUM SCALPER BACKTEST V2")
    print(f"{'='*60}")
    print(f"Days: {days}")
    print(f"Timeframe: {cfg['timeframe']}")
    print(f"Stop: {cfg['sl_atr_mult']}x ATR | TP: {cfg['tp1_rr']}R/{cfg['tp2_rr']}R/{cfg['tp3_rr']}R")
    print(f"Range BB extremes: <{cfg['bb_extreme_low']*100:.0f}% / >{cfg['bb_extreme_high']*100:.0f}%")
    print(f"Min signals: TREND={cfg['min_signals_trend']}, RANGE={cfg['min_signals_range']}")
    print(f"{'='*60}\n")

    for symbol in SYMBOLS:
        candles = fetch_candles(symbol, cfg['timeframe'], days)

        if len(candles) < cfg['lookback_candles'] + 20:
            print(f"  {symbol}: Not enough data")
            continue

        # Build ATR history for regime detection
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        atr_full = calculate_atr(highs, lows, closes, cfg['atr_period'])

        symbol_trades = []
        cooldown_until = 0
        i = cfg['lookback_candles']

        while i < len(candles) - cfg['max_hold_candles']:
            candle = candles[i]

            if candle.timestamp < cooldown_until:
                i += 1
                continue

            # Analyze with ATR history
            window = candles[i - cfg['lookback_candles']:i + 1]
            atr_history = atr_full[max(0, i-50):i+1]
            analysis = analyze_candles_v2(window, cfg, atr_history)

            # Check entry
            should_enter, direction, reason = check_entry_v2(analysis, cfg)

            if should_enter:
                entry_price = candle.close
                slip = cfg['slippage_bps'] / 10000
                if direction == 'LONG':
                    entry_price *= (1 + slip)
                else:
                    entry_price *= (1 - slip)

                curr_atr = analysis['atr']
                sl_dist = curr_atr * cfg['sl_atr_mult']

                # ATR-based stops and targets
                if direction == 'LONG':
                    stop_loss = entry_price - sl_dist
                    tp1 = entry_price + sl_dist * cfg['tp1_rr']
                    tp2 = entry_price + sl_dist * cfg['tp2_rr']
                    tp3 = entry_price + sl_dist * cfg['tp3_rr']
                else:
                    stop_loss = entry_price + sl_dist
                    tp1 = entry_price - sl_dist * cfg['tp1_rr']
                    tp2 = entry_price - sl_dist * cfg['tp2_rr']
                    tp3 = entry_price - sl_dist * cfg['tp3_rr']

                trade = Trade(
                    symbol=symbol,
                    direction=direction,
                    entry_time=candle.timestamp,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    tp1=tp1,
                    tp2=tp2,
                    tp3=tp3,
                    position_size=cfg['position_size'],
                    entry_candle_idx=i,
                    atr_at_entry=curr_atr,
                    features={
                        'regime': analysis['regime'],
                        'atr_pct': analysis['atr_pct'],
                        'bb_position': analysis['bb_position'],
                        'rsi': analysis['rsi'],
                        'macd_hist': analysis['macd_hist'],
                        'volume_ratio': analysis['volume_ratio'],
                        'body_ratio': analysis['body_ratio'],
                        'bullish_signals': analysis['bullish_signals'],
                        'bearish_signals': analysis['bearish_signals'],
                        'kill_zone': analysis['kill_zone'],
                        'is_kill_zone': analysis['is_kill_zone'],
                        'price_above_vwap': analysis['price_above_vwap'],
                        'reason': reason,
                    }
                )

                trade = simulate_trade_v2(trade, candles, i, cfg)
                symbol_trades.append(trade)

                i += trade.holding_candles + 1
                cooldown_until = trade.exit_time + 300000  # 5 min cooldown

            else:
                i += 1

        print(f"  {symbol}: {len(symbol_trades)} trades")
        all_trades.extend(symbol_trades)

    return all_trades


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    rows = []
    for t in trades:
        row = {
            'symbol': t.symbol,
            'direction': t.direction.lower(),
            'entry_time': t.entry_time,
            'entry_price': t.entry_price,
            'exit_time': t.exit_time,
            'exit_price': t.exit_price,
            'exit_reason': t.exit_reason,
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'holding_periods': t.holding_candles,
            'outcome': 'WIN' if t.pnl > 0 else 'LOSS',
            'atr_at_entry': t.atr_at_entry,
            **t.features
        }
        rows.append(row)
    return pd.DataFrame(rows)


def print_summary(trades: List[Trade]):
    if not trades:
        print("\nNo trades generated!")
        return

    df = trades_to_dataframe(trades)

    print(f"\n{'='*60}")
    print("BACKTEST RESULTS V2")
    print(f"{'='*60}")

    total = len(df)
    wins = (df['outcome'] == 'WIN').sum()
    losses = total - wins
    win_rate = wins / total * 100 if total > 0 else 0

    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    avg_winner = df[df['outcome'] == 'WIN']['pnl'].mean() if wins > 0 else 0
    avg_loser = df[df['outcome'] == 'LOSS']['pnl'].mean() if losses > 0 else 0

    print(f"Total Trades: {total}")
    print(f"Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Avg PnL: ${avg_pnl:.2f}")
    print(f"Avg Winner: ${avg_winner:.2f} | Avg Loser: ${avg_loser:.2f}")

    if avg_loser != 0:
        print(f"Profit Factor: {abs(avg_winner/avg_loser):.2f}")

    # By exit reason
    print(f"\nBy Exit Reason:")
    for reason in df['exit_reason'].unique():
        sub = df[df['exit_reason'] == reason]
        wr = (sub['outcome'] == 'WIN').mean() * 100
        pnl = sub['pnl'].sum()
        print(f"  {reason}: {len(sub)} trades | {wr:.1f}% win | ${pnl:,.2f}")

    # By regime
    print(f"\nBy Regime:")
    for r in ['TREND', 'RANGE']:
        sub = df[df['regime'] == r]
        if len(sub) > 0:
            wr = (sub['outcome'] == 'WIN').mean() * 100
            pnl = sub['pnl'].sum()
            print(f"  {r}: {len(sub)} trades | {wr:.1f}% win | ${pnl:,.2f}")

    # By direction
    print(f"\nBy Direction:")
    for d in ['long', 'short']:
        sub = df[df['direction'] == d]
        if len(sub) > 0:
            wr = (sub['outcome'] == 'WIN').mean() * 100
            pnl = sub['pnl'].sum()
            print(f"  {d.upper()}: {len(sub)} trades | {wr:.1f}% win | ${pnl:,.2f}")


def main():
    parser = argparse.ArgumentParser(description='Momentum Scalper Backtester V2')
    parser.add_argument('--days', type=int, default=365, help='Days of data')
    parser.add_argument('--output', '-o', default='data/h2o-training/momentum_backtest_v2.csv')
    parser.add_argument('--kill-zone-only', action='store_true', help='Only trade during kill zones')
    args = parser.parse_args()

    cfg = CONFIG.copy()
    if args.kill_zone_only:
        cfg['require_kill_zone'] = True

    trades = run_backtest_v2(args.days, cfg)
    print_summary(trades)

    if trades:
        df = trades_to_dataframe(trades)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")
        print(f"Rows: {len(df)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
