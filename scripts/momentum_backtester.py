#!/usr/bin/env python3
"""
Momentum Scalper Backtester

Simulates the dual-mode scalp strategy on historical data:
- TREND mode: MACD + EMA + VWAP breakouts
- RANGE mode: BB mean reversion at extremes
- Kill Zone timing
- 3 TP levels with partial closes

Generates training data for ML models.

Usage:
    python scripts/momentum_backtester.py --days 365 --output data/h2o-training/momentum_backtest.csv
"""

import argparse
import json
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
import time

import numpy as np
import pandas as pd

try:
    from binance.client import Client
    BINANCE_AVAILABLE = True
except ImportError:
    BINANCE_AVAILABLE = False
    print("Note: python-binance not installed, using ccxt fallback")

try:
    import ccxt
    CCXT_AVAILABLE = True
except ImportError:
    CCXT_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION (matches scalper)
# ═══════════════════════════════════════════════════════════════

SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
]

CONFIG = {
    'timeframe': '5m',           # Primary timeframe for scalping
    'lookback_candles': 100,     # Candles needed for indicators

    # Momentum thresholds
    'volume_spike_multiple': 1.8,
    'rsi_period': 14,
    'rsi_bullish_cross': 50,
    'rsi_bearish_cross': 50,
    'ema_fast': 9,
    'ema_slow': 21,
    'bb_period': 20,
    'bb_std': 2,
    'macd_fast': 12,
    'macd_slow': 26,
    'macd_signal': 9,
    'min_signals': 2,
    'min_body_ratio': 0.6,

    # Regime detection
    'atr_period': 14,
    'volatility_threshold': 0.015,  # 1.5% = TREND mode

    # Entry/Exit
    'stop_loss_pct': 0.25,
    'tp1_pct': 0.25,
    'tp2_pct': 0.40,
    'tp3_pct': 0.60,
    'max_hold_candles': 6,  # 30 mins at 5m = 6 candles

    # Risk
    'position_size': 1000,  # $1000 per trade
    'taker_fee': 0.0004,    # 4 bps
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
    direction: str  # 'LONG' or 'SHORT'
    entry_time: int
    entry_price: float
    stop_loss: float
    tp1: float
    tp2: float
    tp3: float
    position_size: float
    entry_candle_idx: int

    # Filled on exit
    exit_time: int = 0
    exit_price: float = 0
    exit_reason: str = ''
    pnl: float = 0
    pnl_percent: float = 0
    fees: float = 0
    holding_candles: int = 0

    # Entry features (for ML training)
    features: Dict = field(default_factory=dict)


# ═══════════════════════════════════════════════════════════════
# INDICATOR CALCULATIONS
# ═══════════════════════════════════════════════════════════════

def calculate_ema(closes: np.ndarray, period: int) -> np.ndarray:
    """Calculate EMA"""
    ema = np.zeros_like(closes)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(closes[:period])
    for i in range(period, len(closes)):
        ema[i] = (closes[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema


def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate RSI"""
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

    rs = np.where(avg_loss > 0, avg_gain / avg_loss, 100)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def calculate_bb(closes: np.ndarray, period: int = 20, std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate Bollinger Bands"""
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
    """Calculate MACD"""
    ema_fast = calculate_ema(closes, fast)
    ema_slow = calculate_ema(closes, slow)
    macd_line = ema_fast - ema_slow

    signal_line = calculate_ema(macd_line, signal)
    histogram = macd_line - signal_line

    return macd_line, signal_line, histogram


def calculate_atr(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> np.ndarray:
    """Calculate ATR"""
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


def calculate_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate VWAP and std deviation"""
    typical_price = (highs + lows + closes) / 3
    cumulative_pv = np.cumsum(typical_price * volumes)
    cumulative_vol = np.cumsum(volumes)

    vwap = np.where(cumulative_vol > 0, cumulative_pv / cumulative_vol, closes)

    # Rolling std dev from VWAP
    vwap_std = np.zeros_like(closes)
    for i in range(20, len(closes)):
        window = typical_price[i-20:i+1]
        vwap_std[i] = np.std(window - vwap[i])

    return vwap, vwap_std


def get_kill_zone(timestamp: int) -> Tuple[str, bool]:
    """Get kill zone based on UTC hour"""
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
    else:
        return 'OFF_HOURS', False


# ═══════════════════════════════════════════════════════════════
# SIGNAL GENERATION
# ═══════════════════════════════════════════════════════════════

def analyze_candles(candles: List[Candle], cfg: Dict) -> Dict:
    """Analyze candles and return all indicators + signals"""

    if len(candles) < cfg['lookback_candles']:
        return {'valid': False}

    # Convert to numpy
    closes = np.array([c.close for c in candles])
    highs = np.array([c.high for c in candles])
    lows = np.array([c.low for c in candles])
    volumes = np.array([c.volume for c in candles])
    timestamps = np.array([c.timestamp for c in candles])

    current = candles[-1]
    prev = candles[-2]
    idx = len(candles) - 1

    # Calculate all indicators
    ema_fast = calculate_ema(closes, cfg['ema_fast'])
    ema_slow = calculate_ema(closes, cfg['ema_slow'])
    rsi = calculate_rsi(closes, cfg['rsi_period'])
    bb_upper, bb_middle, bb_lower = calculate_bb(closes, cfg['bb_period'], cfg['bb_std'])
    macd_line, macd_signal, macd_hist = calculate_macd(closes, cfg['macd_fast'], cfg['macd_slow'], cfg['macd_signal'])
    atr = calculate_atr(highs, lows, closes, cfg['atr_period'])
    vwap, vwap_std = calculate_vwap(highs, lows, closes, volumes)

    # Current values
    curr_close = closes[idx]
    curr_ema_fast = ema_fast[idx]
    curr_ema_slow = ema_slow[idx]
    prev_ema_fast = ema_fast[idx-1]
    prev_ema_slow = ema_slow[idx-1]
    curr_rsi = rsi[idx]
    prev_rsi = rsi[idx-1]
    curr_bb_upper = bb_upper[idx]
    curr_bb_lower = bb_lower[idx]
    curr_bb_middle = bb_middle[idx]
    curr_macd = macd_line[idx]
    curr_macd_sig = macd_signal[idx]
    curr_macd_hist = macd_hist[idx]
    prev_macd = macd_line[idx-1]
    prev_macd_sig = macd_signal[idx-1]
    curr_atr = atr[idx]
    curr_vwap = vwap[idx]
    curr_vwap_std = vwap_std[idx]

    # Volume analysis
    avg_volume = np.mean(volumes[idx-20:idx])
    volume_ratio = volumes[idx] / avg_volume if avg_volume > 0 else 1
    volume_spike = volume_ratio >= cfg['volume_spike_multiple']

    # BB position (0 = lower, 0.5 = middle, 1 = upper)
    bb_range = curr_bb_upper - curr_bb_lower
    bb_position = (curr_close - curr_bb_lower) / bb_range if bb_range > 0 else 0.5

    # Regime detection
    atr_percent = curr_atr / curr_close if curr_close > 0 else 0
    regime = 'TREND' if atr_percent >= cfg['volatility_threshold'] else 'RANGE'

    # VWAP deviation
    vwap_deviation = ((curr_close - curr_vwap) / curr_vwap * 100) if curr_vwap > 0 else 0
    vwap_deviation_std = (curr_close - curr_vwap) / curr_vwap_std if curr_vwap_std > 0 else 0
    price_above_vwap = curr_close > curr_vwap

    # Kill zone
    kz_name, kz_active = get_kill_zone(current.timestamp)

    # Crossovers
    ema_bullish_cross = prev_ema_fast <= prev_ema_slow and curr_ema_fast > curr_ema_slow
    ema_bearish_cross = prev_ema_fast >= prev_ema_slow and curr_ema_fast < curr_ema_slow
    ema_aligned = 'bullish' if curr_ema_fast > curr_ema_slow * 1.001 else ('bearish' if curr_ema_fast < curr_ema_slow * 0.999 else 'neutral')

    rsi_bullish_cross = prev_rsi < cfg['rsi_bullish_cross'] and curr_rsi >= cfg['rsi_bullish_cross']
    rsi_bearish_cross = prev_rsi > cfg['rsi_bearish_cross'] and curr_rsi <= cfg['rsi_bearish_cross']
    rsi_overbought = curr_rsi >= 70
    rsi_oversold = curr_rsi <= 30

    macd_bullish_cross = prev_macd <= prev_macd_sig and curr_macd > curr_macd_sig
    macd_bearish_cross = prev_macd >= prev_macd_sig and curr_macd < curr_macd_sig

    # Price breakouts
    lookback_highs = highs[idx-10:idx]
    lookback_lows = lows[idx-10:idx]
    recent_high = np.max(lookback_highs) if len(lookback_highs) > 0 else curr_close
    recent_low = np.min(lookback_lows) if len(lookback_lows) > 0 else curr_close
    price_breakout_up = curr_close > recent_high
    price_breakout_down = curr_close < recent_low

    bb_breakout_up = curr_close > curr_bb_upper
    bb_breakout_down = curr_close < curr_bb_lower

    # Candle momentum
    candle_range = current.high - current.low
    candle_body = abs(current.close - current.open)
    body_ratio = candle_body / candle_range if candle_range > 0 else 0
    bullish_candle = current.close > current.open
    bearish_candle = current.close < current.open

    candle_momentum = 'neutral'
    if body_ratio >= cfg['min_body_ratio']:
        candle_momentum = 'bullish' if bullish_candle else 'bearish'

    # Count signals
    bullish_signals = 0
    bearish_signals = 0

    if volume_spike and candle_momentum == 'bullish':
        bullish_signals += 1
    if volume_spike and candle_momentum == 'bearish':
        bearish_signals += 1

    if rsi_bullish_cross or rsi_oversold:
        bullish_signals += 1
    if rsi_bearish_cross or rsi_overbought:
        bearish_signals += 1

    if ema_bullish_cross or ema_aligned == 'bullish':
        bullish_signals += 1
    if ema_bearish_cross or ema_aligned == 'bearish':
        bearish_signals += 1

    if bb_breakout_up:
        bullish_signals += 1
    if bb_breakout_down:
        bearish_signals += 1

    if price_breakout_up:
        bullish_signals += 1
    if price_breakout_down:
        bearish_signals += 1

    if candle_momentum == 'bullish':
        bullish_signals += 1
    if candle_momentum == 'bearish':
        bearish_signals += 1

    if macd_bullish_cross or curr_macd_hist > 0:
        bullish_signals += 1
    if macd_bearish_cross or curr_macd_hist < 0:
        bearish_signals += 1

    # Direction
    direction = 'NEUTRAL'
    if bullish_signals >= cfg['min_signals'] and bullish_signals > bearish_signals:
        direction = 'LONG'
    elif bearish_signals >= cfg['min_signals'] and bearish_signals > bullish_signals:
        direction = 'SHORT'

    strength = max(bullish_signals, bearish_signals) / 7

    return {
        'valid': True,
        'timestamp': current.timestamp,
        'close': curr_close,
        'direction': direction,
        'strength': strength,
        'bullish_signals': bullish_signals,
        'bearish_signals': bearish_signals,

        # Regime
        'regime': regime,
        'atr': curr_atr,
        'atr_percent': atr_percent,

        # VWAP
        'vwap': curr_vwap,
        'vwap_deviation': vwap_deviation,
        'vwap_deviation_std': vwap_deviation_std,
        'price_above_vwap': price_above_vwap,

        # Kill zone
        'kill_zone': kz_name,
        'is_kill_zone': kz_active,

        # BB
        'bb_position': bb_position,
        'bb_upper': curr_bb_upper,
        'bb_lower': curr_bb_lower,
        'bb_breakout_up': bb_breakout_up,
        'bb_breakout_down': bb_breakout_down,

        # EMA
        'ema_fast': curr_ema_fast,
        'ema_slow': curr_ema_slow,
        'ema_aligned': ema_aligned,
        'ema_bullish_cross': ema_bullish_cross,
        'ema_bearish_cross': ema_bearish_cross,

        # RSI
        'rsi_value': curr_rsi,
        'rsi_bullish_cross': rsi_bullish_cross,
        'rsi_bearish_cross': rsi_bearish_cross,
        'rsi_overbought': rsi_overbought,
        'rsi_oversold': rsi_oversold,

        # MACD
        'macd_line': curr_macd,
        'macd_signal': curr_macd_sig,
        'macd_histogram': curr_macd_hist,
        'macd_bullish_cross': macd_bullish_cross,
        'macd_bearish_cross': macd_bearish_cross,

        # Volume
        'volume_ratio': volume_ratio,
        'volume_spike': volume_spike,

        # Candle
        'body_ratio': body_ratio,
        'candle_momentum': candle_momentum,

        # Breakouts
        'price_breakout_up': price_breakout_up,
        'price_breakout_down': price_breakout_down,
    }


def check_entry(analysis: Dict, cfg: Dict) -> Tuple[bool, str]:
    """Check if we should enter a trade based on dual-mode logic"""

    if not analysis['valid'] or analysis['direction'] == 'NEUTRAL':
        return False, 'No signal'

    direction = analysis['direction']
    regime = analysis['regime']
    bb_pos = analysis['bb_position']

    # VWAP confirmation
    vwap_confirm = (direction == 'LONG' and analysis['price_above_vwap']) or \
                   (direction == 'SHORT' and not analysis['price_above_vwap'])

    if regime == 'TREND':
        # TREND MODE: MACD + EMA alignment + VWAP confirmation
        has_macd = (direction == 'LONG' and (analysis['macd_bullish_cross'] or analysis['macd_histogram'] > 0)) or \
                   (direction == 'SHORT' and (analysis['macd_bearish_cross'] or analysis['macd_histogram'] < 0))

        has_ema = (direction == 'LONG' and analysis['ema_aligned'] == 'bullish') or \
                  (direction == 'SHORT' and analysis['ema_aligned'] == 'bearish')

        has_breakout = (direction == 'LONG' and (analysis['price_breakout_up'] or analysis['bb_breakout_up'])) or \
                       (direction == 'SHORT' and (analysis['price_breakout_down'] or analysis['bb_breakout_down']))

        if not has_macd:
            return False, 'TREND: No MACD confirm'
        if not has_ema and not has_breakout:
            return False, 'TREND: No EMA/breakout'
        if not vwap_confirm:
            return False, 'TREND: Wrong side of VWAP'

        return True, f'TREND {direction}'

    else:
        # RANGE MODE: BB mean reversion at extremes
        if 0.4 <= bb_pos <= 0.6:
            return False, 'RANGE: BB middle zone'

        if direction == 'LONG' and bb_pos > 0.4:
            return False, 'RANGE: LONG needs BB<40%'

        if direction == 'SHORT' and bb_pos < 0.6:
            return False, 'RANGE: SHORT needs BB>60%'

        return True, f'RANGE {direction} @ BB'


# ═══════════════════════════════════════════════════════════════
# TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════

def simulate_trade(trade: Trade, candles: List[Candle], start_idx: int, cfg: Dict) -> Trade:
    """Simulate trade execution through candles"""

    is_long = trade.direction == 'LONG'
    remaining_size = trade.position_size
    total_pnl = 0
    tp1_hit = False
    tp2_hit = False

    for i in range(start_idx + 1, min(start_idx + cfg['max_hold_candles'] + 1, len(candles))):
        candle = candles[i]
        trade.holding_candles = i - start_idx

        # Check stop loss
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

        # Check TP1 (50% close)
        if not tp1_hit:
            if (is_long and candle.high >= trade.tp1) or (not is_long and candle.low <= trade.tp1):
                exit_price = trade.tp1 * (1 - cfg['slippage_bps'] / 10000 if is_long else 1 + cfg['slippage_bps'] / 10000)
                close_size = trade.position_size * 0.5
                if is_long:
                    pnl = (exit_price - trade.entry_price) * close_size / trade.entry_price
                else:
                    pnl = (trade.entry_price - exit_price) * close_size / trade.entry_price
                total_pnl += pnl
                remaining_size -= close_size
                tp1_hit = True
                # Move stop to breakeven
                trade.stop_loss = trade.entry_price

        # Check TP2 (25% close)
        if tp1_hit and not tp2_hit:
            if (is_long and candle.high >= trade.tp2) or (not is_long and candle.low <= trade.tp2):
                exit_price = trade.tp2 * (1 - cfg['slippage_bps'] / 10000 if is_long else 1 + cfg['slippage_bps'] / 10000)
                close_size = trade.position_size * 0.25
                if is_long:
                    pnl = (exit_price - trade.entry_price) * close_size / trade.entry_price
                else:
                    pnl = (trade.entry_price - exit_price) * close_size / trade.entry_price
                total_pnl += pnl
                remaining_size -= close_size
                tp2_hit = True

        # Check TP3 (final 25%)
        if tp2_hit:
            if (is_long and candle.high >= trade.tp3) or (not is_long and candle.low <= trade.tp3):
                exit_price = trade.tp3 * (1 - cfg['slippage_bps'] / 10000 if is_long else 1 + cfg['slippage_bps'] / 10000)
                if is_long:
                    pnl = (exit_price - trade.entry_price) * remaining_size / trade.entry_price
                else:
                    pnl = (trade.entry_price - exit_price) * remaining_size / trade.entry_price
                total_pnl += pnl
                trade.exit_time = candle.timestamp
                trade.exit_price = exit_price
                trade.exit_reason = 'TP3'
                break

    # Timeout - close at current price
    if trade.exit_time == 0:
        candle = candles[min(start_idx + cfg['max_hold_candles'], len(candles) - 1)]
        exit_price = candle.close * (1 - cfg['slippage_bps'] / 10000 if is_long else 1 + cfg['slippage_bps'] / 10000)
        if is_long:
            pnl = (exit_price - trade.entry_price) * remaining_size / trade.entry_price
        else:
            pnl = (trade.entry_price - exit_price) * remaining_size / trade.entry_price
        total_pnl += pnl
        trade.exit_time = candle.timestamp
        trade.exit_price = exit_price
        trade.exit_reason = 'TIMEOUT'

    # Calculate fees
    entry_fee = trade.position_size * cfg['taker_fee']
    exit_fee = trade.position_size * cfg['taker_fee']
    trade.fees = entry_fee + exit_fee

    trade.pnl = total_pnl - trade.fees
    trade.pnl_percent = (trade.pnl / trade.position_size) * 100

    return trade


# ═══════════════════════════════════════════════════════════════
# DATA FETCHING
# ═══════════════════════════════════════════════════════════════

def fetch_candles_ccxt(symbol: str, timeframe: str, days: int) -> List[Candle]:
    """Fetch historical candles using ccxt"""
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
                    timestamp=row[0],
                    open=row[1],
                    high=row[2],
                    low=row[3],
                    close=row[4],
                    volume=row[5]
                ))

            if len(ohlcv) < 1000:
                break

            since = ohlcv[-1][0] + 1
            time.sleep(0.1)  # Rate limit

        except Exception as e:
            print(f"Error: {e}")
            break

    print(f"{len(all_candles)} candles")
    return all_candles


# ═══════════════════════════════════════════════════════════════
# MAIN BACKTEST
# ═══════════════════════════════════════════════════════════════

def run_backtest(days: int, cfg: Dict) -> List[Trade]:
    """Run backtest on all symbols"""

    all_trades = []

    print(f"\n{'='*60}")
    print(f"MOMENTUM SCALPER BACKTEST")
    print(f"{'='*60}")
    print(f"Days: {days}")
    print(f"Timeframe: {cfg['timeframe']}")
    print(f"Symbols: {len(SYMBOLS)}")
    print(f"{'='*60}\n")

    for symbol in SYMBOLS:
        candles = fetch_candles_ccxt(symbol, cfg['timeframe'], days)

        if len(candles) < cfg['lookback_candles'] + 10:
            print(f"  {symbol}: Not enough data")
            continue

        symbol_trades = []
        cooldown_until = 0
        i = cfg['lookback_candles']

        while i < len(candles) - cfg['max_hold_candles']:
            candle = candles[i]

            # Cooldown check (1 candle = 5 mins)
            if candle.timestamp < cooldown_until:
                i += 1
                continue

            # Analyze
            window = candles[i - cfg['lookback_candles']:i + 1]
            analysis = analyze_candles(window, cfg)

            # Check entry
            should_enter, reason = check_entry(analysis, cfg)

            if should_enter:
                direction = analysis['direction']
                entry_price = candle.close * (1 + cfg['slippage_bps'] / 10000 if direction == 'LONG' else 1 - cfg['slippage_bps'] / 10000)

                # Calculate stops and targets
                stop_dist = entry_price * (cfg['stop_loss_pct'] / 100)
                tp1_dist = entry_price * (cfg['tp1_pct'] / 100)
                tp2_dist = entry_price * (cfg['tp2_pct'] / 100)
                tp3_dist = entry_price * (cfg['tp3_pct'] / 100)

                if direction == 'LONG':
                    stop_loss = entry_price - stop_dist
                    tp1 = entry_price + tp1_dist
                    tp2 = entry_price + tp2_dist
                    tp3 = entry_price + tp3_dist
                else:
                    stop_loss = entry_price + stop_dist
                    tp1 = entry_price - tp1_dist
                    tp2 = entry_price - tp2_dist
                    tp3 = entry_price - tp3_dist

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
                    features={
                        'regime': analysis['regime'],
                        'atr_percent': analysis['atr_percent'],
                        'bb_position': analysis['bb_position'],
                        'rsi_value': analysis['rsi_value'],
                        'macd_histogram': analysis['macd_histogram'],
                        'vwap_deviation': analysis['vwap_deviation'],
                        'vwap_deviation_std': analysis['vwap_deviation_std'],
                        'volume_ratio': analysis['volume_ratio'],
                        'body_ratio': analysis['body_ratio'],
                        'strength': analysis['strength'],
                        'kill_zone': analysis['kill_zone'],
                        'is_kill_zone': analysis['is_kill_zone'],
                        'ema_aligned': analysis['ema_aligned'],
                        'price_above_vwap': analysis['price_above_vwap'],
                        'macd_bullish_cross': analysis['macd_bullish_cross'],
                        'macd_bearish_cross': analysis['macd_bearish_cross'],
                    }
                )

                # Simulate trade
                trade = simulate_trade(trade, candles, i, cfg)
                symbol_trades.append(trade)

                # Set cooldown (skip to after trade exit)
                i += trade.holding_candles + 1
                cooldown_until = trade.exit_time + 60000  # 1 min cooldown

            else:
                i += 1

        print(f"  {symbol}: {len(symbol_trades)} trades")
        all_trades.extend(symbol_trades)

    return all_trades


def trades_to_dataframe(trades: List[Trade]) -> pd.DataFrame:
    """Convert trades to DataFrame for ML training"""

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
            **t.features
        }
        rows.append(row)

    return pd.DataFrame(rows)


def print_summary(trades: List[Trade]):
    """Print backtest summary"""

    if not trades:
        print("\nNo trades generated!")
        return

    df = trades_to_dataframe(trades)

    print(f"\n{'='*60}")
    print("BACKTEST RESULTS")
    print(f"{'='*60}")

    total_trades = len(df)
    wins = (df['outcome'] == 'WIN').sum()
    losses = total_trades - wins
    win_rate = wins / total_trades * 100

    total_pnl = df['pnl'].sum()
    avg_pnl = df['pnl'].mean()
    avg_winner = df[df['outcome'] == 'WIN']['pnl'].mean() if wins > 0 else 0
    avg_loser = df[df['outcome'] == 'LOSS']['pnl'].mean() if losses > 0 else 0

    print(f"Total Trades: {total_trades}")
    print(f"Wins: {wins} | Losses: {losses} | Win Rate: {win_rate:.1f}%")
    print(f"Total PnL: ${total_pnl:,.2f}")
    print(f"Avg PnL: ${avg_pnl:.2f}")
    print(f"Avg Winner: ${avg_winner:.2f} | Avg Loser: ${avg_loser:.2f}")

    # By direction
    print(f"\nBy Direction:")
    for d in ['long', 'short']:
        sub = df[df['direction'] == d]
        if len(sub) > 0:
            wr = (sub['outcome'] == 'WIN').mean() * 100
            pnl = sub['pnl'].sum()
            print(f"  {d.upper()}: {len(sub)} trades | {wr:.1f}% win | ${pnl:,.2f}")

    # By regime
    print(f"\nBy Regime:")
    for r in ['TREND', 'RANGE']:
        sub = df[df['regime'] == r]
        if len(sub) > 0:
            wr = (sub['outcome'] == 'WIN').mean() * 100
            pnl = sub['pnl'].sum()
            print(f"  {r}: {len(sub)} trades | {wr:.1f}% win | ${pnl:,.2f}")

    # By symbol
    print(f"\nBy Symbol (Top 5):")
    symbol_pnl = df.groupby('symbol')['pnl'].sum().sort_values(ascending=False)
    for symbol in symbol_pnl.head(5).index:
        sub = df[df['symbol'] == symbol]
        wr = (sub['outcome'] == 'WIN').mean() * 100
        print(f"  {symbol}: {len(sub)} trades | {wr:.1f}% win | ${symbol_pnl[symbol]:,.2f}")


def main():
    parser = argparse.ArgumentParser(description='Momentum Scalper Backtester')
    parser.add_argument('--days', type=int, default=365, help='Days of historical data')
    parser.add_argument('--output', '-o', default='data/h2o-training/momentum_backtest.csv',
                        help='Output CSV path')
    args = parser.parse_args()

    if not CCXT_AVAILABLE:
        print("ERROR: ccxt not installed. Run: pip install ccxt")
        sys.exit(1)

    # Run backtest
    trades = run_backtest(args.days, CONFIG)

    # Print summary
    print_summary(trades)

    # Save to CSV
    if trades:
        df = trades_to_dataframe(trades)
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False)
        print(f"\nSaved: {args.output}")
        print(f"Rows: {len(df)}")

    print("\nDone.")


if __name__ == '__main__':
    main()
