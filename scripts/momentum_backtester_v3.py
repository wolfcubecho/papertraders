#!/usr/bin/env python3
"""
Momentum Scalper Backtester V3

NEW REGIME DETECTION (matches traders):
- MOMENTUM: ADX > 25 AND BB expanding (width > 1.2x avg) -> breakout entries
- RANGE: ADX < 20 AND BB normal (width < 1.5x avg) -> mean reversion
- NONE: Everything else -> skip trading

MOMENTUM MODE:
- Entry: Breakout close outside BB + VWAP aligned + Volume > 2x
- Management: No partial TP, trail at -1.0R, 10 candle time stop
- Exit: VWAP cross opposite, volume divergence, or trail hit

RANGE MODE:
- Entry: BB extreme (<25% or >75%) + Volume 1.5x (not 3x+) + Candle + OFI
- Management: TP1 70% at middle BB, TP2 30% at opposite BB
- Post-TP1: SL to +0.2R, Phase 2 trail at -0.5R
- Time stop: 5 candles

Usage:
    python scripts/momentum_backtester_v3.py --days 365
"""

import argparse
import os
import sys
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
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
# CONFIGURATION V3 - Matches TypeScript traders
# ═══════════════════════════════════════════════════════════════

SYMBOLS = [
    'BTC/USDT', 'ETH/USDT', 'BNB/USDT', 'SOL/USDT', 'XRP/USDT',
    'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT',
    'MATIC/USDT', 'ATOM/USDT', 'NEAR/USDT', 'ARB/USDT', 'INJ/USDT',
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
    'adx_period': 14,

    # REGIME DETECTION (NEW)
    'adx_momentum_min': 25,       # ADX > 25 = strong trend
    'adx_range_max': 20,          # ADX < 20 = weak/range-bound
    'bb_expanding_multiple': 1.2, # BB width > 1.2x avg = expanding
    'bb_normal_max_multiple': 1.5, # BB width < 1.5x avg = normal

    # MOMENTUM MODE (Breakout)
    'momentum_volume_mult': 2.0,  # 2x volume for momentum
    'momentum_trail_r': 1.0,      # Trail at -1R
    'momentum_trail_after_r': 0.5, # Start trailing after 0.5R profit
    'momentum_time_stop': 10,     # 10 candles max hold

    # RANGE MODE (Mean Reversion)
    'range_bb_extreme_low': 0.25,  # BB < 25% for LONG
    'range_bb_extreme_high': 0.75, # BB > 75% for SHORT
    'range_volume_mult': 1.5,      # 1.5x volume required
    'exhaustion_volume_mult': 3.0, # 3x+ = exhaustion, skip
    'tp1_close_pct': 0.70,         # Close 70% at TP1 (middle BB)
    'tp2_close_pct': 0.30,         # Close 30% at TP2 (opposite BB)
    'protected_profit_r': 0.2,     # After TP1, SL to +0.2R
    'phase2_trail_r': 0.5,         # Phase 2 trail at -0.5R
    'range_time_stop': 5,          # 5 candles max hold

    # RISK MANAGEMENT
    'sl_atr_mult': 1.5,            # Stop loss = 1.5x ATR
    'position_size': 1000,
    'taker_fee': 0.00025,
    'slippage_bps': 1,
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
class Indicators:
    """All indicators for a candle"""
    rsi: float
    ema_fast: float
    ema_slow: float
    bb_upper: float
    bb_middle: float
    bb_lower: float
    bb_position: float
    bb_width: float
    bb_width_avg: float
    bb_expanding: bool
    macd_line: float
    macd_signal: float
    macd_histogram: float
    atr: float
    atr_percent: float
    adx: float
    plus_di: float
    minus_di: float
    vwap: float
    price_above_vwap: bool
    volume_ratio: float
    regime: str  # 'MOMENTUM', 'RANGE', 'NONE'


@dataclass
class Trade:
    symbol: str
    direction: str
    regime: str  # 'MOMENTUM' or 'RANGE'
    entry_time: int
    entry_price: float
    stop_loss: float
    take_profit1: float  # Middle BB (RANGE mode)
    take_profit2: float  # Opposite BB (RANGE mode) or target (MOMENTUM mode)
    position_size: float
    entry_candle_idx: int
    atr_at_entry: float
    r_distance: float  # R = TP1 - Entry

    # State
    tp1_hit: bool = False
    trailing_stop: Optional[float] = None
    phase2_active: bool = False
    candles_held: int = 0

    # Exit
    exit_time: int = 0
    exit_price: float = 0
    exit_reason: str = ''
    pnl: float = 0
    pnl_percent: float = 0
    fees: float = 0


# ═══════════════════════════════════════════════════════════════
# INDICATORS
# ═══════════════════════════════════════════════════════════════

def calculate_ema(closes: np.ndarray, period: int) -> np.ndarray:
    ema = np.zeros_like(closes, dtype=float)
    multiplier = 2 / (period + 1)
    ema[period-1] = np.mean(closes[:period])
    for i in range(period, len(closes)):
        ema[i] = (closes[i] - ema[i-1]) * multiplier + ema[i-1]
    return ema


def calculate_rsi(closes: np.ndarray, period: int = 14) -> np.ndarray:
    rsi = np.zeros_like(closes, dtype=float)
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
    rsi[:period] = 50
    return rsi


def calculate_bb(closes: np.ndarray, period: int = 20, std: float = 2) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    middle = np.zeros_like(closes, dtype=float)
    upper = np.zeros_like(closes, dtype=float)
    lower = np.zeros_like(closes, dtype=float)

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
    atr = np.zeros_like(closes, dtype=float)
    tr = np.maximum(
        highs[1:] - lows[1:],
        np.maximum(
            abs(highs[1:] - closes[:-1]),
            abs(lows[1:] - closes[:-1])
        )
    )
    atr[period] = np.mean(tr[:period])
    for i in range(period + 1, len(closes)):
        atr[i] = (atr[i-1] * (period - 1) + tr[i-1]) / period
    return atr


def calculate_adx(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, period: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate ADX, +DI, -DI"""
    adx = np.zeros_like(closes, dtype=float)
    plus_di = np.zeros_like(closes, dtype=float)
    minus_di = np.zeros_like(closes, dtype=float)

    # Calculate +DM and -DM
    plus_dm = np.zeros_like(closes, dtype=float)
    minus_dm = np.zeros_like(closes, dtype=float)

    for i in range(1, len(closes)):
        up_move = highs[i] - highs[i-1]
        down_move = lows[i-1] - lows[i]

        if up_move > down_move and up_move > 0:
            plus_dm[i] = up_move
        if down_move > up_move and down_move > 0:
            minus_dm[i] = down_move

    # Calculate ATR for DI calculation
    atr = calculate_atr(highs, lows, closes, period)

    # Smooth +DM and -DM
    smoothed_plus_dm = np.zeros_like(closes, dtype=float)
    smoothed_minus_dm = np.zeros_like(closes, dtype=float)

    smoothed_plus_dm[period] = np.sum(plus_dm[1:period+1])
    smoothed_minus_dm[period] = np.sum(minus_dm[1:period+1])

    for i in range(period + 1, len(closes)):
        smoothed_plus_dm[i] = smoothed_plus_dm[i-1] - (smoothed_plus_dm[i-1] / period) + plus_dm[i]
        smoothed_minus_dm[i] = smoothed_minus_dm[i-1] - (smoothed_minus_dm[i-1] / period) + minus_dm[i]

    # Calculate +DI and -DI
    with np.errstate(divide='ignore', invalid='ignore'):
        plus_di = np.where(atr > 0, (smoothed_plus_dm / atr) * 100, 0)
        minus_di = np.where(atr > 0, (smoothed_minus_dm / atr) * 100, 0)

    # Calculate DX
    dx = np.zeros_like(closes, dtype=float)
    with np.errstate(divide='ignore', invalid='ignore'):
        dx = np.where(
            (plus_di + minus_di) > 0,
            (abs(plus_di - minus_di) / (plus_di + minus_di)) * 100,
            0
        )

    # Smooth DX to get ADX
    adx[period*2-1] = np.mean(dx[period:period*2])
    for i in range(period * 2, len(closes)):
        adx[i] = (adx[i-1] * (period - 1) + dx[i]) / period

    return adx, plus_di, minus_di


def calculate_vwap(highs: np.ndarray, lows: np.ndarray, closes: np.ndarray, volumes: np.ndarray) -> np.ndarray:
    """Calculate VWAP"""
    typical_price = (highs + lows + closes) / 3
    cumulative_tp_vol = np.cumsum(typical_price * volumes)
    cumulative_vol = np.cumsum(volumes)
    with np.errstate(divide='ignore', invalid='ignore'):
        vwap = np.where(cumulative_vol > 0, cumulative_tp_vol / cumulative_vol, typical_price)
    return vwap


def calculate_volume_ratio(volumes: np.ndarray, period: int = 20) -> np.ndarray:
    """Calculate volume ratio vs average"""
    volume_ratio = np.zeros_like(volumes, dtype=float)
    for i in range(period, len(volumes)):
        avg_vol = np.mean(volumes[i-period:i])
        volume_ratio[i] = volumes[i] / avg_vol if avg_vol > 0 else 1
    return volume_ratio


def calculate_bb_width_avg(bb_upper: np.ndarray, bb_lower: np.ndarray, bb_middle: np.ndarray, lookback: int = 20) -> np.ndarray:
    """Calculate BB width average over lookback period"""
    bb_width = np.where(bb_middle > 0, ((bb_upper - bb_lower) / bb_middle) * 100, 0)
    bb_width_avg = np.zeros_like(bb_width, dtype=float)

    for i in range(lookback, len(bb_width)):
        bb_width_avg[i] = np.mean(bb_width[i-lookback:i])

    return bb_width, bb_width_avg


# ═══════════════════════════════════════════════════════════════
# REGIME DETECTION
# ═══════════════════════════════════════════════════════════════

def detect_regime(adx: float, bb_width: float, bb_width_avg: float) -> str:
    """
    Detect market regime based on ADX and BB width

    MOMENTUM: ADX > 25 AND BB expanding (width > 1.2x avg)
    RANGE: ADX < 20 AND BB normal (width < 1.5x avg)
    NONE: Everything else (transition/squeeze)
    """
    if bb_width_avg == 0:
        return 'NONE'

    bb_expanding = bb_width > bb_width_avg * CONFIG['bb_expanding_multiple']
    bb_normal = bb_width < bb_width_avg * CONFIG['bb_normal_max_multiple']

    if adx > CONFIG['adx_momentum_min'] and bb_expanding:
        return 'MOMENTUM'
    elif adx < CONFIG['adx_range_max'] and bb_normal:
        return 'RANGE'
    else:
        return 'NONE'


# ═══════════════════════════════════════════════════════════════
# BACKTESTER
# ═══════════════════════════════════════════════════════════════

class Backtester:
    def __init__(self):
        self.exchange = ccxt.binance({'enableRateLimit': True})
        self.trades: List[Trade] = []
        self.stats = {
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'by_regime': {
                'MOMENTUM': {'trades': 0, 'wins': 0, 'pnl': 0},
                'RANGE': {'trades': 0, 'wins': 0, 'pnl': 0},
            }
        }

    def fetch_candles(self, symbol: str, days: int) -> List[Candle]:
        """Fetch historical candles"""
        print(f"  Fetching {symbol} ({days} days)...")

        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        all_candles = []

        while since < datetime.now().timestamp() * 1000:
            candles = self.exchange.fetch_ohlcv(
                symbol,
                timeframe=CONFIG['timeframe'],
                since=since,
                limit=1000
            )
            if not candles:
                break
            all_candles.extend(candles)
            since = candles[-1][0] + 1

            if len(all_candles) > days * 24 * 12 * 2:  # 5m candles
                break

        return [
            Candle(
                timestamp=c[0],
                open=c[1],
                high=c[2],
                low=c[3],
                close=c[4],
                volume=c[5]
            )
            for c in all_candles
        ]

    def calculate_indicators(self, candles: List[Candle]) -> List[Indicators]:
        """Calculate all indicators for candles"""
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Calculate indicators
        rsi = calculate_rsi(closes, CONFIG['rsi_period'])
        ema_fast = calculate_ema(closes, CONFIG['ema_fast'])
        ema_slow = calculate_ema(closes, CONFIG['ema_slow'])
        bb_upper, bb_middle, bb_lower = calculate_bb(closes, CONFIG['bb_period'], CONFIG['bb_std'])
        macd_line, macd_signal, macd_histogram = calculate_macd(closes)
        atr = calculate_atr(highs, lows, closes, CONFIG['atr_period'])
        adx, plus_di, minus_di = calculate_adx(highs, lows, closes, CONFIG['adx_period'])
        vwap = calculate_vwap(highs, lows, closes, volumes)
        volume_ratio = calculate_volume_ratio(volumes)

        # BB width
        bb_width, bb_width_avg = calculate_bb_width_avg(bb_upper, bb_lower, bb_middle)

        indicators = []
        for i, candle in enumerate(candles):
            # BB position (0 = at lower, 0.5 = middle, 1 = at upper)
            bb_range = bb_upper[i] - bb_lower[i]
            bb_position = (candle.close - bb_lower[i]) / bb_range if bb_range > 0 else 0.5

            # ATR as % of price
            atr_percent = (atr[i] / candle.close) if candle.close > 0 else 0

            # Price above VWAP
            price_above_vwap = candle.close > vwap[i]

            # BB expanding
            bb_expanding = bb_width[i] > bb_width_avg[i] * CONFIG['bb_expanding_multiple'] if bb_width_avg[i] > 0 else False

            # Detect regime
            regime = detect_regime(adx[i], bb_width[i], bb_width_avg[i])

            indicators.append(Indicators(
                rsi=rsi[i],
                ema_fast=ema_fast[i],
                ema_slow=ema_slow[i],
                bb_upper=bb_upper[i],
                bb_middle=bb_middle[i],
                bb_lower=bb_lower[i],
                bb_position=bb_position,
                bb_width=bb_width[i],
                bb_width_avg=bb_width_avg[i],
                bb_expanding=bb_expanding,
                macd_line=macd_line[i],
                macd_signal=macd_signal[i],
                macd_histogram=macd_histogram[i],
                atr=atr[i],
                atr_percent=atr_percent,
                adx=adx[i],
                plus_di=plus_di[i],
                minus_di=minus_di[i],
                vwap=vwap[i],
                price_above_vwap=price_above_vwap,
                volume_ratio=volume_ratio[i],
                regime=regime
            ))

        return indicators

    def check_momentum_entry(self, candle: Candle, ind: Indicators, prev_candle: Candle, prev_ind: Indicators) -> Tuple[bool, str]:
        """Check MOMENTUM mode entry conditions"""
        direction = None

        # 1. VWAP alignment
        if not ind.price_above_vwap:
            return False, 'LONG'  # Price must be above VWAP for long momentum
        # Note: For shorts, price should be below VWAP

        # 2. Breakout (BB or price)
        # For simplicity, check BB breakout
        bb_breakout_up = candle.close > ind.bb_upper and prev_candle.close <= prev_ind.bb_upper
        bb_breakout_down = candle.close < ind.bb_lower and prev_candle.close >= prev_ind.bb_lower

        if bb_breakout_up:
            direction = 'LONG'
        elif bb_breakout_down:
            direction = 'SHORT'
        else:
            return False, None

        # 3. Volume > 2x
        if ind.volume_ratio < CONFIG['momentum_volume_mult']:
            return False, None

        return True, direction

    def check_range_entry(self, candle: Candle, ind: Indicators) -> Tuple[bool, str]:
        """Check RANGE mode entry conditions"""
        direction = None

        # 1. BB extreme
        if ind.bb_position < CONFIG['range_bb_extreme_low']:
            direction = 'LONG'
        elif ind.bb_position > CONFIG['range_bb_extreme_high']:
            direction = 'SHORT'
        else:
            return False, None

        # 2. Volume (1.5x required, but not 3x+ exhaustion)
        if ind.volume_ratio < CONFIG['range_volume_mult']:
            return False, None
        if ind.volume_ratio > CONFIG['exhaustion_volume_mult']:
            return False, None  # Exhaustion spike

        return True, direction

    def run_backtest(self, symbol: str, candles: List[Candle], indicators: List[Indicators]):
        """Run backtest for a symbol"""
        open_trade = None

        for i in range(50, len(candles)):  # Need warmup for indicators
            candle = candles[i]
            ind = indicators[i]
            prev_candle = candles[i-1]
            prev_ind = indicators[i-1]

            # Check open trade management
            if open_trade:
                self.manage_trade(open_trade, candle, ind)
                if open_trade.exit_reason:  # Trade closed
                    self.record_trade(open_trade)
                    open_trade = None
                continue

            # Check regime and entry
            if ind.regime == 'NONE':
                continue

            should_enter = False
            direction = None

            if ind.regime == 'MOMENTUM':
                should_enter, direction = self.check_momentum_entry(candle, ind, prev_candle, prev_ind)
            elif ind.regime == 'RANGE':
                should_enter, direction = self.check_range_entry(candle, ind)

            if should_enter and direction:
                open_trade = self.open_trade(symbol, candle, ind, direction, ind.regime)

    def open_trade(self, symbol: str, candle: Candle, ind: Indicators, direction: str, regime: str) -> Trade:
        """Open a new trade"""
        is_long = direction == 'LONG'

        # ATR-based stop loss
        sl_distance = ind.atr * CONFIG['sl_atr_mult']
        stop_loss = candle.close - sl_distance if is_long else candle.close + sl_distance

        # R distance (for MOMENTUM, use SL distance; for RANGE, use TP1 distance)
        if regime == 'RANGE':
            # TP1 = Middle BB
            tp1 = ind.bb_middle
            tp2 = ind.bb_upper if is_long else ind.bb_lower
            r_distance = abs(tp1 - candle.close)
        else:
            # MOMENTUM: R = SL distance
            tp1 = candle.close + sl_distance * 2 if is_long else candle.close - sl_distance * 2
            tp2 = candle.close + sl_distance * 3 if is_long else candle.close - sl_distance * 3
            r_distance = sl_distance

        return Trade(
            symbol=symbol,
            direction=direction,
            regime=regime,
            entry_time=candle.timestamp,
            entry_price=candle.close,
            stop_loss=stop_loss,
            take_profit1=tp1,
            take_profit2=tp2,
            position_size=CONFIG['position_size'],
            entry_candle_idx=0,
            atr_at_entry=ind.atr,
            r_distance=r_distance
        )

    def manage_trade(self, trade: Trade, candle: Candle, ind: Indicators):
        """Manage open trade"""
        trade.candles_held += 1
        is_long = trade.direction == 'LONG'

        # Check SL
        if is_long and candle.low <= trade.stop_loss:
            self.close_trade(trade, candle, trade.stop_loss, 'SL')
            return
        if not is_long and candle.high >= trade.stop_loss:
            self.close_trade(trade, candle, trade.stop_loss, 'SL')
            return

        # Check trailing stop
        if trade.trailing_stop:
            if is_long and candle.low <= trade.trailing_stop:
                self.close_trade(trade, candle, trade.trailing_stop, 'TRAIL')
                return
            if not is_long and candle.high >= trade.trailing_stop:
                self.close_trade(trade, candle, trade.trailing_stop, 'TRAIL')
                return

        # Regime-specific management
        if trade.regime == 'MOMENTUM':
            self.manage_momentum_trade(trade, candle, ind)
        else:
            self.manage_range_trade(trade, candle, ind)

    def manage_momentum_trade(self, trade: Trade, candle: Candle, ind: Indicators):
        """Manage MOMENTUM mode trade - trail only, no partial TPs"""
        is_long = trade.direction == 'LONG'

        # Calculate profit in R
        profit_r = (candle.close - trade.entry_price) / trade.r_distance if is_long else (trade.entry_price - candle.close) / trade.r_distance

        # Time stop
        if trade.candles_held >= CONFIG['momentum_time_stop']:
            self.close_trade(trade, candle, candle.close, 'TIME')
            return

        # Start trailing after 0.5R profit
        if profit_r >= CONFIG['momentum_trail_after_r']:
            trail_distance = CONFIG['momentum_trail_r'] * trade.r_distance
            new_trail = candle.close - trail_distance if is_long else candle.close + trail_distance

            # Only tighten
            if trade.trailing_stop is None:
                trade.trailing_stop = new_trail
            elif is_long and new_trail > trade.trailing_stop:
                trade.trailing_stop = new_trail
            elif not is_long and new_trail < trade.trailing_stop:
                trade.trailing_stop = new_trail

        # VWAP cross exit (optional - momentum reversal)
        if is_long and not ind.price_above_vwap and profit_r > 0.5:
            self.close_trade(trade, candle, candle.close, 'VWAP_CROSS')
            return
        if not is_long and ind.price_above_vwap and profit_r > 0.5:
            self.close_trade(trade, candle, candle.close, 'VWAP_CROSS')
            return

    def manage_range_trade(self, trade: Trade, candle: Candle, ind: Indicators):
        """Manage RANGE mode trade - TP1/TP2 partials"""
        is_long = trade.direction == 'LONG'

        # Time stop
        if trade.candles_held >= CONFIG['range_time_stop']:
            self.close_trade(trade, candle, candle.close, 'TIME')
            return

        # TP1 check (70% close at middle BB)
        if not trade.tp1_hit:
            if is_long and candle.high >= trade.take_profit1:
                trade.tp1_hit = True
                # Move SL to +0.2R
                protected_sl_distance = trade.r_distance * CONFIG['protected_profit_r']
                trade.stop_loss = trade.entry_price + protected_sl_distance
                # Reduce position by 70%
                trade.position_size *= (1 - CONFIG['tp1_close_pct'])
                # Record partial
                partial_pnl = (trade.take_profit1 - trade.entry_price) * trade.position_size / 0.3 * 0.7
                trade.pnl += partial_pnl
                return

            if not is_long and candle.low <= trade.take_profit1:
                trade.tp1_hit = True
                protected_sl_distance = trade.r_distance * CONFIG['protected_profit_r']
                trade.stop_loss = trade.entry_price - protected_sl_distance
                trade.position_size *= (1 - CONFIG['tp1_close_pct'])
                partial_pnl = (trade.entry_price - trade.take_profit1) * trade.position_size / 0.3 * 0.7
                trade.pnl += partial_pnl
                return

        # TP2 check (30% close at opposite BB)
        if trade.tp1_hit:
            # Phase 2 trailing
            tp2_distance = abs(trade.take_profit2 - trade.entry_price)
            phase2_trigger = trade.take_profit2 - 0.3 * tp2_distance if is_long else trade.take_profit2 + 0.3 * tp2_distance

            should_phase2 = (is_long and candle.close >= phase2_trigger) or (not is_long and candle.close <= phase2_trigger)

            if should_phase2:
                trade.phase2_active = True
                trail_distance = CONFIG['phase2_trail_r'] * trade.r_distance
                new_trail = candle.close - trail_distance if is_long else candle.close + trail_distance

                if trade.trailing_stop is None:
                    trade.trailing_stop = new_trail
                elif is_long and new_trail > trade.trailing_stop:
                    trade.trailing_stop = new_trail
                elif not is_long and new_trail < trade.trailing_stop:
                    trade.trailing_stop = new_trail

            # TP2 hit
            if is_long and candle.high >= trade.take_profit2:
                self.close_trade(trade, candle, trade.take_profit2, 'TP2')
                return
            if not is_long and candle.low <= trade.take_profit2:
                self.close_trade(trade, candle, trade.take_profit2, 'TP2')
                return

    def close_trade(self, trade: Trade, candle: Candle, exit_price: float, reason: str):
        """Close a trade"""
        trade.exit_time = candle.timestamp
        trade.exit_price = exit_price
        trade.exit_reason = reason

        # Calculate PnL
        is_long = trade.direction == 'LONG'
        if is_long:
            trade.pnl_percent = ((exit_price - trade.entry_price) / trade.entry_price) * 100
        else:
            trade.pnl_percent = ((trade.entry_price - exit_price) / trade.entry_price) * 100

        trade.pnl += trade.position_size * (trade.pnl_percent / 100)
        trade.fees = trade.position_size * CONFIG['taker_fee'] * 2  # Entry + exit

    def record_trade(self, trade: Trade):
        """Record closed trade in stats"""
        self.trades.append(trade)
        self.stats['total_trades'] += 1

        net_pnl = trade.pnl - trade.fees
        self.stats['total_pnl'] += net_pnl

        if net_pnl > 0:
            self.stats['wins'] += 1
        else:
            self.stats['losses'] += 1

        # By regime
        if trade.regime in self.stats['by_regime']:
            self.stats['by_regime'][trade.regime]['trades'] += 1
            self.stats['by_regime'][trade.regime]['pnl'] += net_pnl
            if net_pnl > 0:
                self.stats['by_regime'][trade.regime]['wins'] += 1

    def print_results(self):
        """Print backtest results"""
        print("\n" + "="*60)
        print("BACKTEST RESULTS V3 (MOMENTUM/RANGE/NONE)")
        print("="*60)

        print(f"\nOVERALL:")
        print(f"  Total Trades: {self.stats['total_trades']}")
        print(f"  Win Rate: {self.stats['wins'] / max(1, self.stats['total_trades']) * 100:.1f}%")
        print(f"  Total PnL: ${self.stats['total_pnl']:.2f}")
        print(f"  Avg PnL/Trade: ${self.stats['total_pnl'] / max(1, self.stats['total_trades']):.2f}")

        print(f"\nBY REGIME:")
        for regime, data in self.stats['by_regime'].items():
            if data['trades'] > 0:
                wr = data['wins'] / data['trades'] * 100
                avg_pnl = data['pnl'] / data['trades']
                print(f"  {regime}:")
                print(f"    Trades: {data['trades']}")
                print(f"    Win Rate: {wr:.1f}%")
                print(f"    Total PnL: ${data['pnl']:.2f}")
                print(f"    Avg PnL/Trade: ${avg_pnl:.2f}")

        # Save to CSV
        if self.trades:
            df = pd.DataFrame([
                {
                    'symbol': t.symbol,
                    'regime': t.regime,
                    'direction': t.direction,
                    'entry_time': datetime.fromtimestamp(t.entry_time/1000),
                    'entry_price': t.entry_price,
                    'exit_price': t.exit_price,
                    'exit_reason': t.exit_reason,
                    'pnl': t.pnl,
                    'pnl_percent': t.pnl_percent,
                    'holding_candles': t.candles_held,
                }
                for t in self.trades
            ])
            output_path = 'data/h2o-training/momentum_backtest_v3.csv'
            df.to_csv(output_path, index=False)
            print(f"\nTrades saved to: {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Momentum Backtester V3')
    parser.add_argument('--days', type=int, default=90, help='Days to backtest')
    parser.add_argument('--symbols', nargs='+', help='Specific symbols to test')
    args = parser.parse_args()

    backtester = Backtester()
    symbols = args.symbols if args.symbols else SYMBOLS

    print(f"Backtesting {len(symbols)} symbols over {args.days} days...")
    print(f"Regime Detection: ADX > {CONFIG['adx_momentum_min']} + BB expanding = MOMENTUM")
    print(f"                 ADX < {CONFIG['adx_range_max']} + BB normal = RANGE")
    print(f"                 Everything else = NONE (skip)")

    for symbol in symbols:
        try:
            candles = backtester.fetch_candles(symbol, args.days)
            if len(candles) < 100:
                print(f"  Skipping {symbol}: not enough candles")
                continue

            indicators = backtester.calculate_indicators(candles)
            backtester.run_backtest(symbol, candles, indicators)

        except Exception as e:
            print(f"  Error with {symbol}: {e}")
            continue

    backtester.print_results()


if __name__ == '__main__':
    main()
