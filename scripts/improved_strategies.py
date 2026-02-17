#!/usr/bin/env python3
"""
Improved Trading Strategies - Testing Better Entry Timing

1. Pullback Entry: Wait for pullback after signal before entering
2. Multi-Timeframe: 15m signal, 5m entry
3. Structure-based stops: Use recent swing high/low instead of fixed %
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional

SYMBOLS = ['BTC/USDT', 'ETH/USDT', 'SOL/USDT', 'BNB/USDT', 'XRP/USDT',
           'DOGE/USDT', 'ADA/USDT', 'AVAX/USDT', 'LINK/USDT', 'DOT/USDT']

@dataclass
class Candle:
    timestamp: int
    open: float
    high: float
    low: float
    close: float
    volume: float


def fetch_candles(symbol: str, timeframe: str, days: int) -> List[Candle]:
    exchange = ccxt.binance({'enableRateLimit': True})
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_candles = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=1000)
        if not ohlcv:
            break
        for row in ohlcv:
            all_candles.append(Candle(row[0], row[1], row[2], row[3], row[4], row[5]))
        if len(ohlcv) < 1000:
            break
        since = ohlcv[-1][0] + 1
        time.sleep(0.05)
    return all_candles


def calculate_bb(closes, period=20, std=2):
    upper = np.zeros_like(closes)
    lower = np.zeros_like(closes)
    middle = np.zeros_like(closes)
    for i in range(period - 1, len(closes)):
        window = closes[i-period+1:i+1]
        mean = np.mean(window)
        std_dev = np.std(window)
        middle[i] = mean
        upper[i] = mean + std * std_dev
        lower[i] = mean - std * std_dev
    return upper, middle, lower


def calculate_rsi(closes, period=14):
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
    return 100 - (100 / (1 + rs))


def calculate_ema(closes, period):
    ema = np.zeros_like(closes)
    mult = 2 / (period + 1)
    ema[period-1] = np.mean(closes[:period])
    for i in range(period, len(closes)):
        ema[i] = (closes[i] - ema[i-1]) * mult + ema[i-1]
    return ema


# ═══════════════════════════════════════════════════════════════
# STRATEGY 1: PULLBACK ENTRY
# Signal: BB extreme + RSI extreme
# Entry: Wait for price to pull back 50% of the signal candle
# ═══════════════════════════════════════════════════════════════

def pullback_entry_strategy(days=365, timeframe='15m'):
    """
    Instead of entering immediately on signal, wait for a pullback.

    LONG signal: Price at lower BB + RSI < 30 + bullish candle
    LONG entry: Wait for price to pull back to 50% of signal candle range

    This gets us a better entry price and confirms the reversal has legs.
    """
    all_trades = []

    for symbol in SYMBOLS:
        print(f'  {symbol}...', end=' ', flush=True)
        candles = fetch_candles(symbol, timeframe, days)
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        opens = np.array([c.open for c in candles])

        bb_upper, bb_middle, bb_lower = calculate_bb(closes, 20, 2)
        rsi = calculate_rsi(closes, 14)

        trades = []
        i = 30

        while i < len(candles) - 15:
            curr = candles[i]
            prev_close = closes[i-1]

            # Check for signal (don't enter yet)
            # LONG signal: Previous close at lower BB + RSI < 30, current is bullish
            long_signal = (prev_close <= bb_lower[i-1] and
                          rsi[i-1] <= 30 and
                          curr.close > curr.open)

            # SHORT signal: Previous close at upper BB + RSI > 70, current is bearish
            short_signal = (prev_close >= bb_upper[i-1] and
                           rsi[i-1] >= 70 and
                           curr.close < curr.open)

            if long_signal:
                # Define pullback level: 50% retracement of signal candle
                signal_high = curr.high
                signal_low = curr.low
                pullback_level = signal_low + (signal_high - signal_low) * 0.5

                # Wait up to 3 candles for pullback
                entry_price = None
                entry_idx = None

                for j in range(i+1, min(i+4, len(candles))):
                    # Check if price pulls back to our level
                    if lows[j] <= pullback_level:
                        entry_price = pullback_level
                        entry_idx = j
                        break

                if entry_price:
                    # Enter and hold for 10 candles
                    sl = signal_low * 0.998  # Stop below signal candle

                    for k in range(entry_idx+1, min(entry_idx+11, len(candles))):
                        # Check stop
                        if lows[k] <= sl:
                            pnl = (sl - entry_price) / entry_price * 1000 - 0.8
                            trades.append({'pnl': pnl, 'exit': 'SL', 'dir': 'LONG'})
                            i = k + 1
                            break
                        # Check target (middle BB)
                        if closes[k] >= bb_middle[k]:
                            pnl = (closes[k] - entry_price) / entry_price * 1000 - 0.8
                            trades.append({'pnl': pnl, 'exit': 'TARGET', 'dir': 'LONG'})
                            i = k + 1
                            break
                    else:
                        # Timeout
                        exit_price = closes[min(entry_idx+10, len(candles)-1)]
                        pnl = (exit_price - entry_price) / entry_price * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'TIMEOUT', 'dir': 'LONG'})
                        i = entry_idx + 10
                    continue
                else:
                    # No pullback happened, skip
                    i += 1
                    continue

            if short_signal:
                signal_high = curr.high
                signal_low = curr.low
                pullback_level = signal_high - (signal_high - signal_low) * 0.5

                entry_price = None
                entry_idx = None

                for j in range(i+1, min(i+4, len(candles))):
                    if highs[j] >= pullback_level:
                        entry_price = pullback_level
                        entry_idx = j
                        break

                if entry_price:
                    sl = signal_high * 1.002

                    for k in range(entry_idx+1, min(entry_idx+11, len(candles))):
                        if highs[k] >= sl:
                            pnl = (entry_price - sl) / entry_price * 1000 - 0.8
                            trades.append({'pnl': pnl, 'exit': 'SL', 'dir': 'SHORT'})
                            i = k + 1
                            break
                        if closes[k] <= bb_middle[k]:
                            pnl = (entry_price - closes[k]) / entry_price * 1000 - 0.8
                            trades.append({'pnl': pnl, 'exit': 'TARGET', 'dir': 'SHORT'})
                            i = k + 1
                            break
                    else:
                        exit_price = closes[min(entry_idx+10, len(candles)-1)]
                        pnl = (entry_price - exit_price) / entry_price * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'TIMEOUT', 'dir': 'SHORT'})
                        i = entry_idx + 10
                    continue
                else:
                    i += 1
                    continue

            i += 1

        print(f'{len(trades)} trades')
        all_trades.extend(trades)

    return pd.DataFrame(all_trades)


# ═══════════════════════════════════════════════════════════════
# STRATEGY 2: MULTI-TIMEFRAME
# Signal on 15m, Entry on 5m
# ═══════════════════════════════════════════════════════════════

def multi_timeframe_strategy(days=180):
    """
    Use 15m for signal direction, 5m for precise entry.

    15m: Identify BB extreme + RSI extreme (signal)
    5m: Wait for reversal candle pattern to enter
    """
    all_trades = []

    for symbol in SYMBOLS:
        print(f'  {symbol}...', end=' ', flush=True)

        # Fetch both timeframes
        candles_15m = fetch_candles(symbol, '15m', days)
        candles_5m = fetch_candles(symbol, '5m', days)

        closes_15m = np.array([c.close for c in candles_15m])
        bb_upper_15m, bb_middle_15m, bb_lower_15m = calculate_bb(closes_15m, 20, 2)
        rsi_15m = calculate_rsi(closes_15m, 14)

        closes_5m = np.array([c.close for c in candles_5m])
        highs_5m = np.array([c.high for c in candles_5m])
        lows_5m = np.array([c.low for c in candles_5m])
        opens_5m = np.array([c.open for c in candles_5m])

        # Map 15m candle index to 5m candle index (approx 3:1 ratio)
        def get_5m_idx(idx_15m):
            ts = candles_15m[idx_15m].timestamp
            for j, c in enumerate(candles_5m):
                if c.timestamp >= ts:
                    return j
            return len(candles_5m) - 1

        trades = []
        i = 30  # 15m index

        while i < len(candles_15m) - 5:
            prev_close = closes_15m[i-1]

            # 15m LONG signal
            long_signal = prev_close <= bb_lower_15m[i-1] and rsi_15m[i-1] <= 30
            # 15m SHORT signal
            short_signal = prev_close >= bb_upper_15m[i-1] and rsi_15m[i-1] >= 70

            if long_signal:
                # Switch to 5m to find entry
                idx_5m = get_5m_idx(i)

                # Look for bullish 5m candle in next 6 candles (30 min window)
                for j in range(idx_5m, min(idx_5m + 6, len(candles_5m) - 15)):
                    c5 = candles_5m[j]
                    # Bullish candle with decent body
                    if c5.close > c5.open and (c5.close - c5.open) > (c5.high - c5.low) * 0.5:
                        entry = c5.close
                        sl = c5.low * 0.998

                        # Hold for 15 5m-candles (75 min)
                        for k in range(j+1, min(j+16, len(candles_5m))):
                            if lows_5m[k] <= sl:
                                pnl = (sl - entry) / entry * 1000 - 0.8
                                trades.append({'pnl': pnl, 'exit': 'SL', 'dir': 'LONG'})
                                break
                            # Target: back to 15m middle BB (use current 15m value)
                            curr_15m_idx = min(i + (k - idx_5m) // 3, len(candles_15m) - 1)
                            if closes_5m[k] >= bb_middle_15m[curr_15m_idx]:
                                pnl = (closes_5m[k] - entry) / entry * 1000 - 0.8
                                trades.append({'pnl': pnl, 'exit': 'TARGET', 'dir': 'LONG'})
                                break
                        else:
                            pnl = (closes_5m[min(j+15, len(candles_5m)-1)] - entry) / entry * 1000 - 0.8
                            trades.append({'pnl': pnl, 'exit': 'TIMEOUT', 'dir': 'LONG'})
                        break

                i += 2  # Skip ahead on 15m
                continue

            if short_signal:
                idx_5m = get_5m_idx(i)

                for j in range(idx_5m, min(idx_5m + 6, len(candles_5m) - 15)):
                    c5 = candles_5m[j]
                    if c5.close < c5.open and (c5.open - c5.close) > (c5.high - c5.low) * 0.5:
                        entry = c5.close
                        sl = c5.high * 1.002

                        for k in range(j+1, min(j+16, len(candles_5m))):
                            if highs_5m[k] >= sl:
                                pnl = (entry - sl) / entry * 1000 - 0.8
                                trades.append({'pnl': pnl, 'exit': 'SL', 'dir': 'SHORT'})
                                break
                            curr_15m_idx = min(i + (k - idx_5m) // 3, len(candles_15m) - 1)
                            if closes_5m[k] <= bb_middle_15m[curr_15m_idx]:
                                pnl = (entry - closes_5m[k]) / entry * 1000 - 0.8
                                trades.append({'pnl': pnl, 'exit': 'TARGET', 'dir': 'SHORT'})
                                break
                        else:
                            pnl = (entry - closes_5m[min(j+15, len(candles_5m)-1)]) / entry * 1000 - 0.8
                            trades.append({'pnl': pnl, 'exit': 'TIMEOUT', 'dir': 'SHORT'})
                        break

                i += 2
                continue

            i += 1

        print(f'{len(trades)} trades')
        all_trades.extend(trades)

    return pd.DataFrame(all_trades)


# ═══════════════════════════════════════════════════════════════
# STRATEGY 3: STRUCTURE-BASED STOPS
# Use swing highs/lows for stops instead of fixed %
# ═══════════════════════════════════════════════════════════════

def find_swing_points(highs, lows, lookback=5):
    """Find swing highs and lows"""
    swing_highs = np.zeros_like(highs)
    swing_lows = np.zeros_like(lows)

    for i in range(lookback, len(highs) - lookback):
        # Swing high: higher than surrounding candles
        if highs[i] == np.max(highs[i-lookback:i+lookback+1]):
            swing_highs[i] = highs[i]
        # Swing low
        if lows[i] == np.min(lows[i-lookback:i+lookback+1]):
            swing_lows[i] = lows[i]

    return swing_highs, swing_lows


def structure_stop_strategy(days=365, timeframe='15m'):
    """
    Mean reversion with structure-based stops.

    LONG: Stop below most recent swing low
    SHORT: Stop above most recent swing high

    This respects market structure instead of arbitrary % stops.
    """
    all_trades = []

    for symbol in SYMBOLS:
        print(f'  {symbol}...', end=' ', flush=True)
        candles = fetch_candles(symbol, timeframe, days)
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])

        bb_upper, bb_middle, bb_lower = calculate_bb(closes, 20, 2)
        rsi = calculate_rsi(closes, 14)
        swing_highs, swing_lows = find_swing_points(highs, lows, lookback=3)

        trades = []
        i = 30

        while i < len(candles) - 12:
            curr = candles[i]
            prev_close = closes[i-1]

            # Find most recent swing low (for LONG stop)
            recent_swing_low = None
            for j in range(i-1, max(i-20, 0), -1):
                if swing_lows[j] > 0:
                    recent_swing_low = swing_lows[j]
                    break

            # Find most recent swing high (for SHORT stop)
            recent_swing_high = None
            for j in range(i-1, max(i-20, 0), -1):
                if swing_highs[j] > 0:
                    recent_swing_high = swing_highs[j]
                    break

            # LONG signal
            long_signal = (prev_close <= bb_lower[i-1] and
                          rsi[i-1] <= 30 and
                          curr.close > curr.open)

            if long_signal and recent_swing_low:
                entry = curr.close
                sl = recent_swing_low * 0.999  # Just below swing low

                # Skip if stop is too far (>2% risk)
                risk_pct = (entry - sl) / entry * 100
                if risk_pct > 2:
                    i += 1
                    continue

                for j in range(i+1, min(i+12, len(candles))):
                    if lows[j] <= sl:
                        pnl = (sl - entry) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'SL', 'dir': 'LONG', 'risk': risk_pct})
                        i = j + 1
                        break
                    if closes[j] >= bb_middle[j]:
                        pnl = (closes[j] - entry) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'TARGET', 'dir': 'LONG', 'risk': risk_pct})
                        i = j + 1
                        break
                else:
                    exit_price = closes[min(i+11, len(candles)-1)]
                    pnl = (exit_price - entry) / entry * 1000 - 0.8
                    trades.append({'pnl': pnl, 'exit': 'TIMEOUT', 'dir': 'LONG', 'risk': risk_pct})
                    i += 11
                continue

            # SHORT signal
            short_signal = (prev_close >= bb_upper[i-1] and
                           rsi[i-1] >= 70 and
                           curr.close < curr.open)

            if short_signal and recent_swing_high:
                entry = curr.close
                sl = recent_swing_high * 1.001

                risk_pct = (sl - entry) / entry * 100
                if risk_pct > 2:
                    i += 1
                    continue

                for j in range(i+1, min(i+12, len(candles))):
                    if highs[j] >= sl:
                        pnl = (entry - sl) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'SL', 'dir': 'SHORT', 'risk': risk_pct})
                        i = j + 1
                        break
                    if closes[j] <= bb_middle[j]:
                        pnl = (entry - closes[j]) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'TARGET', 'dir': 'SHORT', 'risk': risk_pct})
                        i = j + 1
                        break
                else:
                    exit_price = closes[min(i+11, len(candles)-1)]
                    pnl = (entry - exit_price) / entry * 1000 - 0.8
                    trades.append({'pnl': pnl, 'exit': 'TIMEOUT', 'dir': 'SHORT', 'risk': risk_pct})
                    i += 11
                continue

            i += 1

        print(f'{len(trades)} trades')
        all_trades.extend(trades)

    return pd.DataFrame(all_trades)


def print_results(name, df):
    if len(df) == 0:
        print(f'{name}: No trades')
        return

    wins = (df['pnl'] > 0).sum()
    total = len(df)

    print(f'\n{"="*60}')
    print(f'{name}')
    print(f'{"="*60}')
    print(f'Trades: {total} | Win Rate: {wins/total*100:.1f}%')
    print(f'Total PnL: ${df["pnl"].sum():.2f}')

    gp = df[df['pnl'] > 0]['pnl'].sum()
    gl = abs(df[df['pnl'] < 0]['pnl'].sum())
    if gl > 0:
        print(f'Profit Factor: {gp/gl:.2f}')

    if wins > 0 and (total - wins) > 0:
        print(f'Avg Win: ${df[df["pnl"] > 0]["pnl"].mean():.2f} | Avg Loss: ${abs(df[df["pnl"] < 0]["pnl"].mean()):.2f}')

    print('\nBy Exit:')
    for ex in df['exit'].unique():
        sub = df[df['exit'] == ex]
        wr = (sub['pnl'] > 0).mean() * 100
        print(f'  {ex}: {len(sub)} | {wr:.0f}% win | ${sub["pnl"].sum():.0f}')

    if 'dir' in df.columns:
        print('\nBy Direction:')
        for d in df['dir'].unique():
            sub = df[df['dir'] == d]
            wr = (sub['pnl'] > 0).mean() * 100
            print(f'  {d}: {len(sub)} | {wr:.0f}% win | ${sub["pnl"].sum():.0f}')


if __name__ == '__main__':
    print('='*60)
    print('TESTING IMPROVED ENTRY STRATEGIES')
    print('='*60)

    print('\n>>> STRATEGY 1: PULLBACK ENTRY <<<')
    df1 = pullback_entry_strategy(days=365, timeframe='15m')
    print_results('Pullback Entry (15m)', df1)

    print('\n>>> STRATEGY 2: MULTI-TIMEFRAME <<<')
    df2 = multi_timeframe_strategy(days=180)
    print_results('Multi-TF (15m signal, 5m entry)', df2)

    print('\n>>> STRATEGY 3: STRUCTURE-BASED STOPS <<<')
    df3 = structure_stop_strategy(days=365, timeframe='15m')
    print_results('Structure Stops (15m)', df3)
