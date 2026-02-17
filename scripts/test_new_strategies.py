#!/usr/bin/env python3
"""Test new trading strategies"""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import ccxt
import time
from dataclasses import dataclass

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

def fetch_candles(symbol, timeframe, days):
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


def momentum_breakout(days=180, timeframe='15m'):
    """
    MOMENTUM BREAKOUT:
    - LONG: Price breaks above 10-candle high with volume spike
    - SHORT: Price breaks below 10-candle low with volume spike
    - Trail stop using 3-candle low/high
    """
    all_trades = []

    for symbol in SYMBOLS:
        print(f'  {symbol}...', end=' ', flush=True)
        candles = fetch_candles(symbol, timeframe, days)
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        trades = []
        i = 15

        while i < len(candles) - 10:
            recent_high = np.max(highs[i-10:i])
            recent_low = np.min(lows[i-10:i])
            avg_vol = np.mean(volumes[i-10:i])

            curr = candles[i]
            vol_spike = curr.volume > avg_vol * 1.5

            # LONG: Break above recent high with volume
            if curr.close > recent_high and vol_spike and curr.close > curr.open:
                entry = curr.close

                for j in range(i+1, min(i+12, len(candles))):
                    trail_stop = np.min(lows[max(0,j-3):j]) if j >= 3 else recent_low
                    if closes[j] < trail_stop:
                        pnl = (closes[j] - entry) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'TRAIL'})
                        i = j + 1
                        break
                else:
                    pnl = (closes[min(i+11, len(candles)-1)] - entry) / entry * 1000 - 0.8
                    trades.append({'pnl': pnl, 'exit': 'TIMEOUT'})
                    i += 11
                continue

            # SHORT: Break below recent low with volume
            if curr.close < recent_low and vol_spike and curr.close < curr.open:
                entry = curr.close

                for j in range(i+1, min(i+12, len(candles))):
                    trail_stop = np.max(highs[max(0,j-3):j]) if j >= 3 else recent_high
                    if closes[j] > trail_stop:
                        pnl = (entry - closes[j]) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'TRAIL'})
                        i = j + 1
                        break
                else:
                    pnl = (entry - closes[min(i+11, len(candles)-1)]) / entry * 1000 - 0.8
                    trades.append({'pnl': pnl, 'exit': 'TIMEOUT'})
                    i += 11
                continue

            i += 1

        print(f'{len(trades)} trades')
        all_trades.extend(trades)

    return pd.DataFrame(all_trades)


def engulfing_candle(days=180, timeframe='15m'):
    """
    ENGULFING CANDLE:
    - Bullish: Previous red candle fully engulfed by current green candle + volume
    - Bearish: Previous green candle fully engulfed by current red candle + volume
    """
    all_trades = []

    for symbol in SYMBOLS:
        print(f'  {symbol}...', end=' ', flush=True)
        candles = fetch_candles(symbol, timeframe, days)
        closes = np.array([c.close for c in candles])
        opens = np.array([c.open for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        trades = []
        i = 20

        while i < len(candles) - 8:
            prev = candles[i-1]
            curr = candles[i]
            avg_vol = np.mean(volumes[i-10:i])

            prev_body = abs(prev.close - prev.open)
            curr_body = abs(curr.close - curr.open)

            # Bullish engulfing
            bullish_engulf = (prev.close < prev.open and
                            curr.close > curr.open and
                            curr.close > prev.open and
                            curr.open < prev.close and
                            curr_body > prev_body * 1.2 and
                            curr.volume > avg_vol * 1.3)

            # Bearish engulfing
            bearish_engulf = (prev.close > prev.open and
                            curr.close < curr.open and
                            curr.close < prev.open and
                            curr.open > prev.close and
                            curr_body > prev_body * 1.2 and
                            curr.volume > avg_vol * 1.3)

            if bullish_engulf:
                entry = curr.close
                sl = curr.low * 0.998
                tp = entry * 1.005

                for j in range(i+1, min(i+8, len(candles))):
                    if lows[j] <= sl:
                        pnl = (sl - entry) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'SL'})
                        i = j + 1
                        break
                    if highs[j] >= tp:
                        pnl = (tp - entry) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'TP'})
                        i = j + 1
                        break
                else:
                    pnl = (closes[min(i+7, len(candles)-1)] - entry) / entry * 1000 - 0.8
                    trades.append({'pnl': pnl, 'exit': 'TIMEOUT'})
                    i += 7
                continue

            if bearish_engulf:
                entry = curr.close
                sl = curr.high * 1.002
                tp = entry * 0.995

                for j in range(i+1, min(i+8, len(candles))):
                    if highs[j] >= sl:
                        pnl = (entry - sl) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'SL'})
                        i = j + 1
                        break
                    if lows[j] <= tp:
                        pnl = (entry - tp) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'TP'})
                        i = j + 1
                        break
                else:
                    pnl = (entry - closes[min(i+7, len(candles)-1)]) / entry * 1000 - 0.8
                    trades.append({'pnl': pnl, 'exit': 'TIMEOUT'})
                    i += 7
                continue

            i += 1

        print(f'{len(trades)} trades')
        all_trades.extend(trades)

    return pd.DataFrame(all_trades)


def vwap_bounce(days=180, timeframe='15m'):
    """
    VWAP BOUNCE:
    - Calculate session VWAP
    - LONG when price dips to VWAP from above in uptrend
    - SHORT when price rises to VWAP from below in downtrend
    """
    all_trades = []

    for symbol in SYMBOLS:
        print(f'  {symbol}...', end=' ', flush=True)
        candles = fetch_candles(symbol, timeframe, days)
        closes = np.array([c.close for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])
        volumes = np.array([c.volume for c in candles])

        # Calculate rolling VWAP (50 period)
        typical_price = (highs + lows + closes) / 3
        vwap = np.zeros_like(closes)
        for i in range(50, len(closes)):
            cum_pv = np.sum(typical_price[i-50:i+1] * volumes[i-50:i+1])
            cum_vol = np.sum(volumes[i-50:i+1])
            vwap[i] = cum_pv / cum_vol if cum_vol > 0 else closes[i]

        # Simple trend: EMA20 > EMA50 = uptrend
        ema20 = np.zeros_like(closes)
        ema50 = np.zeros_like(closes)
        ema20[19] = np.mean(closes[:20])
        ema50[49] = np.mean(closes[:50])
        for i in range(20, len(closes)):
            ema20[i] = (closes[i] - ema20[i-1]) * (2/21) + ema20[i-1]
        for i in range(50, len(closes)):
            ema50[i] = (closes[i] - ema50[i-1]) * (2/51) + ema50[i-1]

        trades = []
        i = 55

        while i < len(candles) - 8:
            curr = candles[i]
            prev = candles[i-1]

            uptrend = ema20[i] > ema50[i]
            downtrend = ema20[i] < ema50[i]

            # Price near VWAP (within 0.1%)
            near_vwap = abs(curr.close - vwap[i]) / vwap[i] < 0.001

            # LONG: Uptrend + price bounces off VWAP
            if uptrend and near_vwap and curr.close > curr.open and prev.close < vwap[i-1]:
                entry = curr.close
                sl = vwap[i] * 0.997
                tp = entry * 1.004

                for j in range(i+1, min(i+8, len(candles))):
                    if lows[j] <= sl:
                        pnl = (sl - entry) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'SL'})
                        i = j + 1
                        break
                    if highs[j] >= tp:
                        pnl = (tp - entry) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'TP'})
                        i = j + 1
                        break
                else:
                    pnl = (closes[min(i+7, len(candles)-1)] - entry) / entry * 1000 - 0.8
                    trades.append({'pnl': pnl, 'exit': 'TIMEOUT'})
                    i += 7
                continue

            # SHORT: Downtrend + price rejects from VWAP
            if downtrend and near_vwap and curr.close < curr.open and prev.close > vwap[i-1]:
                entry = curr.close
                sl = vwap[i] * 1.003
                tp = entry * 0.996

                for j in range(i+1, min(i+8, len(candles))):
                    if highs[j] >= sl:
                        pnl = (entry - sl) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'SL'})
                        i = j + 1
                        break
                    if lows[j] <= tp:
                        pnl = (entry - tp) / entry * 1000 - 0.8
                        trades.append({'pnl': pnl, 'exit': 'TP'})
                        i = j + 1
                        break
                else:
                    pnl = (entry - closes[min(i+7, len(candles)-1)]) / entry * 1000 - 0.8
                    trades.append({'pnl': pnl, 'exit': 'TIMEOUT'})
                    i += 7
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
    print(f'\n{name}:')
    print(f'  Trades: {total} | Win Rate: {wins/total*100:.1f}%')
    print(f'  Total PnL: ${df["pnl"].sum():.2f}')
    for ex in df['exit'].unique():
        sub = df[df['exit'] == ex]
        print(f'    {ex}: {len(sub)} | {(sub["pnl"]>0).mean()*100:.0f}% win | ${sub["pnl"].sum():.0f}')
    gp = df[df['pnl'] > 0]['pnl'].sum()
    gl = abs(df[df['pnl'] < 0]['pnl'].sum())
    if gl > 0:
        print(f'  Profit Factor: {gp/gl:.2f}')


if __name__ == '__main__':
    print('=' * 60)
    print('TESTING NEW STRATEGIES (15m, 180 days)')
    print('=' * 60)

    print('\n>>> MOMENTUM BREAKOUT <<<')
    df1 = momentum_breakout(180, '15m')
    print_results('Momentum Breakout', df1)

    print('\n>>> ENGULFING CANDLE <<<')
    df2 = engulfing_candle(180, '15m')
    print_results('Engulfing Candle', df2)

    print('\n>>> VWAP BOUNCE <<<')
    df3 = vwap_bounce(180, '15m')
    print_results('VWAP Bounce', df3)
