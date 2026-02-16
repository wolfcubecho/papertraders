#!/usr/bin/env python3
"""Test confirmed reversal strategy"""
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

def calculate_bb(closes, period=20, std=2):
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

def fetch_candles(symbol, days=365):
    exchange = ccxt.binance({'enableRateLimit': True})
    since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
    all_candles = []
    while True:
        ohlcv = exchange.fetch_ohlcv(symbol, '5m', since=since, limit=1000)
        if not ohlcv: break
        for row in ohlcv:
            all_candles.append(Candle(row[0], row[1], row[2], row[3], row[4], row[5]))
        if len(ohlcv) < 1000: break
        since = ohlcv[-1][0] + 1
        time.sleep(0.05)
    return all_candles

def run_confirmed_reversal(days=365):
    """
    CONFIRMED REVERSAL STRATEGY:
    1. Previous candle touched BB extreme + RSI extreme
    2. Current candle is a REVERSAL candle
    3. Stop below reversal candle low
    4. Target: Middle BB
    """
    all_trades = []

    for symbol in SYMBOLS:
        print(f'  {symbol}...', end=' ', flush=True)
        candles = fetch_candles(symbol, days)
        closes = np.array([c.close for c in candles])
        opens = np.array([c.open for c in candles])
        highs = np.array([c.high for c in candles])
        lows = np.array([c.low for c in candles])

        bb_upper, bb_middle, bb_lower = calculate_bb(closes, 20, 2)
        rsi = calculate_rsi(closes, 14)

        trades = []
        i = 31

        while i < len(candles) - 15:
            prev_close = closes[i-1]
            curr_candle = candles[i]

            candle_range = curr_candle.high - curr_candle.low
            body_ratio = abs(curr_candle.close - curr_candle.open) / candle_range if candle_range > 0 else 0

            # LONG: Previous touched lower BB + RSI<30, current is bullish
            prev_touched_lower = prev_close <= bb_lower[i-1] and rsi[i-1] <= 30
            curr_bullish = curr_candle.close > curr_candle.open

            if prev_touched_lower and curr_bullish and body_ratio > 0.4:
                entry = curr_candle.close
                sl = curr_candle.low * 0.999

                for j in range(i+1, min(i+15, len(candles))):
                    if lows[j] <= sl:
                        pnl = (sl - entry) / entry * 1000 - 0.8
                        trades.append({'dir': 'LONG', 'pnl': pnl, 'exit': 'SL'})
                        i = j + 1
                        break
                    if closes[j] >= bb_middle[j]:
                        pnl = (closes[j] - entry) / entry * 1000 - 0.8
                        trades.append({'dir': 'LONG', 'pnl': pnl, 'exit': 'TARGET'})
                        i = j + 1
                        break
                else:
                    pnl = (closes[min(i+14, len(candles)-1)] - entry) / entry * 1000 - 0.8
                    trades.append({'dir': 'LONG', 'pnl': pnl, 'exit': 'TIMEOUT'})
                    i += 14
                continue

            # SHORT: Previous touched upper BB + RSI>70, current is bearish
            prev_touched_upper = prev_close >= bb_upper[i-1] and rsi[i-1] >= 70
            curr_bearish = curr_candle.close < curr_candle.open

            if prev_touched_upper and curr_bearish and body_ratio > 0.4:
                entry = curr_candle.close
                sl = curr_candle.high * 1.001

                for j in range(i+1, min(i+15, len(candles))):
                    if highs[j] >= sl:
                        pnl = (entry - sl) / entry * 1000 - 0.8
                        trades.append({'dir': 'SHORT', 'pnl': pnl, 'exit': 'SL'})
                        i = j + 1
                        break
                    if closes[j] <= bb_middle[j]:
                        pnl = (entry - closes[j]) / entry * 1000 - 0.8
                        trades.append({'dir': 'SHORT', 'pnl': pnl, 'exit': 'TARGET'})
                        i = j + 1
                        break
                else:
                    pnl = (entry - closes[min(i+14, len(candles)-1)]) / entry * 1000 - 0.8
                    trades.append({'dir': 'SHORT', 'pnl': pnl, 'exit': 'TIMEOUT'})
                    i += 14
                continue

            i += 1

        print(f'{len(trades)} trades')
        all_trades.extend(trades)

    return pd.DataFrame(all_trades)

if __name__ == '__main__':
    print('=' * 60)
    print('CONFIRMED REVERSAL STRATEGY')
    print('Wait for reversal candle after BB+RSI extreme touch')
    print('Stop below reversal candle | Target: Middle BB')
    print('=' * 60)

    df = run_confirmed_reversal(365)
    if len(df) > 0:
        wins = (df['pnl'] > 0).sum()
        total = len(df)
        print(f'\nTrades: {total} | Win Rate: {wins/total*100:.1f}%')
        print(f'Total PnL: ${df["pnl"].sum():.2f}')
        print(f'Avg PnL: ${df["pnl"].mean():.2f}')

        print('\nBy Exit:')
        for ex in ['TARGET', 'SL', 'TIMEOUT']:
            if ex in df['exit'].values:
                sub = df[df['exit']==ex]
                print(f'  {ex}: {len(sub)} trades | {(sub["pnl"]>0).mean()*100:.0f}% win | ${sub["pnl"].sum():.0f}')

        gross_profit = df[df['pnl'] > 0]['pnl'].sum()
        gross_loss = abs(df[df['pnl'] < 0]['pnl'].sum())
        pf = gross_profit / gross_loss if gross_loss > 0 else 0
        print(f'\nProfit Factor: {pf:.2f}')

        if wins > 0 and (total-wins) > 0:
            avg_win = df[df['pnl'] > 0]['pnl'].mean()
            avg_loss = abs(df[df['pnl'] < 0]['pnl'].mean())
            print(f'Avg Win: ${avg_win:.2f} | Avg Loss: ${avg_loss:.2f}')
