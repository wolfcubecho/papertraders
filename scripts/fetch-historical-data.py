#!/usr/bin/env python3
"""
Fetch historical kline data from Binance for ML training.
Run this on fresh installs to get training data.

Usage: python scripts/fetch-historical-data.py
"""

import os
import sys
import time
import requests
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path

# Configuration
SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'SOLUSDT',
           'XRPUSDT', 'DOGEUSDT', 'DOTUSDT', 'AVAXUSDT', 'LINKUSDT']
TIMEFRAMES = ['1d', '1h', '5m']
DAYS_HISTORY = {'1d': 1500, '1h': 365, '5m': 60}  # How many days back

BASE_URL = 'https://api.binance.com/api/v3/klines'
OUTPUT_DIR = Path(__file__).parent.parent / 'Historical_Data_Lite'


def fetch_klines(symbol: str, interval: str, start_time: int, end_time: int) -> list:
    """Fetch klines from Binance API"""
    params = {
        'symbol': symbol,
        'interval': interval,
        'startTime': start_time,
        'endTime': end_time,
        'limit': 1000
    }

    response = requests.get(BASE_URL, params=params)
    if response.status_code == 200:
        return response.json()
    else:
        print(f"  Error: {response.status_code} - {response.text[:100]}")
        return []


def fetch_all_klines(symbol: str, interval: str, days: int) -> pd.DataFrame:
    """Fetch all klines for a symbol/interval going back N days"""
    end_time = int(datetime.now().timestamp() * 1000)
    start_time = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)

    all_klines = []
    current_start = start_time

    while current_start < end_time:
        klines = fetch_klines(symbol, interval, current_start, end_time)
        if not klines:
            break

        all_klines.extend(klines)

        # Move start to after last candle
        current_start = klines[-1][0] + 1

        # Rate limiting
        time.sleep(0.1)

    if not all_klines:
        return pd.DataFrame()

    # Convert to DataFrame
    df = pd.DataFrame(all_klines, columns=[
        'timestamp', 'open', 'high', 'low', 'close', 'volume',
        'close_time', 'quote_volume', 'trades', 'taker_buy_base',
        'taker_buy_quote', 'ignore'
    ])

    # Convert types
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = df[col].astype(float)

    return df[['timestamp', 'open', 'high', 'low', 'close', 'volume']]


def main():
    print("=" * 60)
    print("Historical Data Fetcher for ML Training")
    print("=" * 60)
    print(f"\nSymbols: {len(SYMBOLS)}")
    print(f"Timeframes: {TIMEFRAMES}")
    print(f"Output: {OUTPUT_DIR}\n")

    # Create output directories
    for tf in TIMEFRAMES:
        (OUTPUT_DIR / tf).mkdir(parents=True, exist_ok=True)

    total = len(SYMBOLS) * len(TIMEFRAMES)
    completed = 0

    for tf in TIMEFRAMES:
        days = DAYS_HISTORY[tf]
        print(f"\n[{tf}] Fetching {days} days of data...")

        for symbol in SYMBOLS:
            completed += 1
            progress = int(completed / total * 100)
            print(f"  [{progress:3d}%] {symbol}...", end='', flush=True)

            try:
                df = fetch_all_klines(symbol, tf, days)

                if len(df) > 0:
                    # Save as parquet
                    output_path = OUTPUT_DIR / tf / f"{symbol}_{tf}.parquet"
                    df.to_parquet(output_path, index=False)
                    print(f" {len(df):,} candles")
                else:
                    print(" FAILED (no data)")

            except Exception as e:
                print(f" ERROR: {e}")

            # Rate limiting between symbols
            time.sleep(0.2)

    print("\n" + "=" * 60)
    print("Download complete!")
    print(f"Data saved to: {OUTPUT_DIR}")
    print("\nNext: Run 'npm run learn-loop' to train the ML model")
    print("=" * 60)


if __name__ == '__main__':
    main()
