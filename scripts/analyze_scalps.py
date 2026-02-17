#!/usr/bin/env python3
"""Analyze scalper trades to find what makes winners"""
import pandas as pd
import numpy as np

df = pd.read_csv('data/h2o-training/paper_scalp_2026-02-04T04-07-42.csv')

print('=== SCALPER TRADE ANALYSIS ===')
wins_count = (df['outcome']=='WIN').sum()
total = len(df)
print(f'Total: {total} | Wins: {wins_count} ({wins_count/total*100:.1f}%) | Losses: {total-wins_count}')
print(f'Total PnL: ${df["pnl"].sum():.2f}')
print(f'Avg Winner: ${df[df["outcome"]=="WIN"]["pnl"].mean():.2f}')
print(f'Avg Loser: ${df[df["outcome"]=="LOSS"]["pnl"].mean():.2f}')

wins = df[df['outcome'] == 'WIN']
losses = df[df['outcome'] == 'LOSS']

print('\n=== WHAT MAKES WINNING SCALPS? ===')
key_features = ['rsi_value', 'volatility', 'volume_ratio', 'bb_position', 'bb_width',
                'body_ratio', 'trend_strength', 'momentum_score', 'ob_distance', 'atr',
                'distance_to_high', 'distance_to_low', 'ema_distance']

for col in key_features:
    if col in df.columns:
        w = wins[col].mean()
        l = losses[col].mean()
        diff_pct = ((w - l) / l * 100) if l != 0 else 0
        marker = '***' if abs(diff_pct) > 15 else ''
        print(f'{marker}{col:25} Win:{w:8.3f} Loss:{l:8.3f} ({diff_pct:+.1f}%){marker}')

print('\n=== BY DIRECTION ===')
for d in ['long', 'short']:
    sub = df[df['direction'] == d]
    if len(sub) > 0:
        wr = (sub['outcome']=='WIN').mean()*100
        pnl = sub['pnl'].sum()
        print(f'{d.upper():6}: {len(sub):3} trades | {wr:.1f}% win | ${pnl:.2f} PnL')

print('\n=== BY RSI ZONE ===')
df['rsi_zone'] = pd.cut(df['rsi_value'], bins=[0, 30, 45, 55, 70, 100], labels=['<30', '30-45', '45-55', '55-70', '>70'])
for zone in ['<30', '30-45', '45-55', '55-70', '>70']:
    sub = df[df['rsi_zone'] == zone]
    if len(sub) >= 5:
        wr = (sub['outcome']=='WIN').mean()*100
        print(f'RSI {zone:6}: {len(sub):3} trades | {wr:.1f}% win')

print('\n=== BY VOLATILITY ===')
vol_median = df['volatility'].median()
high_vol = df[df['volatility'] > vol_median]
low_vol = df[df['volatility'] <= vol_median]
print(f'High Vol: {len(high_vol)} trades | {(high_vol["outcome"]=="WIN").mean()*100:.1f}% win')
print(f'Low Vol:  {len(low_vol)} trades | {(low_vol["outcome"]=="WIN").mean()*100:.1f}% win')

print('\n=== RECOMMENDATION ===')
# Find best filters
best_rsi = df.groupby('rsi_zone').apply(lambda x: (x['outcome']=='WIN').mean()).idxmax()
print(f'Best RSI zone: {best_rsi}')
