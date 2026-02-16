#!/usr/bin/env python3
"""
Technical Analysis Indicators Service
Provides TA indicators via subprocess for use in TypeScript
"""

import sys
import json
import pandas as pd
import numpy as np
from ta import add_all_ta_features
from typing import Dict, Any

def calculate_indicators(data: list) -> Dict[str, Any]:
    """
    Calculate various TA indicators from OHLCV data.
    
    Args:
        data: List of dictionaries with 'timestamp', 'open', 'high', 'low', 'close', 'volume'
    
    Returns:
        Dictionary with calculated indicators
    """
    if not data:
        return {}
    
    # Convert to DataFrame
    df = pd.DataFrame(data)
    
    # Ensure proper column names
    df.columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    # Convert types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df['timestamp'] = pd.to_numeric(df['timestamp'], errors='coerce')
    
    # Add all TA features
    df = add_all_ta_features(
        df,
        open="open",
        high="high",
        low="low",
        close="close",
        volume="volume",
        fillna=True
    )
    
    # Get the latest values (last row)
    latest = df.iloc[-1].to_dict()
    
    # Extract relevant indicators
    result = {
        # EMA (already in SMC indicators, but keeping for completeness)
        'ema_50': latest.get('ema_50', None),
        'ema_200': latest.get('ema_200', None),
        
        # MACD
        'macd': latest.get('macd', None),
        'macd_signal': latest.get('macd_signal', None),
        'macd_diff': latest.get('macd_diff', None),
        
        # RSI (already in SMC, but for completeness)
        'rsi': latest.get('rsi', None),
        
        # Bollinger Bands
        'bb_upper': latest.get('bb_bbm', None),  # Middle band (SMA)
        'bb_upper': latest.get('bb_bbh', None),  # Upper band
        'bb_lower': latest.get('bb_bbl', None),  # Lower band
        'bb_width': latest.get('bb_bbwidth', None),  # Band width
        'bb_pct': latest.get('bb_bbpercent', None),  # Price position within bands
        
        # Stochastic Oscillator
        'stoch_k': latest.get('stoch_k', None),
        'stoch_d': latest.get('stoch_d', None),
        
        # Average True Range (ATR)
        'atr': latest.get('atr_14', None),
        
        # Average Directional Index (ADX)
        'adx': latest.get('adx_14', None),
        'dmi': latest.get('dmi_14', None),  # +DI
        'dmi': latest.get('dmi_14', None),  # -DI
        
        # Volume indicators
        'volume_ma': latest.get('volume_sma_14', None),
        'volume_ratio': latest.get('volume_obv', None),
        
        # Price momentum
        'momentum': latest.get('mom_10', None),
        'roc': latest.get('roc_10', None),  # Rate of change
        
        # Volatility
        'volatility': latest.get('volatility_atr', None),
    }
    
    return result

def main():
    """
    Main entry point - reads JSON from stdin and writes to stdout.
    """
    try:
        # Read input data
        input_data = json.loads(sys.stdin.read())
        
        # Calculate indicators
        result = calculate_indicators(input_data)
        
        # Output as JSON
        print(json.dumps(result, indent=2))
        
    except Exception as e:
        error_result = {
            'error': str(e),
            'type': type(e).__name__
        }
        print(json.dumps(error_result, indent=2))
        sys.exit(1)

if __name__ == '__main__':
    main()