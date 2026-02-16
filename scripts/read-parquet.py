#!/usr/bin/env python3
"""
Read parquet file and output as JSON for Node.js
"""

import sys
import json
import pandas as pd

def main():
    if len(sys.argv) < 2:
        print(json.dumps({"error": "No parquet file specified"}))
        sys.exit(1)
    
    parquet_file = sys.argv[1]
    
    try:
        # Read parquet file
        df = pd.read_parquet(parquet_file)
        
        # Convert to list of dictionaries
        # Columns are indexed: 0=timestamp, 1=open, 2=high, 3=low, 4=close, 5=volume
        candles = []
        for _, row in df.iterrows():
            candle = {
                "timestamp": int(row.iloc[0]),
                "open": float(row.iloc[1]),
                "high": float(row.iloc[2]),
                "low": float(row.iloc[3]),
                "close": float(row.iloc[4]),
                "volume": float(row.iloc[5])
            }
            candles.append(candle)
        
        # Output as JSON
        result = {
            "success": True,
            "count": len(candles),
            "candles": candles
        }
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            "success": False,
            "error": str(e)
        }
        print(json.dumps(error_result))
        sys.exit(1)

if __name__ == "__main__":
    main()