/**
 * Local Data Loader
 * Loads historical candle data from CSV files
 */

import { Candle } from './smc-indicators.js';
import * as fs from 'fs';
import * as path from 'path';
import { exec } from 'child_process';

export interface DataLoadResult {
  symbol: string;
  interval: string;
  candles: Candle[];
  loaded: number;
  total: number;
}

export class LocalDataLoader {
  private dataPath: string;
  
  constructor(dataPath: string = './Historical_Data_Lite') {
    this.dataPath = dataPath;
  }
  
  /**
   * Find data file for symbol/timeframe - tries multiple patterns
   * Priority: lite folder -> parquet (archive) -> csv (filtered) -> csv (flat)
   */
  private findDataFile(symbol: string, interval: string): { path: string; type: 'parquet' | 'csv' } | null {
    // Get base path (remove /archive suffix if present for lite folder check)
    const basePath = this.dataPath.replace(/[\/\\]archive$/, '');

    const possiblePaths: { path: string; type: 'parquet' | 'csv' }[] = [
      // Historical_Data_Lite folder (highest priority - smaller, top 10 only)
      { path: path.join(basePath, '..', 'Historical_Data_Lite', interval, `${symbol}_${interval}.parquet`), type: 'parquet' },
      { path: path.join(this.dataPath, '..', 'Historical_Data_Lite', interval, `${symbol}_${interval}.parquet`), type: 'parquet' },

      // Parquet files from archive
      { path: path.join(this.dataPath, 'kilnes_TRADING', interval, 'TRADING', `${symbol}_${interval}.parquet`), type: 'parquet' },

      // Flat structure: Binance_BTCUSDT_1h.csv
      { path: path.join(this.dataPath, `Binance_${symbol}_${interval}.csv`), type: 'csv' },
      // Flat structure: BTCUSDT_1h.csv
      { path: path.join(this.dataPath, `${symbol}_${interval}.csv`), type: 'csv' },
      // Hierarchical: symbol/interval/BTCUSDT_1h.csv
      { path: path.join(this.dataPath, symbol, interval, `${symbol}_${interval}.csv`), type: 'csv' },
      // Hierarchical: symbol/interval/Binance_BTCUSDT_1h.csv
      { path: path.join(this.dataPath, symbol, interval, `Binance_${symbol}_${interval}.csv`), type: 'csv' },
      // With filtered/ prefix
      { path: path.join(this.dataPath, 'filtered', symbol, interval, `${symbol}_${interval}.csv`), type: 'csv' },
      { path: path.join(this.dataPath, 'filtered', symbol, interval, `Binance_${symbol}_${interval}.csv`), type: 'csv' },
    ];
    
    for (const filePath of possiblePaths) {
      if (fs.existsSync(filePath.path)) {
        return filePath;
      }
    }
    
    return null;
  }
  
  /**
   * Load candle data from CSV file
   * CSV format: timestamp,open,high,low,close,volume
   */
  private async loadCSV(fullPath: string): Promise<Candle[]> {
    
    return new Promise((resolve, reject) => {
      fs.readFile(fullPath, 'utf8', (err, data) => {
        if (err) {
          reject(err);
          return;
        }
        
        try {
          const lines = data.trim().split('\n');
          const candles: Candle[] = [];
          
          // Skip header row
          for (let i = 1; i < lines.length; i++) {
            const line = lines[i].trim();
            if (!line) continue;
            
            const parts = line.split(',');
            if (parts.length < 5) continue;
            
            const candle: Candle = {
              timestamp: parseInt(parts[0]),
              open: parseFloat(parts[1]),
              high: parseFloat(parts[2]),
              low: parseFloat(parts[3]),
              close: parseFloat(parts[4]),
              volume: parseFloat(parts[5])
            };
            
            // Validate data
            if (candle.open > 0 && candle.high > 0 && candle.low > 0 && candle.close > 0) {
              candles.push(candle);
            }
          }
          
          resolve(candles);
        } catch (error) {
          reject(error);
        }
      });
    });
  }
  
  /**
   * Load candle data from Parquet file using Python script
   */
  private async loadParquet(fullPath: string): Promise<Candle[]> {
    return new Promise((resolve, reject) => {
      // Get script path - resolve relative to current working directory
      const scriptPath = path.join(process.cwd(), 'scripts', 'read-parquet.py');
      
      exec(`python "${scriptPath}" "${fullPath}"`, { maxBuffer: 100 * 1024 * 1024 }, (error, stdout, stderr) => {
        if (error) {
          reject(new Error(`Failed to read parquet: ${error.message}\n${stderr}`));
          return;
        }
        
        try {
          const result = JSON.parse(stdout);
          if (result.success) {
            resolve(result.candles);
          } else {
            reject(new Error(result.error || 'Failed to parse parquet'));
          }
        } catch (e) {
          reject(new Error(`Failed to parse Python output: ${e}`));
        }
      });
    });
  }
  
  /**
   * Load data for a specific symbol and interval
   */
  async loadData(symbol: string, interval: string): Promise<DataLoadResult> {
    // Find the data file
    const fileData = this.findDataFile(symbol, interval);
    
    if (!fileData) {
      throw new Error(
        `Data file not found for ${symbol} ${interval}\n` +
        `Looked in: ${this.dataPath}\n` +
        `Expected patterns: ${symbol}_${interval}.parquet or ${symbol}_${interval}.csv\n` +
        `Archive path: Historical_Data/archive/kilnes_TRADING/${interval}/TRADING/`
      );
    }
    
    console.log(`Found data file: ${fileData.path} (${fileData.type})`);
    
    let candles: Candle[];
    if (fileData.type === 'parquet') {
      // Convert to absolute path
      const absolutePath = path.resolve(fileData.path);
      candles = await this.loadParquet(absolutePath);
    } else {
      candles = await this.loadCSV(fileData.path);
    }
    
    return {
      symbol,
      interval,
      candles,
      loaded: candles.length,
      total: candles.length
    };
  }
  
  /**
   * Load data for multiple symbols
   */
  async loadMulti(symbols: string[], interval: string): Promise<Map<string, Candle[]>> {
    const results = new Map<string, Candle[]>();
    
    const promises = symbols.map(symbol =>
      this.loadData(symbol, interval)
        .then(result => results.set(symbol, result.candles))
        .catch(err => {
          console.warn(`Failed to load ${symbol}: ${err.message}`);
          return null;
        })
    );
    
    await Promise.all(promises);
    
    return results;
  }
  
  /**
   * Get available symbols and intervals
   */
  getAvailableData(): { symbols: string[]; intervals: string[] } {
    const symbols = new Set<string>();
    const intervals = new Set<string>();

    // Check directory structure
    const dataDir = this.dataPath;

    if (fs.existsSync(dataDir)) {
      const files = fs.readdirSync(dataDir);

      files.forEach(file => {
        if (file.endsWith('.csv')) {
          // Parse filename to extract symbol and interval
          const match = file.match(/(\w+)[_-](\d+[dhm])\.csv/);
          if (match) {
            symbols.add(match[1].replace('USDT', 'USDT')); // Normalize
            intervals.add(match[2]);
          }
        }
      });
    }

    return {
      symbols: Array.from(symbols),
      intervals: Array.from(intervals)
    };
  }

  /**
   * Fetch recent candles from Binance API
   * @param symbol Trading pair (e.g., BTCUSDT)
   * @param interval Timeframe (1m, 5m, 1h, 4h, 1d)
   * @param days Number of days to fetch (default 30)
   */
  async fetchRecentFromBinance(symbol: string, interval: string, days: number = 30): Promise<Candle[]> {
    const intervalMs: Record<string, number> = {
      '1m': 60 * 1000,
      '5m': 5 * 60 * 1000,
      '15m': 15 * 60 * 1000,
      '1h': 60 * 60 * 1000,
      '4h': 4 * 60 * 60 * 1000,
      '1d': 24 * 60 * 60 * 1000
    };

    const msPerCandle = intervalMs[interval] || intervalMs['1h'];
    const endTime = Date.now();
    const startTime = endTime - (days * 24 * 60 * 60 * 1000);
    const candlesNeeded = Math.ceil((endTime - startTime) / msPerCandle);

    console.log(`  Fetching ${candlesNeeded} ${interval} candles from Binance for ${symbol}...`);

    const allCandles: Candle[] = [];
    let currentStart = startTime;
    const limit = 1000;  // Binance max per request

    try {
      while (currentStart < endTime) {
        const url = `https://api.binance.com/api/v3/klines?symbol=${symbol}&interval=${interval}&startTime=${currentStart}&limit=${limit}`;

        const response = await fetch(url);
        if (!response.ok) {
          throw new Error(`Binance API error: ${response.status}`);
        }

        const data = await response.json() as any[];

        if (data.length === 0) break;

        for (const k of data) {
          allCandles.push({
            timestamp: k[0],
            open: parseFloat(k[1]),
            high: parseFloat(k[2]),
            low: parseFloat(k[3]),
            close: parseFloat(k[4]),
            volume: parseFloat(k[5])
          });
        }

        // Move to next batch
        currentStart = data[data.length - 1][0] + msPerCandle;

        // Rate limiting - be nice to Binance
        await new Promise(r => setTimeout(r, 100));
      }

      console.log(`  Fetched ${allCandles.length} recent candles from Binance`);
      return allCandles;

    } catch (error: any) {
      console.error(`  Failed to fetch from Binance: ${error.message}`);
      return [];
    }
  }

  /**
   * Load data with optional Binance recent data merge
   * Combines historical files with fresh API data
   */
  async loadWithRecentData(
    symbol: string,
    interval: string,
    recentDays: number = 30
  ): Promise<DataLoadResult> {
    // Load historical data first
    const historical = await this.loadData(symbol, interval);

    // Fetch recent from Binance
    const recent = await this.fetchRecentFromBinance(symbol, interval, recentDays);

    if (recent.length === 0) {
      return historical;
    }

    // Merge: historical + recent (deduplicate by timestamp)
    const seen = new Set<number>();
    const merged: Candle[] = [];

    // Add historical first
    for (const c of historical.candles) {
      if (!seen.has(c.timestamp)) {
        seen.add(c.timestamp);
        merged.push(c);
      }
    }

    // Add recent (will override/append)
    for (const c of recent) {
      if (!seen.has(c.timestamp)) {
        seen.add(c.timestamp);
        merged.push(c);
      }
    }

    // Sort by timestamp
    merged.sort((a, b) => a.timestamp - b.timestamp);

    console.log(`  Merged: ${historical.candles.length} historical + ${recent.length} recent = ${merged.length} total`);

    return {
      symbol,
      interval,
      candles: merged,
      loaded: merged.length,
      total: merged.length
    };
  }
}