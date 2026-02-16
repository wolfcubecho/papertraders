# Learning Orchestrator

An advanced ML-powered trading strategy evolution system that uses genetic algorithms and H2O machine learning to continuously evolve trading strategies based on historical cryptocurrency data.

## Overview

This orchestrator implements an automated learning loop that:
1. **Fetches historical data** from Binance for multiple cryptocurrency pairs
2. **Evolves trading strategies** using genetic algorithms
3. **Trains ML models** using H2O (Random Forest and Gradient Boosting)
4. **Backtests strategies** to evaluate performance
5. **Iterates continuously** to improve strategies over time

## Features

- **Multi-timeframe analysis**: 1m, 5m, 1h, 1d intervals
- **Technical indicators**: RSI, MACD, Bollinger Bands, SMA/EMA, ATR, etc.
- **Genetic algorithm optimization**: Evolves strategy parameters
- **Ensemble ML voting**: Combines multiple model predictions
- **Reinforcement learning**: Position sizing based on ML confidence
- **Smart Money Concepts (SMC)**: Market structure analysis
- **Automated backtesting**: Weekly evaluation cycles

## Prerequisites

- Node.js 18+
- Python 3.8+
- H2O-3 (auto-installed via pip)
- npm

## Installation

### Ubuntu/Debian Server (Quick Install)

For fresh Ubuntu/Debian servers, use the automated installer:

```bash
# Clone and run the installer
git clone https://github.com/wolfcubecho/learning-orchestrator.git
cd learning-orchestrator
sudo ./install-ubuntu.sh
```

The installer will:
- Install Node.js 20.x, Python 3, LightGBM
- Install PM2 for 24/7 process management
- Build the project
- Create data directories
- Set up `.env` file (edit with your Binance API keys)

Then start the traders with PM2:
```bash
pm2 start npm --name 'scalp-trader' -- run paper-trade-multi-scalp
pm2 start npm --name 'swing-trader' -- run paper-trade-multi-swing
pm2 save
pm2 startup  # Run the command it outputs for auto-start on boot
```

### Option 1: Using Trading Stack Installer (Recommended)

The easiest way to install Learning Orchestrator is through the main trading-stack installer:

```bash
# Linux/macOS
./install.sh

# Windows
.\install.ps1
```

The installer will:
- Clone this repository
- Install Node.js dependencies
- Install Python dependencies (pyarrow, pandas, requests, h2o)
- Download historical data (see note below)
- Build the project

### Option 2: Manual Installation

```bash
# Clone the repository
git clone https://github.com/wolfcubecho/learning-orchestrator.git
cd learning-orchestrator

# Install Node.js dependencies
npm install

# Install Python dependencies
pip install pyarrow pandas requests h2o

# Download historical data (required)
python scripts/fetch-historical-data.py

# Build the project
npm run build
```

## ⚠️ Important: Historical Data

**Historical data is NOT included in this repository** due to GitHub's file size limits (100 MB per file). The data files (`.parquet` format) range from 68-110 MB each.

### How to Get Historical Data

**Automatic (via installer):**
The install scripts (`install.sh` and `install.ps1`) will automatically download historical data when run.

**Manual:**
Run the data fetch script:
```bash
python scripts/fetch-historical-data.py
```

This downloads historical data for the following pairs:
- BTC/USDT
- ETH/USDT
- BNB/USDT
- SOL/USDT
- ADA/USDT
- AVAX/USDT
- DOT/USDT
- LINK/USDT
- DOGE/USDT
- XRP/USDT

Data is saved in `Historical_Data_Lite/` with subdirectories for different timeframes.

## Usage

### Start Learning Loop

The main learning loop runs continuously, evolving strategies and training models:

```bash
npm run learn-loop
```

This will:
1. Fetch latest data
2. Generate trade signals
3. Evolve new strategies
4. Train ML models
5. Evaluate performance
6. Repeat every cycle

### CLI Commands

```bash
# Display CLI help
npm start -- --help

# Generate trade signals with current best model
npm start -- generate-signals

# Run a single evolution cycle
npm start -- evolve

# Train ML models on current data
npm start -- train-model

# Backtest a strategy
npm start -- backtest

# Evaluate current strategy performance
npm start -- evaluate
```

### Standalone Scripts

```bash
# Fetch historical data manually
python scripts/fetch-historical-data.py

# Read and display parquet data
python scripts/read-parquet.py

# Compute technical indicators
python scripts/ta-indicators.py

# Train H2O model
python scripts/h2o_trainer.py

# Generate predictions
python scripts/h2o_predict.py
```

## Project Structure

```
learning-orchestrator/
├── src/                          # TypeScript source code
│   ├── cli.ts                    # Command-line interface
│   ├── backtest-learn-loop.ts    # Main learning loop
│   ├── data-loader.ts            # Parquet data loading
│   ├── h2o-integration.ts        # H2O ML integration
│   ├── h2o-learning-service.ts   # H2O service layer
│   ├── ml-trade-advisor.ts       # ML-based trade advisor
│   ├── ensemble-voting.ts        # Ensemble model voting
│   ├── rl-position-sizer.ts      # RL-based position sizing
│   ├── smc-indicators.ts         # Smart Money Concepts
│   ├── trade-decision-pipeline.ts # Decision pipeline
│   └── types.ts                 # TypeScript types
├── scripts/                      # Python utility scripts
│   ├── fetch-historical-data.py  # Download Binance data
│   ├── h2o_trainer.py           # Train H2O models
│   ├── h2o_predict.py           # Generate predictions
│   ├── ta-indicators.py          # Compute indicators
│   └── read-parquet.py           # Read parquet files
├── config/                       # Configuration files
│   └── features.json             # Feature definitions
├── data/                         # Runtime data (gitignored)
│   ├── ml-results/              # ML predictions/results
│   ├── models/                   # Saved models
│   ├── evolved-strategies/       # Evolved strategies
│   └── learning/                # Learning state
├── Historical_Data_Lite/         # Historical data (gitignored)
│   ├── 1m/                      # 1-minute candles
│   ├── 5m/                      # 5-minute candles
│   ├── 1h/                      # 1-hour candles
│   └── 1d/                      # 1-day candles
├── dist/                         # Compiled JavaScript
├── package.json
├── tsconfig.json
└── .env.example                  # Environment variables template
```

## Configuration

Create a `.env` file based on `.env.example`:

```env
# Binance API (optional, for live data)
BINANCE_API_KEY=your_api_key
BINANCE_API_SECRET=your_api_secret
BINANCE_TESTNET=false

# Learning parameters
EVOLUTION_INTERVAL=7d    # Evolution cycle frequency
TRAINING_INTERVAL=1d     # Model retraining frequency
```

## How It Works

### 1. Data Loading
- Loads historical OHLCV data from parquet files
- Computes 50+ technical indicators
- Normalizes and prepares features

### 2. Strategy Evolution
- Uses genetic algorithm to evolve strategy parameters
- Mutation and crossover operations
- Selection based on backtest performance
- Maintains population of top strategies

### 3. ML Training
- Uses H2O AutoML for model selection
- Trains Random Forest and Gradient Boosting
- Ensemble voting across multiple models
- Features include technical indicators and market structure

### 4. Position Sizing
- Reinforcement learning for optimal position sizes
- Confidence-based sizing based on ML predictions
- Risk management with stop-loss/take-profit

### 5. Continuous Learning
- Weekly evolution cycles
- Daily model retraining
- Performance tracking and logging
- Automatic strategy improvement

## Integration with Trading Stack

Learning Orchestrator integrates with other MCP servers:

1. **Actions Bridge**: Provides ML trade recommendations via `get_ml_trade_advice` tool
2. **Binance/Bitget**: Receives trade signals from evolved strategies
3. **CLI**: Standalone tool for running learning experiments

Example MCP tool usage in Claude:
```
User: What's the ML trade advice for BTC/USDT?
Claude: [calls get_ml_trade_advice tool]
       BTC/USDT Analysis:
       - Signal: LONG
       - Confidence: 0.85
       - Entry: 43,500
       - Stop Loss: 42,800
       - Take Profit: 44,500
       - Position Size: 0.5
```

## Performance Tips

- Use SSD for faster data loading
- Allocate sufficient RAM for H2O (2GB+ recommended)
- Run learning loop during off-hours to avoid interference
- Monitor disk space (data can be several GB)

## Troubleshooting

### "Historical data not found"
Run: `python scripts/fetch-historical-data.py`

### H2O not found
Install: `pip install h2o`

### Port already in use
H2O uses port 54321 by default. Change in code if needed.

### Memory errors
Increase H2O memory allocation in `h2o-integration.ts`

## Contributing

Contributions welcome! Areas for improvement:
- Additional technical indicators
- Alternative ML models
- More cryptocurrency pairs
- Real-time data integration
- Enhanced backtesting metrics

## License

MIT License - See LICENSE file for details

## Acknowledgments

- H2O.ai for excellent ML framework
- CCXT for exchange integration
- Pandas and PyArrow for data handling