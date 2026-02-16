#!/bin/bash
# Ubuntu Installation Script for learning-orchestrator
# Run: sudo ./install-ubuntu.sh

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  learning-orchestrator - Ubuntu Installer                      ║"
echo "║  Installs Node.js, Python, LightGBM, and dependencies          ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""

# Check if running as root
if [ "$EUID" -ne 0 ]; then
  echo "❌ Please run with sudo: sudo ./install-ubuntu.sh"
  exit 1
fi

# Detect Ubuntu/Debian
if [ ! -f /etc/os-release ]; then
  echo "❌ Cannot detect OS. This script is for Ubuntu/Debian."
  exit 1
fi

. /etc/os-release
if [[ ! "$ID" =~ ^(ubuntu|debian)$ ]]; then
  echo "⚠️  Warning: This script is designed for Ubuntu/Debian."
  echo "   Your OS: $PRETTY_NAME"
  read -p "   Continue anyway? (y/N) " -n 1 -r
  echo
  if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    exit 1
  fi
fi

# Update system
echo "📦 Updating system packages..."
apt update && apt upgrade -y

# Install Node.js 20.x
echo "📦 Installing Node.js 20.x..."
if ! command -v node &> /dev/null; then
  curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
  apt install -y nodejs
else
  NODE_VERSION=$(node -v)
  echo "   ✅ Node.js already installed: $NODE_VERSION"
  MAJOR_VERSION=$(echo $NODE_VERSION | cut -d'v' -f2 | cut -d'.' -f1)
  if [ "$MAJOR_VERSION" -lt 20 ]; then
    echo "   ⚠️  Node.js 20+ required. Current: $NODE_VERSION"
    read -p "   Upgrade to Node.js 20? (y/N) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
      curl -fsSL https://deb.nodesource.com/setup_20.x | bash -
      apt install -y nodejs
    fi
  fi
fi

# Install Python and build tools
echo "📦 Installing Python and build tools..."
apt install -y python3 python3-pip python3-venv build-essential git curl

# Install PM2 for process management
echo "📦 Installing PM2 (for 24/7 operation)..."
if ! command -v pm2 &> /dev/null; then
  npm install -g pm2
else
  echo "   ✅ PM2 already installed"
fi

# Install LightGBM
echo "📦 Installing LightGBM (Python ML library)..."
if ! python3 -c "import lightgbm" 2>/dev/null; then
  # Try apt first (available in Ubuntu 24.04+)
  if apt install -y python3-lightgbm 2>/dev/null; then
    echo "   ✅ LightGBM installed via apt"
  else
    # Fallback: Use pip with --break-system-packages for externally-managed env
    pip3 install --break-system-packages lightgbm
  fi
else
  echo "   ✅ LightGBM already installed"
fi

# Check if we're in the learning-orchestrator directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ ! -f "$SCRIPT_DIR/package.json" ]; then
  echo "❌ package.json not found. Please run this script from the learning-orchestrator directory."
  exit 1
fi

# Install npm dependencies
echo "📦 Installing npm dependencies..."
cd "$SCRIPT_DIR"
npm install

# Build TypeScript
echo "🔨 Building TypeScript..."
npm run build

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
  echo "📝 Creating .env file from template..."
  cp .env.example .env
  echo "⚠️  IMPORTANT: Edit .env and add your Binance API keys!"
  echo "   nano .env"
fi

# Create data directories
echo "📁 Creating data directories..."
mkdir -p data/paper-trades-scalp
mkdir -p data/paper-trades-swing
mkdir -p data/models-paper-scalp
mkdir -p data/models-paper-swing

echo ""
echo "╔═══════════════════════════════════════════════════════════════╗"
echo "║  ✅ Installation Complete!                                     ║"
echo "╚═══════════════════════════════════════════════════════════════╝"
echo ""
echo "Next Steps:"
echo "─────────────────────────────────────────────────────────────────"
echo "1. Edit .env with your Binance API credentials:"
echo "   nano .env"
echo ""
echo "2. Run traders (foreground for testing):"
echo "   npm run paper-trade-multi-scalp"
echo "   npm run paper-trade-multi-swing"
echo ""
echo "3. Or run with PM2 (24/7 background):"
echo "   pm2 start npm --name 'scalp-trader' -- run paper-trade-multi-scalp"
echo "   pm2 start npm --name 'swing-trader' -- run paper-trade-multi-swing"
echo "   pm2 save"
echo "   pm2 startup  # Run the command it outputs to enable auto-start"
echo ""
echo "4. Monitor logs:"
echo "   pm2 logs scalp-trader"
echo "   pm2 logs swing-trader"
echo "   pm2 status"
echo "─────────────────────────────────────────────────────────────────"
echo ""
