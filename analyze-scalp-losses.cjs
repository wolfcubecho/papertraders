const fs = require('fs');
const path = require('path');

const tradesDir = path.join(__dirname, 'data', 'paper-trades-scalp');

function analyzeSymbol(symbol) {
  const filePath = path.join(tradesDir, `${symbol}.json`);
  if (!fs.existsSync(filePath)) return null;

  const data = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
  const trades = data.trades || [];
  const closed = trades.filter(t => t.status === 'CLOSED' && t.pnl !== undefined);

  return {
    symbol,
    total: closed.length,
    wins: closed.filter(t => t.pnl > 0).length,
    losses: closed.filter(t => t.pnl <= 0).length,
    winRate: closed.length > 0 ? (closed.filter(t => t.pnl > 0).length / closed.length) : 0,
    totalPnl: closed.reduce((s, t) => s + (t.pnl || 0), 0),
    trades: closed.slice(-20).reverse()  // Last 20 trades
  };
}

// Analyze worst performer
console.log('\n=== SCALP TRADER DIAGNOSTICS ===\n');

const symbols = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'NEARUSDT', 'XRPUSDT'];
const results = symbols.map(analyzeSymbol).filter(r => r);

// Sort by win rate
results.sort((a, b) => a.winRate - b.winRate);

console.log('WORST PERFORMER:\n');
const worst = results[0];
console.log(`${worst.symbol}: ${worst.wins}W/${worst.losses}L (${(worst.winRate * 100).toFixed(1)}%) | PnL: $${worst.totalPnl.toFixed(2)}\n`);

console.log('Last 20 trades (newest first):');
console.log('ID       | Dir  | EntryPrice | ExitPrice | PnL     | PnL%   | ExitReason       | Signals');
console.log('---------|------|------------|-----------|---------|--------|------------------|---------');

for (const t of worst.trades) {
  const id = t.id.slice(-7);
  const dir = (t.direction || '??').padEnd(4);
  const entry = t.entryPrice?.toFixed(5).padStart(10) || 'N/A';
  const exit = t.exitPrice?.toFixed(5).padStart(9) || 'N/A';
  const pnl = t.pnl?.toFixed(2).padStart(7) || 'N/A';
  const pnlPct = t.pnlPercent?.toFixed(2).padStart(6) || 'N/A';
  const reason = (t.exitReason || 'UNKNOWN').slice(0, 15).padEnd(15);
  const signals = (t.signals || []).slice(0, 4).join(',').slice(0, 14);
  const winLoss = t.pnl > 0 ? '✓' : '✗';

  console.log(`${id} | ${dir} | ${entry} | ${exit} | ${pnl} | ${pnlPct}% | ${reason} | ${signals} ${winLoss}`);
}

// Analyze by exit reason
console.log('\n\nEXIT REASON ANALYSIS:\n');
const exitReasons = {};
for (const t of worst.trades) {
  const reason = t.exitReason || 'UNKNOWN';
  if (!exitReasons[reason]) exitReasons[reason] = { wins: 0, losses: 0, totalPnl: 0 };
  exitReasons[reason].totalPnl += t.pnl || 0;
  if (t.pnl > 0) exitReasons[reason].wins++;
  else exitReasons[reason].losses++;
}

console.log('Exit Reason        | Wins | Losses | Win%  | Total PnL');
console.log('-------------------|------|--------|-------|-----------');
for (const [reason, stats] of Object.entries(exitReasons)) {
  const total = stats.wins + stats.losses;
  const winRate = ((stats.wins / total) * 100).toFixed(1);
  console.log(`${reason.padEnd(18)} | ${stats.wins.toString().padStart(4)} | ${stats.losses.toString().padStart(6)} | ${winRate.padStart(5)}% | ${stats.totalPnl.toFixed(2).padStart(9)}`);
}

// Analyze by regime
console.log('\n\nREGIME ANALYSIS:\n');
const regimes = {};
for (const t of worst.trades) {
  const regime = t.regime || 'UNKNOWN';
  if (!regimes[regime]) regimes[regime] = { wins: 0, losses: 0, totalPnl: 0 };
  regimes[regime].totalPnl += t.pnl || 0;
  if (t.pnl > 0) regimes[regime].wins++;
  else regimes[regime].losses++;
}

console.log('Regime  | Wins | Losses | Win%  | Total PnL');
console.log('--------|------|--------|-------|-----------');
for (const [regime, stats] of Object.entries(regimes)) {
  const total = stats.wins + stats.losses;
  const winRate = ((stats.wins / total) * 100).toFixed(1);
  console.log(`${regime.padEnd(7)} | ${stats.wins.toString().padStart(4)} | ${stats.losses.toString().padStart(6)} | ${winRate.padStart(5)}% | ${stats.totalPnl.toFixed(2).padStart(9)}`);
}
