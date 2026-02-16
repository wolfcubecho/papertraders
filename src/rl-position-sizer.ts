/**
 * Reinforcement Learning Position Sizer
 * Uses Q-Learning to dynamically size positions based on market conditions
 *
 * Inspired by Larry Williams Bot V4 but adapted for our SMC + ML stack
 *
 * Key differences from original:
 * - Conservative allocation (1-5% of portfolio)
 * - Higher leverage options (1x-10x)
 * - Integration with our ML confidence scores
 */

import fs from 'fs';
import path from 'path';

// State features for Q-learning
export interface MarketState {
  volatility: number;      // ATR as % of price (0-1)
  trendStrength: number;   // 0-1, from our SMC indicators
  recentWinRate: number;   // Last N trades win rate (0-1)
  currentDrawdown: number; // Current DD from peak (0-1)
  openPositions: number;   // Count of open positions
  mlConfidence: number;    // ML model confidence (0-1)
  smcScore: number;        // SMC confluence score (0-100)
}

// Position sizing decision
export interface PositionAction {
  allocationPct: number;   // % of portfolio (1-5%)
  leverage: number;        // 1x-10x
  effectiveSize: number;   // allocation * leverage
}

// Q-Learning configuration
export interface RLConfig {
  learningRate: number;    // Alpha - how fast to learn (0.1 default)
  discountFactor: number;  // Gamma - future reward importance (0.95)
  explorationRate: number; // Epsilon - exploration vs exploitation (0.1)
  minExploration: number;  // Minimum epsilon after decay (0.01)
  explorationDecay: number;// Decay rate per trade (0.995)

  // Risk limits
  maxAllocationPct: number;  // Max single position % (5%)
  maxLeverage: number;       // Max leverage (10x)
  maxEffectiveSize: number;  // Max allocation * leverage (15%)
  maxOpenPositions: number;  // Max concurrent positions (5)
}

const DEFAULT_CONFIG: RLConfig = {
  learningRate: 0.1,
  discountFactor: 0.95,
  explorationRate: 0.15,    // Start with 15% exploration
  minExploration: 0.02,
  explorationDecay: 0.998,

  maxAllocationPct: 5,
  maxLeverage: 10,
  maxEffectiveSize: 20,     // Max 20% effective exposure
  maxOpenPositions: 5,
};

// Discretization buckets for state
const VOLATILITY_BUCKETS = [0.01, 0.02, 0.04, 0.08];      // <1%, 1-2%, 2-4%, 4-8%, >8%
const TREND_BUCKETS = [0.3, 0.5, 0.7];                     // weak, moderate, strong, very strong
const WINRATE_BUCKETS = [0.3, 0.5, 0.7];                   // poor, ok, good, excellent
const DRAWDOWN_BUCKETS = [0.05, 0.1, 0.15, 0.25];         // <5%, 5-10%, 10-15%, 15-25%, >25%
const POSITIONS_BUCKETS = [1, 2, 3];                       // 0, 1, 2, 3+
const CONFIDENCE_BUCKETS = [0.4, 0.6, 0.8];               // low, medium, high, very high
const SMC_BUCKETS = [40, 60, 80];                          // weak, ok, strong, excellent

// Action space
const ALLOCATION_OPTIONS = [1, 2, 3, 4, 5];               // 1-5% of portfolio
const LEVERAGE_OPTIONS = [1, 2, 3, 5, 7, 10];             // Conservative to aggressive

export class RLPositionSizer {
  private config: RLConfig;
  private qTable: Map<string, Map<string, number>>;
  private epsilon: number;
  private statePath: string;
  private tradeHistory: Array<{
    state: string;
    action: string;
    reward: number;
    timestamp: number;
  }>;

  constructor(config: Partial<RLConfig> = {}) {
    this.config = { ...DEFAULT_CONFIG, ...config };
    this.qTable = new Map();
    this.epsilon = this.config.explorationRate;
    this.statePath = path.join(process.cwd(), 'data', 'models', 'rl_position_sizer.json');
    this.tradeHistory = [];

    this.loadState();
  }

  /**
   * Discretize continuous state into buckets for Q-table lookup
   */
  private discretizeState(state: MarketState): string {
    const volBucket = this.getBucket(state.volatility, VOLATILITY_BUCKETS);
    const trendBucket = this.getBucket(state.trendStrength, TREND_BUCKETS);
    const winBucket = this.getBucket(state.recentWinRate, WINRATE_BUCKETS);
    const ddBucket = this.getBucket(state.currentDrawdown, DRAWDOWN_BUCKETS);
    const posBucket = this.getBucket(state.openPositions, POSITIONS_BUCKETS);
    const confBucket = this.getBucket(state.mlConfidence, CONFIDENCE_BUCKETS);
    const smcBucket = this.getBucket(state.smcScore / 100, SMC_BUCKETS.map(b => b / 100));

    return `v${volBucket}_t${trendBucket}_w${winBucket}_d${ddBucket}_p${posBucket}_c${confBucket}_s${smcBucket}`;
  }

  private getBucket(value: number, buckets: number[]): number {
    for (let i = 0; i < buckets.length; i++) {
      if (value < buckets[i]) return i;
    }
    return buckets.length;
  }

  /**
   * Encode action as string for Q-table
   */
  private encodeAction(allocation: number, leverage: number): string {
    return `a${allocation}_l${leverage}`;
  }

  private decodeAction(actionStr: string): PositionAction {
    const match = actionStr.match(/a(\d+)_l(\d+)/);
    if (!match) {
      return { allocationPct: 2, leverage: 1, effectiveSize: 2 };
    }
    const allocation = parseInt(match[1]);
    const leverage = parseInt(match[2]);
    return {
      allocationPct: allocation,
      leverage: leverage,
      effectiveSize: allocation * leverage
    };
  }

  /**
   * Get all possible actions
   */
  private getAllActions(): string[] {
    const actions: string[] = [];
    for (const alloc of ALLOCATION_OPTIONS) {
      for (const lev of LEVERAGE_OPTIONS) {
        // Skip combinations that exceed max effective size
        if (alloc * lev <= this.config.maxEffectiveSize) {
          actions.push(this.encodeAction(alloc, lev));
        }
      }
    }
    return actions;
  }

  /**
   * Get Q-value for state-action pair
   */
  private getQValue(state: string, action: string): number {
    const stateActions = this.qTable.get(state);
    if (!stateActions) return 0;
    return stateActions.get(action) || 0;
  }

  /**
   * Set Q-value for state-action pair
   */
  private setQValue(state: string, action: string, value: number): void {
    if (!this.qTable.has(state)) {
      this.qTable.set(state, new Map());
    }
    this.qTable.get(state)!.set(action, value);
  }

  /**
   * Select action using epsilon-greedy policy
   */
  selectAction(marketState: MarketState): PositionAction {
    const stateKey = this.discretizeState(marketState);
    const allActions = this.getAllActions();

    // Risk-based filtering: reduce options in high-risk conditions
    let filteredActions = allActions;

    // If high drawdown, only allow conservative sizing
    if (marketState.currentDrawdown > 0.15) {
      filteredActions = filteredActions.filter(a => {
        const decoded = this.decodeAction(a);
        return decoded.effectiveSize <= 5; // Max 5% effective in drawdown
      });
    }

    // If low confidence, reduce leverage options
    if (marketState.mlConfidence < 0.5) {
      filteredActions = filteredActions.filter(a => {
        const decoded = this.decodeAction(a);
        return decoded.leverage <= 3;
      });
    }

    // If many open positions, reduce size
    if (marketState.openPositions >= 3) {
      filteredActions = filteredActions.filter(a => {
        const decoded = this.decodeAction(a);
        return decoded.allocationPct <= 2;
      });
    }

    // Ensure we have at least one action
    if (filteredActions.length === 0) {
      filteredActions = [this.encodeAction(1, 1)]; // Minimum size
    }

    let selectedAction: string;

    // Epsilon-greedy: explore with probability epsilon
    if (Math.random() < this.epsilon) {
      // Explore: random action from filtered set
      selectedAction = filteredActions[Math.floor(Math.random() * filteredActions.length)];
    } else {
      // Exploit: best known action
      let bestAction = filteredActions[0];
      let bestValue = this.getQValue(stateKey, bestAction);

      for (const action of filteredActions) {
        const value = this.getQValue(stateKey, action);
        if (value > bestValue) {
          bestValue = value;
          bestAction = action;
        }
      }
      selectedAction = bestAction;
    }

    const decoded = this.decodeAction(selectedAction);

    console.log(`[RL] State: ${stateKey}`);
    console.log(`[RL] Action: ${decoded.allocationPct}% × ${decoded.leverage}x = ${decoded.effectiveSize}% effective`);
    console.log(`[RL] Exploration rate: ${(this.epsilon * 100).toFixed(1)}%`);

    return decoded;
  }

  /**
   * Update Q-values based on trade outcome
   */
  updateFromTrade(
    marketState: MarketState,
    action: PositionAction,
    pnlPercent: number,
    hitStopLoss: boolean,
    hitTakeProfit: boolean
  ): void {
    const stateKey = this.discretizeState(marketState);
    const actionKey = this.encodeAction(action.allocationPct, action.leverage);

    // Calculate reward
    let reward = pnlPercent / 10; // Normalize PnL

    // Bonuses and penalties
    if (pnlPercent > 5) {
      reward *= 1.5; // Bonus for big wins
    } else if (pnlPercent < -3) {
      reward *= 1.5; // Extra penalty for big losses (makes it more negative)
    }

    // Risk management rewards
    if (hitTakeProfit) {
      reward += 0.1; // Reward for taking profits
    }
    if (hitStopLoss && pnlPercent > -2) {
      reward += 0.05; // Small reward for tight stop
    }

    // Leverage efficiency: reward same PnL with lower leverage
    const leverageEfficiency = pnlPercent / (action.leverage || 1);
    if (leverageEfficiency > 1) {
      reward += 0.1; // Efficient use of leverage
    }

    // Q-learning update
    const oldQ = this.getQValue(stateKey, actionKey);
    const maxFutureQ = this.getMaxQValue(stateKey);
    const newQ = oldQ + this.config.learningRate * (
      reward + this.config.discountFactor * maxFutureQ - oldQ
    );

    this.setQValue(stateKey, actionKey, newQ);

    // Record trade
    this.tradeHistory.push({
      state: stateKey,
      action: actionKey,
      reward,
      timestamp: Date.now()
    });

    // Decay exploration
    this.epsilon = Math.max(
      this.config.minExploration,
      this.epsilon * this.config.explorationDecay
    );

    console.log(`[RL] Trade update:`);
    console.log(`     PnL: ${pnlPercent.toFixed(2)}% → Reward: ${reward.toFixed(3)}`);
    console.log(`     Q-value: ${oldQ.toFixed(3)} → ${newQ.toFixed(3)}`);
    console.log(`     Exploration: ${(this.epsilon * 100).toFixed(1)}%`);

    // Save state
    this.saveState();
  }

  private getMaxQValue(state: string): number {
    const stateActions = this.qTable.get(state);
    if (!stateActions || stateActions.size === 0) return 0;
    return Math.max(...Array.from(stateActions.values()));
  }

  /**
   * Get recommended position size without learning (for paper trading)
   */
  getRecommendation(marketState: MarketState): PositionAction & { reasoning: string[] } {
    const action = this.selectAction(marketState);
    const stateKey = this.discretizeState(marketState);
    const actionKey = this.encodeAction(action.allocationPct, action.leverage);
    const qValue = this.getQValue(stateKey, actionKey);

    const reasoning: string[] = [];

    // Explain the decision
    if (marketState.currentDrawdown > 0.1) {
      reasoning.push(`High drawdown (${(marketState.currentDrawdown * 100).toFixed(1)}%) → Conservative sizing`);
    }
    if (marketState.mlConfidence > 0.7) {
      reasoning.push(`High ML confidence (${(marketState.mlConfidence * 100).toFixed(0)}%) → Larger position allowed`);
    }
    if (marketState.volatility > 0.04) {
      reasoning.push(`High volatility (${(marketState.volatility * 100).toFixed(1)}%) → Reduced leverage`);
    }
    if (marketState.recentWinRate > 0.6) {
      reasoning.push(`Good win rate (${(marketState.recentWinRate * 100).toFixed(0)}%) → Increased confidence`);
    }
    if (marketState.openPositions >= 3) {
      reasoning.push(`Multiple positions open (${marketState.openPositions}) → Smaller allocation`);
    }

    reasoning.push(`Q-value for this state-action: ${qValue.toFixed(3)}`);
    reasoning.push(`Based on ${this.tradeHistory.length} learned trades`);

    return { ...action, reasoning };
  }

  /**
   * Get statistics about learned behavior
   */
  getStats(): {
    totalStates: number;
    totalTrades: number;
    avgReward: number;
    explorationRate: number;
    topActions: Array<{ state: string; action: string; qValue: number }>;
  } {
    const totalStates = this.qTable.size;
    const totalTrades = this.tradeHistory.length;
    const avgReward = totalTrades > 0
      ? this.tradeHistory.reduce((sum, t) => sum + t.reward, 0) / totalTrades
      : 0;

    // Find top state-action pairs
    const topActions: Array<{ state: string; action: string; qValue: number }> = [];
    for (const [state, actions] of this.qTable.entries()) {
      for (const [action, qValue] of actions.entries()) {
        topActions.push({ state, action, qValue });
      }
    }
    topActions.sort((a, b) => b.qValue - a.qValue);

    return {
      totalStates,
      totalTrades,
      avgReward,
      explorationRate: this.epsilon,
      topActions: topActions.slice(0, 10)
    };
  }

  /**
   * Save Q-table and state to disk
   */
  private saveState(): void {
    const dir = path.dirname(this.statePath);
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }

    const state = {
      qTable: Object.fromEntries(
        Array.from(this.qTable.entries()).map(([k, v]) => [k, Object.fromEntries(v)])
      ),
      epsilon: this.epsilon,
      tradeHistory: this.tradeHistory.slice(-1000), // Keep last 1000 trades
      savedAt: new Date().toISOString()
    };

    fs.writeFileSync(this.statePath, JSON.stringify(state, null, 2));
  }

  /**
   * Load Q-table and state from disk
   */
  private loadState(): void {
    if (!fs.existsSync(this.statePath)) {
      console.log('[RL] No saved state found, starting fresh');
      return;
    }

    try {
      const data = JSON.parse(fs.readFileSync(this.statePath, 'utf-8'));

      // Restore Q-table
      this.qTable = new Map(
        Object.entries(data.qTable).map(([k, v]) => [k, new Map(Object.entries(v as object))])
      );

      this.epsilon = data.epsilon || this.config.explorationRate;
      this.tradeHistory = data.tradeHistory || [];

      console.log(`[RL] Loaded state: ${this.qTable.size} states, ${this.tradeHistory.length} trades`);
      console.log(`[RL] Exploration rate: ${(this.epsilon * 100).toFixed(1)}%`);
    } catch (e) {
      console.error('[RL] Error loading state:', e);
    }
  }

  /**
   * Reset learning (for testing)
   */
  reset(): void {
    this.qTable = new Map();
    this.epsilon = this.config.explorationRate;
    this.tradeHistory = [];
    this.saveState();
    console.log('[RL] State reset');
  }
}

// Export singleton instance
export const positionSizer = new RLPositionSizer();
