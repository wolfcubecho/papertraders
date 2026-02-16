#!/usr/bin/env python3
"""
CNN-LSTM Trade Predictor (PyTorch)

Deep learning model for trade outcome prediction:
- 1D CNN to find patterns in feature relationships
- LSTM for temporal dependencies across trades
- Walk-forward validation (same as LightGBM)
- PnL-based evaluation

Usage:
    python scripts/cnn_lstm_trainer.py --input data/learning-loop/training_data_*.csv
"""

import argparse
import json
import os
import sys
import glob
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd

# Check for PyTorch
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    from sklearn.preprocessing import StandardScaler, LabelEncoder
    from sklearn.metrics import accuracy_score, roc_auc_score
    print(f"PyTorch version: {torch.__version__}")
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")
except ImportError:
    print("ERROR: PyTorch not installed.")
    print("Run: pip install torch scikit-learn pandas numpy")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

CONFIG = {
    # Model architecture
    'sequence_length': 10,       # Use last N trades as context
    'cnn_filters': [64, 128],    # CNN filter sizes
    'cnn_kernel_size': 3,        # CNN kernel size
    'lstm_units': 64,            # LSTM hidden units
    'dense_units': [64, 32],     # Dense layer sizes
    'dropout_rate': 0.3,         # Dropout for regularization

    # Training
    'epochs': 100,
    'batch_size': 64,
    'learning_rate': 0.001,
    'early_stopping_patience': 15,

    # Walk-forward
    'train_pct': 0.70,
    'val_pct': 0.15,
    'test_pct': 0.15,
    'min_samples': 1000,

    # Threshold tuning
    'threshold_search_min': 0.15,  # Lower to find trades model likes
    'threshold_search_max': 0.65,
    'threshold_search_step': 0.02,
    'min_filtered_trades': 100,  # Need enough trades for statistical significance
}


# ═══════════════════════════════════════════════════════════════
# DATA LOADING AND PREPARATION
# ═══════════════════════════════════════════════════════════════

def load_training_data(input_path: str) -> pd.DataFrame:
    """Load training data from CSV, handling glob patterns"""
    if '*' in input_path:
        files = sorted(glob.glob(input_path))
        if not files:
            raise FileNotFoundError(f"No files match pattern: {input_path}")
        files_with_size = [(f, os.path.getsize(f)) for f in files]
        input_path = max(files_with_size, key=lambda x: x[1])[0]
        print(f"Using largest file: {input_path}")

    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")

    required = ['entry_time', 'outcome', 'pnl']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    return df


def prepare_features(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], object, Dict]:
    """Prepare features for CNN-LSTM"""

    df = df.sort_values('entry_time').reset_index(drop=True)

    y = (df['outcome'] == 'WIN').astype(int).values
    pnl = df['pnl'].values

    # Categorical columns (string values)
    categorical_cols = ['trend_direction', 'ob_type', 'fvg_type', 'ema_trend',
                       'rsi_state', 'direction', 'session', 'pullback_fib',
                       'smart_money_direction', 'ob_state', 'regime',
                       'kill_zone',    # 'LONDON', 'NY_OPEN', etc.
                       'ema_aligned']  # 'bullish', 'bearish', 'neutral' - NOT boolean!

    # Boolean columns (True/False only)
    bool_cols = ['ob_near', 'fvg_near', 'liquidity_near',
                'mtf_aligned', 'volume_spike', 'trend_bos_aligned', 'is_pullback',
                'ob_is_fresh', 'has_ob_return', 'has_fvg_fill',
                'ob_confirmed_mitigated', 'ob_in_mitigation', 'ob_caused_bos',
                'ob_has_displacement', 'bb_squeeze', 'bb_breakout_upper',
                'bb_breakout_lower', 'large_volume_spike', 'accumulation_detected',
                'distribution_detected', 'macd_bullish_cross', 'macd_bearish_cross',
                'price_above_vwap', 'ema_bullish_cross', 'ema_bearish_cross',
                'is_kill_zone']  # True/False boolean

    exclude_cols = ['entry_time', 'entry_price', 'outcome', 'pnl', 'pnl_percent',
                   'exit_reason', 'holding_periods', 'symbol', 'exit_time', 'exit_price']

    all_cols = set(df.columns)
    used_cols = set(categorical_cols) | set(bool_cols) | set(exclude_cols)
    numeric_cols = [c for c in df.columns if c not in used_cols and df[c].dtype in ['int64', 'float64']]

    available_categorical = [c for c in categorical_cols if c in df.columns]
    available_numeric = [c for c in numeric_cols if c in df.columns]
    available_bool = [c for c in bool_cols if c in df.columns]

    # Check for string values in bool columns and move to categorical
    actual_bool_cols = []
    for col in available_bool:
        sample_vals = df[col].dropna().head(100)
        if sample_vals.dtype == object:
            unique_vals = set(str(v).lower() for v in sample_vals.unique())
            if unique_vals - {'true', 'false', '1', '0', '1.0', '0.0'}:
                print(f"  Warning: '{col}' has string values {unique_vals}, treating as categorical")
                available_categorical.append(col)
                continue
        actual_bool_cols.append(col)
    available_bool = actual_bool_cols

    print(f"Features: {len(available_numeric)} numeric, {len(available_categorical)} categorical, {len(available_bool)} boolean")

    feature_list = []
    feature_names = []
    encoders = {}

    if available_numeric:
        numeric_data = df[available_numeric].fillna(0).values
        feature_list.append(numeric_data)
        feature_names.extend(available_numeric)

    if available_bool:
        bool_data = df[available_bool].fillna(False).astype(bool).astype(float).values
        feature_list.append(bool_data)
        feature_names.extend(available_bool)

    if available_categorical:
        for col in available_categorical:
            le = LabelEncoder()
            encoded = le.fit_transform(df[col].fillna('unknown').astype(str))
            feature_list.append(encoded.reshape(-1, 1))
            feature_names.append(col)
            encoders[col] = le

    X = np.hstack(feature_list)

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, pnl, feature_names, scaler, encoders


def create_sequences(X: np.ndarray, y: np.ndarray, pnl: np.ndarray,
                     seq_length: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Create sequences for CNN-LSTM"""

    sequences = []
    labels = []
    pnls = []

    for i in range(seq_length, len(X)):
        seq = X[i-seq_length:i]
        sequences.append(seq)
        labels.append(y[i])
        pnls.append(pnl[i])

    return np.array(sequences), np.array(labels), np.array(pnls)


# ═══════════════════════════════════════════════════════════════
# CNN-LSTM MODEL (PyTorch)
# ═══════════════════════════════════════════════════════════════

class CNNLSTM(nn.Module):
    """CNN-LSTM model for trade prediction"""

    def __init__(self, input_size: int, seq_length: int, config: Dict):
        super().__init__()

        self.seq_length = seq_length
        self.input_size = input_size

        # CNN layers
        cnn_layers = []
        in_channels = input_size
        for filters in config['cnn_filters']:
            cnn_layers.extend([
                nn.Conv1d(in_channels, filters, kernel_size=config['cnn_kernel_size'], padding=1),
                nn.BatchNorm1d(filters),
                nn.ReLU(),
                nn.Dropout(config['dropout_rate']),
            ])
            in_channels = filters
        self.cnn = nn.Sequential(*cnn_layers)

        # LSTM layer
        self.lstm = nn.LSTM(
            input_size=config['cnn_filters'][-1],
            hidden_size=config['lstm_units'],
            batch_first=True,
            dropout=config['dropout_rate'] if len(config['dense_units']) > 1 else 0
        )

        # Dense layers
        dense_layers = []
        in_features = config['lstm_units']
        for units in config['dense_units']:
            dense_layers.extend([
                nn.Linear(in_features, units),
                nn.BatchNorm1d(units),
                nn.ReLU(),
                nn.Dropout(config['dropout_rate']),
            ])
            in_features = units
        dense_layers.append(nn.Linear(in_features, 1))
        dense_layers.append(nn.Sigmoid())
        self.dense = nn.Sequential(*dense_layers)

    def forward(self, x):
        # x shape: (batch, seq_length, features)
        # Conv1d expects: (batch, channels, length)
        x = x.permute(0, 2, 1)
        x = self.cnn(x)
        x = x.permute(0, 2, 1)

        # LSTM
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]  # Take last timestep

        # Dense
        x = self.dense(x)
        return x.squeeze(-1)


# ═══════════════════════════════════════════════════════════════
# PNL METRICS
# ═══════════════════════════════════════════════════════════════

def compute_pnl_metrics(y_pred_proba: np.ndarray, pnl_values: np.ndarray,
                        threshold: float) -> Dict:
    """Compute PnL metrics for filtered trades"""

    approved_mask = y_pred_proba >= threshold

    baseline_trades = len(pnl_values)
    baseline_pnl = float(pnl_values.sum())
    baseline_win_rate = float((pnl_values > 0).mean()) if baseline_trades > 0 else 0.0
    baseline_avg_pnl = baseline_pnl / baseline_trades if baseline_trades > 0 else 0.0

    filtered_pnl_values = pnl_values[approved_mask]
    filtered_trades = len(filtered_pnl_values)
    filtered_pnl = float(filtered_pnl_values.sum()) if filtered_trades > 0 else 0.0
    filtered_win_rate = float((filtered_pnl_values > 0).mean()) if filtered_trades > 0 else 0.0
    filtered_avg_pnl = filtered_pnl / filtered_trades if filtered_trades > 0 else 0.0

    return {
        'baseline_trades': baseline_trades,
        'baseline_pnl': baseline_pnl,
        'baseline_win_rate': baseline_win_rate,
        'baseline_avg_pnl': baseline_avg_pnl,
        'filtered_trades': filtered_trades,
        'filtered_pnl': filtered_pnl,
        'filtered_win_rate': filtered_win_rate,
        'filtered_avg_pnl': filtered_avg_pnl,
        'pnl_improvement': filtered_pnl - baseline_pnl,
        'win_rate_improvement': filtered_win_rate - baseline_win_rate,
    }


def tune_threshold(y_pred_proba: np.ndarray, pnl_values: np.ndarray,
                   config: Dict) -> Tuple[float, Dict]:
    """Find optimal probability threshold"""

    thresholds = np.arange(
        config['threshold_search_min'],
        config['threshold_search_max'] + 0.001,
        config['threshold_search_step']
    )

    best_threshold = 0.5
    best_score = -float('inf')
    best_metrics = None

    for thresh in thresholds:
        metrics = compute_pnl_metrics(y_pred_proba, pnl_values, thresh)

        if metrics['filtered_trades'] < config['min_filtered_trades']:
            continue

        trade_retention = metrics['filtered_trades'] / metrics['baseline_trades']
        score = (
            metrics['filtered_avg_pnl'] * 0.6 +
            metrics['win_rate_improvement'] * 100 * 0.25 +
            trade_retention * 5 * 0.15
        )

        if score > best_score:
            best_score = score
            best_threshold = thresh
            best_metrics = metrics

    if best_metrics is None:
        best_threshold = 0.5
        best_metrics = compute_pnl_metrics(y_pred_proba, pnl_values, 0.5)

    return best_threshold, best_metrics


# ═══════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════

def train_cnn_lstm(df: pd.DataFrame, config: Dict) -> Dict:
    """Train CNN-LSTM with time-based split"""

    print("\n" + "="*65)
    print("CNN-LSTM TRAINING (PyTorch)")
    print("="*65)

    # Prepare features
    X, y, pnl, feature_names, scaler, encoders = prepare_features(df)
    print(f"\nPrepared {len(X)} samples with {X.shape[1]} features")

    # Create sequences
    seq_length = config['sequence_length']
    X_seq, y_seq, pnl_seq = create_sequences(X, y, pnl, seq_length)
    print(f"Created {len(X_seq)} sequences of length {seq_length}")

    # Time-based split
    n = len(X_seq)
    train_end = int(n * config['train_pct'])
    val_end = int(n * (config['train_pct'] + config['val_pct']))

    X_train, y_train, pnl_train = X_seq[:train_end], y_seq[:train_end], pnl_seq[:train_end]
    X_val, y_val, pnl_val = X_seq[train_end:val_end], y_seq[train_end:val_end], pnl_seq[train_end:val_end]
    X_test, y_test, pnl_test = X_seq[val_end:], y_seq[val_end:], pnl_seq[val_end:]

    print(f"\nTrain: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    print(f"Train WIN rate: {y_train.mean():.1%}")
    print(f"Test WIN rate: {y_test.mean():.1%}")

    # Class weight for imbalance
    pos_weight = (y_train == 0).sum() / max(1, (y_train == 1).sum())
    print(f"Positive class weight: {pos_weight:.3f}")

    # Convert to tensors
    X_train_t = torch.FloatTensor(X_train).to(DEVICE)
    y_train_t = torch.FloatTensor(y_train).to(DEVICE)
    X_val_t = torch.FloatTensor(X_val).to(DEVICE)
    y_val_t = torch.FloatTensor(y_val).to(DEVICE)
    X_test_t = torch.FloatTensor(X_test).to(DEVICE)

    train_dataset = TensorDataset(X_train_t, y_train_t)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)

    # Build model
    model = CNNLSTM(X.shape[1], seq_length, config).to(DEVICE)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Loss and optimizer
    criterion = nn.BCELoss(weight=None)  # We'll handle class weight differently
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    # Training loop
    print("\nTraining...")
    best_val_auc = 0
    patience_counter = 0
    best_model_state = None

    for epoch in range(config['epochs']):
        model.train()
        train_loss = 0

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            outputs = model(batch_X)

            # Apply class weight manually
            weights = torch.where(batch_y == 1, pos_weight, 1.0)
            loss = (criterion(outputs, batch_y) * weights).mean()

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val_t).cpu().numpy()
            val_pred = (val_outputs >= 0.5).astype(int)
            val_acc = accuracy_score(y_val, val_pred)
            try:
                val_auc = roc_auc_score(y_val, val_outputs)
            except ValueError:
                val_auc = 0.5

        scheduler.step(1 - val_auc)

        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {train_loss/len(train_loader):.4f} | "
                  f"Val Acc: {val_acc:.3f} | Val AUC: {val_auc:.3f}")

        # Early stopping
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_model_state = model.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch+1}")
                break

    # Load best model
    if best_model_state:
        model.load_state_dict(best_model_state)

    # Predict on test
    print("\nEvaluating on test set...")
    model.eval()
    with torch.no_grad():
        y_train_proba = model(X_train_t).cpu().numpy()
        y_test_proba = model(X_test_t).cpu().numpy()

    # Tune threshold
    optimal_threshold, train_metrics = tune_threshold(y_train_proba, pnl_train, config)
    print(f"Optimal threshold (from train): {optimal_threshold:.2f}")

    # Test metrics
    y_test_pred = (y_test_proba >= optimal_threshold).astype(int)
    test_metrics = compute_pnl_metrics(y_test_proba, pnl_test, optimal_threshold)

    accuracy = accuracy_score(y_test, y_test_pred)
    try:
        auc_roc = roc_auc_score(y_test, y_test_proba)
    except ValueError:
        auc_roc = 0.5

    # Print results
    print(f"\n{'='*65}")
    print("RESULTS")
    print(f"{'='*65}")
    print(f"Accuracy: {accuracy:.1%}, AUC: {auc_roc:.3f}")
    print(f"Baseline:  {test_metrics['baseline_trades']} trades, "
          f"PnL: ${test_metrics['baseline_pnl']:,.0f}, "
          f"Win: {test_metrics['baseline_win_rate']:.1%}")
    print(f"Filtered:  {test_metrics['filtered_trades']} trades, "
          f"PnL: ${test_metrics['filtered_pnl']:,.0f}, "
          f"Win: {test_metrics['filtered_win_rate']:.1%}")
    print(f"Improvement: ${test_metrics['pnl_improvement']:,.0f}, "
          f"Win rate: {test_metrics['win_rate_improvement']:+.1%}")

    return {
        'model': model,
        'scaler': scaler,
        'encoders': encoders,
        'feature_names': feature_names,
        'optimal_threshold': optimal_threshold,
        'accuracy': accuracy,
        'auc': auc_roc,
        'test_metrics': test_metrics,
        'config': config,
    }


# ═══════════════════════════════════════════════════════════════
# SAVE MODEL
# ═══════════════════════════════════════════════════════════════

def save_model(results: Dict, output_dir: str):
    """Save CNN-LSTM model and metadata"""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    # Save PyTorch model
    model_path = os.path.join(output_dir, 'cnn_lstm_model.pt')
    torch.save({
        'model_state_dict': results['model'].state_dict(),
        'config': results['config'],
        'input_size': len(results['feature_names']),
    }, model_path)

    # Save scaler
    scaler_path = os.path.join(output_dir, 'cnn_lstm_scaler.npy')
    np.save(scaler_path, {
        'mean': results['scaler'].mean_,
        'scale': results['scaler'].scale_,
    }, allow_pickle=True)

    # Save metadata
    metadata = {
        'modelId': f'cnn_lstm_{timestamp}',
        'trainedAt': datetime.now().isoformat(),
        'algorithm': 'cnn-lstm-pytorch',
        'optimal_threshold': float(results['optimal_threshold']),
        'accuracy': float(results['accuracy']),
        'auc': float(results['auc']),
        'test_metrics': {k: float(v) if isinstance(v, (np.floating, float)) else v
                        for k, v in results['test_metrics'].items()},
        'config': {
            'sequence_length': results['config']['sequence_length'],
            'cnn_filters': results['config']['cnn_filters'],
            'lstm_units': results['config']['lstm_units'],
        },
        'feature_names': results['feature_names'],
        'num_features': len(results['feature_names']),
    }

    metadata_path = os.path.join(output_dir, 'cnn_lstm_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved: {model_path}")
    print(f"Saved: {metadata_path}")


# ═══════════════════════════════════════════════════════════════
# COMPARE WITH LIGHTGBM
# ═══════════════════════════════════════════════════════════════

def compare_with_lightgbm(cnn_results: Dict, lightgbm_metadata_path: str):
    """Compare CNN-LSTM results with LightGBM"""

    print(f"\n{'='*65}")
    print("COMPARISON: CNN-LSTM vs LightGBM")
    print(f"{'='*65}")

    if not os.path.exists(lightgbm_metadata_path):
        print("LightGBM metadata not found, skipping comparison")
        return

    with open(lightgbm_metadata_path) as f:
        lgbm_meta = json.load(f)

    lgbm_wf = lgbm_meta.get('walk_forward', {})

    print(f"\n{'Metric':<25} {'LightGBM':>15} {'CNN-LSTM':>15} {'Winner':>10}")
    print("-"*65)

    # Accuracy
    lgbm_acc = lgbm_meta.get('accuracy', 0)
    cnn_acc = cnn_results['accuracy']
    winner = 'CNN-LSTM' if cnn_acc > lgbm_acc else 'LightGBM'
    print(f"{'Accuracy':<25} {lgbm_acc:>14.1%} {cnn_acc:>14.1%} {winner:>10}")

    # AUC
    lgbm_auc = lgbm_meta.get('auc', 0)
    cnn_auc = cnn_results['auc']
    winner = 'CNN-LSTM' if cnn_auc > lgbm_auc else 'LightGBM'
    print(f"{'AUC':<25} {lgbm_auc:>15.3f} {cnn_auc:>15.3f} {winner:>10}")

    # PnL Improvement
    lgbm_pnl = lgbm_wf.get('pnl_improvement', lgbm_wf.get('total_filtered_pnl', 0) - lgbm_wf.get('total_baseline_pnl', 0))
    cnn_pnl = cnn_results['test_metrics']['pnl_improvement']
    winner = 'CNN-LSTM' if cnn_pnl > lgbm_pnl else 'LightGBM'
    print(f"{'PnL Improvement':<25} ${lgbm_pnl:>14,.0f} ${cnn_pnl:>14,.0f} {winner:>10}")

    # Win Rate
    lgbm_wr = lgbm_wf.get('avg_filtered_win_rate', 0)
    cnn_wr = cnn_results['test_metrics']['filtered_win_rate']
    winner = 'CNN-LSTM' if cnn_wr > lgbm_wr else 'LightGBM'
    print(f"{'Filtered Win Rate':<25} {lgbm_wr:>14.1%} {cnn_wr:>14.1%} {winner:>10}")

    # Threshold
    lgbm_thresh = lgbm_wf.get('optimal_threshold', 0.5)
    cnn_thresh = cnn_results['optimal_threshold']
    print(f"{'Optimal Threshold':<25} {lgbm_thresh:>15.2f} {cnn_thresh:>15.2f}")


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='Train CNN-LSTM for trade prediction')
    parser.add_argument('--input', '-i', required=True,
                        help='Path to training CSV (supports glob patterns)')
    parser.add_argument('--output', '-o', default='data/models',
                        help='Output directory for model')
    parser.add_argument('--sequence-length', type=int, default=10,
                        help='Number of trades to use as context')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Maximum training epochs')
    args = parser.parse_args()

    config = CONFIG.copy()
    config['sequence_length'] = args.sequence_length
    config['epochs'] = args.epochs

    print("="*65)
    print("CNN-LSTM TRADE PREDICTOR (PyTorch)")
    print("="*65)

    df = load_training_data(args.input)

    if len(df) < config['min_samples']:
        print(f"ERROR: Need at least {config['min_samples']} samples, got {len(df)}")
        sys.exit(1)

    results = train_cnn_lstm(df, config)

    save_model(results, args.output)

    lgbm_path = os.path.join(args.output, 'lightgbm_metadata.json')
    compare_with_lightgbm(results, lgbm_path)

    print("\nDone.")


if __name__ == '__main__':
    main()
