#!/usr/bin/env python3
"""
LightGBM Training Script for Trading ML Model

Advantages over gradient descent logistic regression:
- Handles non-linear patterns
- Built-in feature importance
- Native categorical feature handling
- Fast training with early stopping
- Better generalization with regularization

Usage:
    python scripts/lightgbm_trainer.py --input data/learning-loop/training_data_*.csv
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install lightgbm scikit-learn pandas numpy")
    sys.exit(1)


def load_training_data(input_path: str) -> pd.DataFrame:
    """Load training data from CSV"""
    print(f"Loading data from: {input_path}")
    df = pd.read_csv(input_path)
    print(f"Loaded {len(df)} rows, {len(df.columns)} columns")
    return df


def prepare_features(df: pd.DataFrame) -> tuple:
    """Prepare features and target for training"""

    # Target: outcome (WIN=1, LOSS=0)
    df['target'] = (df['outcome'] == 'WIN').astype(int)

    # Categorical features to encode
    categorical_cols = [
        'trend_direction', 'ob_type', 'fvg_type', 'ema_trend',
        'rsi_state', 'direction', 'session', 'pullback_fib',
        'smart_money_direction',  # Institutional flow direction
        'ob_state'  # OB State Machine (CRITICAL for SMC)
    ]

    # Numeric features
    numeric_cols = [
        'trend_strength', 'ob_distance', 'ob_size', 'ob_age',
        'fvg_nearest_distance', 'fvg_size', 'fvg_count',
        'volatility', 'rsi_value', 'atr_value',
        'price_position', 'distance_to_high', 'distance_to_low',
        'volume_ratio', 'confluence_score', 'potential_rr',
        'pullback_depth', 'pullback_bars',
        # OB State Machine metrics (CRITICAL for SMC)
        'ob_test_count', 'ob_return_age', 'smc_signal_strength',
        'ob_impulse_size',  # ATR multiples of move after OB
        # Bollinger Bands
        'bb_position', 'bb_width',
        # Institutional flow
        'volume_delta'
    ]

    # Boolean features
    bool_cols = [
        'ob_near', 'fvg_near', 'ema_aligned', 'liquidity_near',
        'mtf_aligned', 'volume_spike', 'trend_bos_aligned', 'is_pullback',
        # OB State Machine booleans (CRITICAL for SMC)
        'ob_is_fresh', 'has_ob_return', 'has_fvg_fill',
        'ob_confirmed_mitigated',  # BEST: Rejection + BOS confirmed
        'ob_in_mitigation',        # Currently testing zone
        'ob_caused_bos',           # OB caused break of structure
        'ob_has_displacement',     # Displacement after OB
        # Bollinger Bands
        'bb_squeeze', 'bb_breakout_upper', 'bb_breakout_lower',
        # Institutional flow
        'large_volume_spike', 'accumulation_detected', 'distribution_detected'
    ]

    # Filter to columns that exist
    available_categorical = [c for c in categorical_cols if c in df.columns]
    available_numeric = [c for c in numeric_cols if c in df.columns]
    available_bool = [c for c in bool_cols if c in df.columns]

    print(f"\nFeatures found:")
    print(f"  Categorical: {len(available_categorical)}")
    print(f"  Numeric: {len(available_numeric)}")
    print(f"  Boolean: {len(available_bool)}")

    # Build feature matrix
    feature_dfs = []

    # Numeric features (as-is)
    if available_numeric:
        feature_dfs.append(df[available_numeric].fillna(0))

    # Boolean features (convert to int)
    if available_bool:
        bool_df = df[available_bool].fillna(False).astype(int)
        feature_dfs.append(bool_df)

    # Categorical features (label encode for LightGBM)
    cat_feature_names = []
    if available_categorical:
        for col in available_categorical:
            df[col] = df[col].fillna('unknown').astype('category')
            cat_feature_names.append(col)
        feature_dfs.append(df[available_categorical])

    X = pd.concat(feature_dfs, axis=1)
    y = df['target']

    # Get categorical feature indices for LightGBM
    cat_indices = [X.columns.get_loc(c) for c in cat_feature_names if c in X.columns]

    print(f"\nTotal features: {len(X.columns)}")
    print(f"Categorical indices: {cat_indices}")

    return X, y, cat_indices, list(X.columns)


def train_lightgbm(X: pd.DataFrame, y: pd.Series, cat_indices: list) -> tuple:
    """Train LightGBM model"""

    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    print(f"\nTraining set: {len(X_train)}")
    print(f"Validation set: {len(X_val)}")
    print(f"Win rate (train): {y_train.mean():.1%}")
    print(f"Win rate (val): {y_val.mean():.1%}")

    # Small-data guardrails
    # With tiny datasets (like ~50 rows), default params like min_data_in_leaf=20
    # can prevent *any* split, yielding constant predictions (AUC=0.5) and 0 importances.
    n_train = len(X_train)
    n_total = len(X)

    # Heuristic: keep leaves small enough to allow splits, but not 1.
    auto_min_leaf = max(2, min(20, n_train // 8))  # e.g., 38 rows -> 4
    # Make model simpler when data is tiny to reduce overfit instability.
    small_data_mode = n_total < 500

    # LightGBM parameters
    params = {
        'objective': 'binary',
        'metric': ['binary_logloss', 'auc'],
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.8,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_data_in_leaf': auto_min_leaf,
        'lambda_l1': 0.1,
        'lambda_l2': 0.1,
        'max_depth': 6,
        'verbose': -1,
        'seed': 42
    }

    if small_data_mode:
        # Reduce randomness and constraints so we can learn *something*.
        params.update({
            'num_leaves': 15,
            'max_depth': 4,
            'learning_rate': 0.1,
            'feature_fraction': 1.0,
            'bagging_fraction': 1.0,
            'bagging_freq': 0,
            'min_data_in_leaf': max(2, min(10, auto_min_leaf)),
            'min_sum_hessian_in_leaf': 1e-3,
            'min_gain_to_split': 0.0,
            'max_bin': 63,
        })

    print(f"\nLightGBM params:")
    print(f"  small_data_mode: {small_data_mode} (rows={n_total}, train={n_train})")
    print(f"  min_data_in_leaf: {params['min_data_in_leaf']}")
    print(f"  num_leaves: {params['num_leaves']}, max_depth: {params['max_depth']}, lr: {params['learning_rate']}")

    # Create datasets
    train_data = lgb.Dataset(
        X_train, label=y_train,
        categorical_feature=cat_indices if cat_indices else 'auto'
    )
    val_data = lgb.Dataset(
        X_val, label=y_val,
        reference=train_data,
        categorical_feature=cat_indices if cat_indices else 'auto'
    )

    # Train with early stopping
    print("\nTraining LightGBM...")
    callbacks = [
        lgb.early_stopping(stopping_rounds=20),
        lgb.log_evaluation(period=50)
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'valid'],
        callbacks=callbacks
    )

    # Evaluate
    y_pred_proba = model.predict(X_val, num_iteration=model.best_iteration)
    y_pred = (y_pred_proba >= 0.5).astype(int)

    accuracy = accuracy_score(y_val, y_pred)
    auc = roc_auc_score(y_val, y_pred_proba)

    print(f"\n{'='*50}")
    print(f"RESULTS (Best Iteration: {model.best_iteration})")
    print(f"{'='*50}")
    print(f"Accuracy: {accuracy:.1%}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"\nClassification Report:")
    print(classification_report(y_val, y_pred, target_names=['LOSS', 'WIN']))

    return model, accuracy, auc


def get_feature_importance(model, feature_names: list) -> dict:
    """Get feature importance from trained model"""

    importance = model.feature_importance(importance_type='gain')
    importance_dict = dict(zip(feature_names, importance))

    # Sort by importance
    sorted_importance = dict(sorted(
        importance_dict.items(),
        key=lambda x: x[1],
        reverse=True
    ))

    print(f"\n{'='*50}")
    print("TOP 15 FEATURES BY IMPORTANCE")
    print(f"{'='*50}")
    for i, (feat, imp) in enumerate(list(sorted_importance.items())[:15]):
        print(f"  {i+1:2d}. {feat:30s} {imp:,.0f}")

    return sorted_importance


def save_model(model, output_dir: str, feature_names: list, accuracy: float, auc: float):
    """Save model and metadata"""

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')

    # Save LightGBM model (native format)
    model_path = os.path.join(output_dir, 'lightgbm_model.txt')
    model.save_model(model_path)
    print(f"\nSaved model: {model_path}")

    # Save as JSON for JavaScript loading
    # LightGBM models can be dumped to JSON
    model_json_path = os.path.join(output_dir, 'lightgbm_model.json')
    model_json = model.dump_model()
    with open(model_json_path, 'w') as f:
        json.dump(model_json, f)
    print(f"Saved model JSON: {model_json_path}")

    # Save metadata
    metadata = {
        'modelId': f'lightgbm_{timestamp}',
        'trainedAt': datetime.now().isoformat(),
        'algorithm': 'lightgbm-gbdt',
        'accuracy': accuracy,
        'auc': auc,
        'bestIteration': model.best_iteration,
        'numFeatures': len(feature_names),
        'features': feature_names,
        'featureImportance': get_feature_importance(model, feature_names)
    }

    metadata_path = os.path.join(output_dir, 'lightgbm_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Train LightGBM model for trading')
    parser.add_argument('--input', '-i', required=True, help='Path to training CSV')
    parser.add_argument('--output', '-o', default='data/models', help='Output directory')
    args = parser.parse_args()

    # Load data
    df = load_training_data(args.input)

    # Prepare features
    X, y, cat_indices, feature_names = prepare_features(df)

    # Train model
    model, accuracy, auc = train_lightgbm(X, y, cat_indices)

    # Save model
    save_model(model, args.output, feature_names, accuracy, auc)

    print(f"\n{'='*50}")
    print("TRAINING COMPLETE")
    print(f"{'='*50}")
    print(f"Model accuracy: {accuracy:.1%}")
    print(f"Model AUC: {auc:.4f}")
    print(f"Output: {args.output}")


if __name__ == '__main__':
    main()
