#!/usr/bin/env python3
"""LightGBM Evaluation Script

Evaluates the current LightGBM model on a labeled CSV (must include `outcome` column).

Usage:
  python scripts/lightgbm_evaluate.py --input data/h2o-training/historical_15m_*.csv
  python scripts/lightgbm_evaluate.py --input ... --model data/models/lightgbm_model.txt
"""

import argparse
import json
import os
from datetime import datetime

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
except ImportError:
    print("ERROR: Required packages not installed.")
    print("Run: pip install lightgbm scikit-learn pandas numpy")
    raise


def load_model(model_path: str) -> lgb.Booster:
    return lgb.Booster(model_file=model_path)


def prepare_features_like_trainer(df: pd.DataFrame):
    """Match the feature prep used by lightgbm_trainer.py."""

    if 'outcome' not in df.columns:
        raise ValueError("Input CSV must include 'outcome' column")

    # Target: outcome (WIN=1, LOSS=0)
    y = (df['outcome'] == 'WIN').astype(int)

    categorical_cols = [
        'trend_direction', 'ob_type', 'fvg_type', 'ema_trend',
        'rsi_state', 'direction', 'session', 'pullback_fib',
        'smart_money_direction',
        'ob_state'
    ]

    numeric_cols = [
        'trend_strength', 'ob_distance', 'ob_size', 'ob_age',
        'fvg_nearest_distance', 'fvg_size', 'fvg_count',
        'volatility', 'rsi_value', 'atr_value',
        'price_position', 'distance_to_high', 'distance_to_low',
        'volume_ratio', 'confluence_score', 'potential_rr',
        'pullback_depth', 'pullback_bars',
        'ob_test_count', 'ob_return_age', 'smc_signal_strength',
        'ob_impulse_size',
        'bb_position', 'bb_width',
        'volume_delta'
    ]

    bool_cols = [
        'ob_near', 'fvg_near', 'ema_aligned', 'liquidity_near',
        'mtf_aligned', 'volume_spike', 'trend_bos_aligned', 'is_pullback',
        'ob_is_fresh', 'has_ob_return', 'has_fvg_fill',
        'ob_confirmed_mitigated',
        'ob_in_mitigation',
        'ob_caused_bos',
        'ob_has_displacement',
        'bb_squeeze', 'bb_breakout_upper', 'bb_breakout_lower',
        'large_volume_spike', 'accumulation_detected', 'distribution_detected'
    ]

    available_categorical = [c for c in categorical_cols if c in df.columns]
    available_numeric = [c for c in numeric_cols if c in df.columns]
    available_bool = [c for c in bool_cols if c in df.columns]

    feature_dfs = []

    if available_numeric:
        feature_dfs.append(df[available_numeric].fillna(0))

    if available_bool:
        bool_df = df[available_bool].fillna(False).astype(int)
        feature_dfs.append(bool_df)

    if available_categorical:
        cat_df = df[available_categorical].copy()
        for col in available_categorical:
            cat_df[col] = cat_df[col].fillna('unknown').astype('category')
        feature_dfs.append(cat_df)

    if not feature_dfs:
        raise ValueError("No usable feature columns found")

    X = pd.concat(feature_dfs, axis=1)
    return X, y


def align_to_model_features(X: pd.DataFrame, model: lgb.Booster) -> pd.DataFrame:
    """Ensure X columns match model.feature_name() order."""
    model_features = model.feature_name()

    # Add missing columns
    for col in model_features:
        if col not in X.columns:
            # If model expects categorical, we don't know; fall back to 0.
            X[col] = 0

    # Drop extra columns
    extra = [c for c in X.columns if c not in model_features]
    if extra:
        X = X.drop(columns=extra)

    # Reorder
    X = X[model_features]
    return X


def main():
    parser = argparse.ArgumentParser(description='Evaluate LightGBM model on labeled CSV')
    parser.add_argument('--input', '-i', required=True, help='Path to labeled CSV (must include outcome)')
    parser.add_argument('--model', '-m', default=os.path.join('data', 'models', 'lightgbm_model.txt'), help='Path to LightGBM model file')
    parser.add_argument('--output', '-o', default=os.path.join('data', 'ml-results'), help='Output directory for eval JSON')
    parser.add_argument('--threshold', '-t', type=float, default=0.5, help='Classification threshold')
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    model = load_model(args.model)

    X, y = prepare_features_like_trainer(df)
    X = align_to_model_features(X, model)

    # Predict
    proba = model.predict(X)
    pred = (proba >= args.threshold).astype(int)

    acc = float(accuracy_score(y, pred))
    auc = float(roc_auc_score(y, proba)) if len(np.unique(y)) > 1 else float('nan')

    report = classification_report(y, pred, target_names=['LOSS', 'WIN'], zero_division=0, output_dict=True)

    print("\n==================================================")
    print("EVALUATION")
    print("==================================================")
    print(f"Rows: {len(df)}")
    print(f"Accuracy: {acc:.1%}")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"Threshold: {args.threshold}")

    # Save JSON
    os.makedirs(args.output, exist_ok=True)
    ts = datetime.now().strftime('%Y-%m-%dT%H-%M-%S')
    out_path = os.path.join(args.output, f"lgbm_eval_{ts}.json")

    payload = {
        'timestamp': ts,
        'input': args.input,
        'model': args.model,
        'rows': int(len(df)),
        'accuracy': acc,
        'auc': auc,
        'threshold': args.threshold,
        'report': report,
    }

    with open(out_path, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)

    print(f"\nSaved eval JSON: {out_path}")


if __name__ == '__main__':
    main()
