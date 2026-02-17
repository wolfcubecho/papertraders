#!/usr/bin/env python3
"""
H2O Model Training Script
Trains ML models on trading data using H2O AutoML or specific algorithms
"""

import h2o
from h2o.automl import H2OAutoML
from h2o.estimators import H2ORandomForestEstimator, H2OXGBoostEstimator, H2OGradientBoostingEstimator, H2OGeneralizedLinearEstimator
import argparse
import json
import os
import sys
from datetime import datetime

def init_h2o():
    """Initialize H2O cluster"""
    try:
        h2o.init(nthreads=-1, max_mem_size="4G")
        print(f"H2O cluster initialized: {h2o.cluster().cloud_name}")
        return True
    except Exception as e:
        print(f"ERROR: Failed to init H2O: {e}")
        return False

def train_model(csv_path: str, algorithm: str = "xgboost", target: str = "outcome"):
    """Train a model on the CSV data"""

    print(f"\n{'='*60}")
    print(f"H2O Model Training")
    print(f"{'='*60}")
    print(f"Data: {csv_path}")
    print(f"Algorithm: {algorithm}")
    print(f"Target: {target}")

    # Load data
    print("\nLoading data...")
    data = h2o.import_file(csv_path)
    print(f"Loaded {data.nrows} rows, {data.ncols} columns")

    # Convert target to factor for classification
    data[target] = data[target].asfactor()

    # Get feature columns (exclude target and non-predictive columns)
    exclude_cols = [target, 'symbol', 'timeframe', 'exit_reason', 'pnl_percent', 'entry_price']
    features = [c for c in data.columns if c not in exclude_cols]
    print(f"Features: {len(features)}")

    # Split data
    train, test = data.split_frame(ratios=[0.8], seed=42)
    print(f"Train: {train.nrows} rows, Test: {test.nrows} rows")

    # Train model based on algorithm
    print(f"\nTraining {algorithm} model...")

    if algorithm == "automl":
        model = H2OAutoML(max_runtime_secs=120, max_models=10, seed=42)
        model.train(x=features, y=target, training_frame=train)
        best_model = model.leader
        model_id = best_model.model_id
    elif algorithm == "xgboost":
        model = H2OXGBoostEstimator(
            ntrees=100,
            max_depth=6,
            learn_rate=0.1,
            seed=42
        )
        model.train(x=features, y=target, training_frame=train)
        best_model = model
        model_id = f"xgboost_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    elif algorithm == "drf":
        model = H2ORandomForestEstimator(
            ntrees=100,
            max_depth=20,
            seed=42
        )
        model.train(x=features, y=target, training_frame=train)
        best_model = model
        model_id = f"drf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    elif algorithm == "gbm":
        model = H2OGradientBoostingEstimator(
            ntrees=100,
            max_depth=6,
            learn_rate=0.1,
            seed=42
        )
        model.train(x=features, y=target, training_frame=train)
        best_model = model
        model_id = f"gbm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    elif algorithm == "glm":
        model = H2OGeneralizedLinearEstimator(family="binomial", seed=42)
        model.train(x=features, y=target, training_frame=train)
        best_model = model
        model_id = f"glm_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    else:
        print(f"ERROR: Unknown algorithm: {algorithm}")
        return None

    # Evaluate on test set
    print("\nEvaluating model...")
    perf = best_model.model_performance(test)

    # Get metrics
    try:
        auc = perf.auc() if hasattr(perf, 'auc') else 0
    except:
        auc = 0
    try:
        mpce = perf.mean_per_class_error()
        accuracy = 1 - mpce[0][1] if isinstance(mpce, list) else 1 - mpce
    except:
        accuracy = 0
    try:
        logloss = perf.logloss() if hasattr(perf, 'logloss') else 0
    except:
        logloss = 0

    # Get confusion matrix
    cm = perf.confusion_matrix()

    print(f"\n{'='*60}")
    print(f"RESULTS")
    print(f"{'='*60}")
    print(f"Model ID: {model_id}")
    print(f"AUC: {auc:.4f}")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"LogLoss: {logloss:.4f}")
    print(f"\nConfusion Matrix:")
    print(cm)

    # Variable importance
    if hasattr(best_model, 'varimp'):
        varimp = best_model.varimp()
        if varimp:
            print(f"\nTop 10 Important Features:")
            for i, (var, rel_imp, scaled_imp, pct) in enumerate(varimp[:10]):
                print(f"  {i+1}. {var}: {pct:.2%}")

    # Save model
    models_dir = os.path.join(os.path.dirname(csv_path), '..', 'models', 'h2o')
    os.makedirs(models_dir, exist_ok=True)

    model_path = h2o.save_model(model=best_model, path=models_dir, force=True)
    print(f"\nModel saved: {model_path}")

    # Save metadata
    metadata = {
        "model_id": model_id,
        "algorithm": algorithm,
        "accuracy": float(accuracy),
        "auc": float(auc),
        "logloss": float(logloss),
        "features": features,
        "train_rows": train.nrows,
        "test_rows": test.nrows,
        "model_path": model_path,
        "timestamp": datetime.now().isoformat()
    }

    metadata_path = os.path.join(models_dir, f"{model_id}_metadata.json")
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"Metadata saved: {metadata_path}")

    # Output JSON for Node.js to parse
    print(f"\n__JSON_OUTPUT__")
    print(json.dumps(metadata))
    print(f"__END_JSON__")

    return metadata

def predict(model_path: str, csv_path: str, output_path: str = None):
    """Make predictions using a saved model"""

    print(f"\n{'='*60}")
    print(f"H2O Prediction")
    print(f"{'='*60}")
    print(f"Model: {model_path}")
    print(f"Data: {csv_path}")

    # Load model
    model = h2o.load_model(model_path)
    print(f"Model loaded: {model.model_id}")

    # Load data
    data = h2o.import_file(csv_path)
    print(f"Data loaded: {data.nrows} rows")

    # Make predictions
    preds = model.predict(data)

    # Convert to pandas for easier handling
    preds_df = preds.as_data_frame()

    # Output predictions
    if output_path:
        preds_df.to_csv(output_path, index=False)
        print(f"Predictions saved: {output_path}")

    # Summary
    if 'WIN' in preds_df.columns:
        win_probs = preds_df['WIN']
        print(f"\nPrediction Summary:")
        print(f"  Mean WIN probability: {win_probs.mean():.4f}")
        print(f"  Trades above 60%: {(win_probs > 0.6).sum()}")
        print(f"  Trades above 70%: {(win_probs > 0.7).sum()}")

    return preds_df

def main():
    parser = argparse.ArgumentParser(description='H2O Model Training')
    parser.add_argument('action', choices=['train', 'predict', 'automl'], help='Action to perform')
    parser.add_argument('--data', '-d', required=True, help='Path to CSV data')
    parser.add_argument('--algorithm', '-a', default='xgboost',
                       choices=['xgboost', 'drf', 'gbm', 'glm', 'automl'],
                       help='Algorithm to use')
    parser.add_argument('--target', '-t', default='outcome', help='Target column')
    parser.add_argument('--model', '-m', help='Model path for predictions')
    parser.add_argument('--output', '-o', help='Output path for predictions')

    args = parser.parse_args()

    # Initialize H2O
    if not init_h2o():
        sys.exit(1)

    try:
        if args.action == 'train':
            train_model(args.data, args.algorithm, args.target)
        elif args.action == 'automl':
            train_model(args.data, 'automl', args.target)
        elif args.action == 'predict':
            if not args.model:
                print("ERROR: --model required for predictions")
                sys.exit(1)
            predict(args.model, args.data, args.output)
    finally:
        h2o.cluster().shutdown()
        print("\nH2O cluster shutdown")

if __name__ == '__main__':
    main()
