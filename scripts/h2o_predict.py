#!/usr/bin/env python3
"""
H2O Prediction Script
Makes predictions using a trained H2O model
Can predict from CSV file or JSON input
"""

import h2o
import argparse
import json
import sys
import os

def init_h2o():
    """Initialize H2O cluster"""
    try:
        h2o.init(nthreads=-1, max_mem_size="2G")
        return True
    except Exception as e:
        print(f"ERROR: Failed to init H2O: {e}", file=sys.stderr)
        return False

def predict_from_csv(model_path: str, csv_path: str):
    """Make predictions from a CSV file"""

    # Load model
    model = h2o.load_model(model_path)

    # Load data
    data = h2o.import_file(csv_path)

    # Make predictions
    preds = model.predict(data)
    preds_df = preds.as_data_frame()

    results = []
    for i, row in preds_df.iterrows():
        win_prob = row.get('WIN', row.get('p1', 0.5))
        results.append({
            'prediction': 'WIN' if win_prob > 0.5 else 'LOSS',
            'win_probability': float(win_prob),
            'confidence': float(abs(win_prob - 0.5) * 2)
        })

    return results

def predict_single(model_path: str, features: dict):
    """Make a single prediction from features dict"""

    # Load model
    model = h2o.load_model(model_path)

    # Create H2O frame from features
    data = h2o.H2OFrame([features])

    # Make prediction
    preds = model.predict(data)
    preds_df = preds.as_data_frame()

    row = preds_df.iloc[0]
    win_prob = row.get('WIN', row.get('p1', 0.5))

    return {
        'prediction': 'WIN' if win_prob > 0.5 else 'LOSS',
        'win_probability': float(win_prob),
        'confidence': float(abs(win_prob - 0.5) * 2)
    }

def main():
    parser = argparse.ArgumentParser(description='H2O Predictions')
    parser.add_argument('--model', '-m', required=True, help='Path to H2O model')
    parser.add_argument('--csv', '-c', help='Path to CSV file for batch predictions')
    parser.add_argument('--json', '-j', help='JSON string with features for single prediction')
    parser.add_argument('--output', '-o', help='Output file path')

    args = parser.parse_args()

    if not init_h2o():
        sys.exit(1)

    try:
        if args.csv:
            results = predict_from_csv(args.model, args.csv)
        elif args.json:
            features = json.loads(args.json)
            results = [predict_single(args.model, features)]
        else:
            print("ERROR: Either --csv or --json required", file=sys.stderr)
            sys.exit(1)

        output = json.dumps(results, indent=2)

        if args.output:
            with open(args.output, 'w') as f:
                f.write(output)
            print(f"Predictions saved to {args.output}")
        else:
            print(output)

    finally:
        try:
            h2o.cluster().shutdown()
        except:
            pass

if __name__ == '__main__':
    main()
