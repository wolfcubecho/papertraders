#!/usr/bin/env python3
"""
LightGBM Prediction Server
Serves the 77% accuracy model via HTTP API
"""

import json
import os
from pathlib import Path
from flask import Flask, request, jsonify
import lightgbm as lgb
import numpy as np

app = Flask(__name__)

# Load model on startup
MODEL_DIR = Path(__file__).parent.parent / "data" / "models"
MODEL_PATH = MODEL_DIR / "lightgbm_model.txt"
METADATA_PATH = MODEL_DIR / "lightgbm_metadata.json"

model = None
metadata = None
feature_names = None

def load_model():
    global model, metadata, feature_names

    if not MODEL_PATH.exists():
        print(f"[LightGBM] Model not found at {MODEL_PATH}")
        return False

    try:
        model = lgb.Booster(model_file=str(MODEL_PATH))
        print(f"[LightGBM] Model loaded from {MODEL_PATH}")

        if METADATA_PATH.exists():
            with open(METADATA_PATH) as f:
                metadata = json.load(f)
            feature_names = metadata.get("features", [])
            print(f"[LightGBM] Accuracy: {metadata.get('accuracy', 0) * 100:.1f}%")
            print(f"[LightGBM] AUC: {metadata.get('auc', 0) * 100:.1f}%")
            print(f"[LightGBM] Features: {len(feature_names)}")

        return True
    except Exception as e:
        print(f"[LightGBM] Failed to load model: {e}")
        return False


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok" if model else "no_model",
        "model_id": metadata.get("modelId") if metadata else None,
        "accuracy": metadata.get("accuracy") if metadata else None,
        "auc": metadata.get("auc") if metadata else None,
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    Predict win probability for a trade setup.

    Request body:
    {
        "features": { ... feature dict ... }
    }

    Response:
    {
        "win_probability": 0.77,
        "confidence": 0.85,
        "prediction": "WIN",
        "model_id": "lightgbm_..."
    }
    """
    if model is None:
        return jsonify({"error": "Model not loaded"}), 500

    try:
        data = request.get_json()
        features_dict = data.get("features", {})

        # Build feature vector in correct order
        feature_vector = []
        for fname in feature_names:
            val = features_dict.get(fname, 0)
            # Handle boolean/string conversions
            if isinstance(val, bool):
                val = 1 if val else 0
            elif isinstance(val, str):
                # Encode categorical features
                if fname == "trend_direction":
                    val = {"up": 1, "down": -1, "none": 0}.get(val, 0)
                elif fname == "ob_type":
                    val = {"bull": 1, "bear": -1, "none": 0}.get(val, 0)
                elif fname == "fvg_type":
                    val = {"bull": 1, "bear": -1, "none": 0}.get(val, 0)
                elif fname == "ema_trend":
                    val = {"bullish": 1, "bearish": -1, "neutral": 0}.get(val, 0)
                elif fname == "rsi_state":
                    val = {"oversold": -1, "overbought": 1, "neutral": 0}.get(val, 0)
                elif fname == "direction":
                    val = {"long": 1, "short": -1}.get(val, 0)
                elif fname == "session":
                    val = {"london": 1, "new_york": 2, "asian": 3, "none": 0}.get(val, 0)
                elif fname == "pullback_fib":
                    val = {"0.382": 0.382, "0.5": 0.5, "0.618": 0.618, "0.786": 0.786, "none": 0}.get(val, 0)
                else:
                    val = 0
            feature_vector.append(float(val) if val is not None else 0.0)

        # Make prediction
        X = np.array([feature_vector])
        prob = model.predict(X)[0]

        # LightGBM returns probability of positive class
        win_prob = float(prob)
        confidence = abs(win_prob - 0.5) * 2  # How far from 50%

        return jsonify({
            "win_probability": win_prob,
            "confidence": confidence,
            "prediction": "WIN" if win_prob > 0.5 else "LOSS",
            "model_id": metadata.get("modelId") if metadata else "unknown",
            "model_accuracy": metadata.get("accuracy") if metadata else 0,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


@app.route("/feature_importance", methods=["GET"])
def feature_importance():
    """Get feature importance from the model"""
    if metadata is None:
        return jsonify({"error": "No metadata"}), 500

    importance = metadata.get("featureImportance", {})
    # Sort by importance
    sorted_features = sorted(importance.items(), key=lambda x: x[1], reverse=True)

    return jsonify({
        "features": [{"name": k, "importance": v} for k, v in sorted_features],
        "top_5": [k for k, v in sorted_features[:5]]
    })


if __name__ == "__main__":
    print("=" * 60)
    print("LightGBM Prediction Server")
    print("=" * 60)

    if load_model():
        print("\nStarting server on http://localhost:5555")
        print("Endpoints:")
        print("  GET  /health           - Server health check")
        print("  POST /predict          - Get prediction for features")
        print("  GET  /feature_importance - Get feature importance")
        print("=" * 60)
        app.run(host="0.0.0.0", port=5555, debug=False)
    else:
        print("Failed to load model. Run the learning loop first.")
