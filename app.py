"""Flask backend: serves the UI and a /predict API.

Run:  python app.py   ->  http://127.0.0.1:5000
"""
import json
import os

import joblib
import pandas as pd
from flask import Flask, jsonify, render_template, request

BASE = os.path.dirname(os.path.abspath(__file__))
app = Flask(__name__)

bundle = joblib.load(os.path.join(BASE, "model.pkl"))
model, scaler, FEATURES = bundle["model"], bundle["scaler"], bundle["features"]

try:
    with open(os.path.join(BASE, "samples.json")) as f:
        SAMPLES = json.load(f)
except FileNotFoundError:
    SAMPLES = {"normal": [], "fraud": []}


def parse_row(payload):
    """Validate one transaction dict -> DataFrame with scaled Amount."""
    missing = [c for c in FEATURES if c not in payload or payload[c] in ("", None)]
    if missing:
        raise ValueError(f"Missing values for: {', '.join(missing)}")
    try:
        row = {c: float(payload[c]) for c in FEATURES}
    except (TypeError, ValueError):
        raise ValueError("All fields must be numbers.")
    if row["Amount"] < 0:
        raise ValueError("Amount cannot be negative.")
    df = pd.DataFrame([row], columns=FEATURES)
    df["Amount"] = scaler.transform(df[["Amount"]])
    return df


@app.get("/")
def index():
    return render_template("index.html", features=FEATURES)


@app.get("/healthz")
def healthz():
    return jsonify(status="ok")


@app.get("/api/samples")
def samples():
    return jsonify(SAMPLES)


@app.post("/api/predict")
def predict():
    try:
        df = parse_row(request.get_json(force=True, silent=True) or {})
    except ValueError as e:
        return jsonify(error=str(e)), 400
    label = int(model.predict(df)[0])
    proba = float(model.predict_proba(df)[0][1]) if hasattr(model, "predict_proba") else float(label)
    return jsonify(is_fraud=bool(label), fraud_probability=round(proba, 4),
                   verdict="Fraudulent transaction" if label else "Normal transaction")


if __name__ == "__main__":
    # Local development only. Render starts the app with Gunicorn.
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
