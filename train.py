"""Train the credit card fraud model and save everything the web app needs.

Usage:
    python train.py --data path/to/creditcard.csv [--model dt|rf]

Outputs (in this folder):
    model.pkl     -> {"model", "scaler", "features"}
    samples.json  -> a few real rows from your data for the "Load sample" buttons
"""
import argparse
import json

import joblib
import pandas as pd
from imblearn.over_sampling import SMOTE
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

FEATURES = [f"V{i}" for i in range(1, 29)] + ["Amount"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="path to creditcard.csv")
    ap.add_argument("--model", default="dt", choices=["dt", "rf"],
                    help="dt = Decision Tree (as in your notebook), rf = Random Forest")
    args = ap.parse_args()

    df = pd.read_csv(args.data).drop(columns=["Time"]).drop_duplicates()
    print(f"Rows after de-duplication: {len(df)}  | fraud: {int(df['Class'].sum())}")

    # Keep a few raw rows for the UI demo buttons (raw Amount, before scaling)
    samples = {
        "normal": df[df.Class == 0].sample(3, random_state=1)[FEATURES].round(6).to_dict("records"),
        "fraud": df[df.Class == 1].sample(3, random_state=1)[FEATURES].round(6).to_dict("records"),
    }
    with open("samples.json", "w") as f:
        json.dump(samples, f)

    X, y = df[FEATURES].copy(), df["Class"]
    # Split FIRST so the test set stays untouched by scaling/SMOTE (no leakage)
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # Scale Amount with a scaler we SAVE, so raw user input can be scaled the same way
    scaler = StandardScaler().fit(X_tr[["Amount"]])
    X_tr["Amount"] = scaler.transform(X_tr[["Amount"]])
    X_te["Amount"] = scaler.transform(X_te[["Amount"]])

    # Oversample only the training data
    X_res, y_res = SMOTE(random_state=42).fit_resample(X_tr, y_tr)

    clf = (RandomForestClassifier(n_estimators=100, n_jobs=-1, random_state=42)
           if args.model == "rf" else DecisionTreeClassifier(random_state=42))
    clf.fit(X_res, y_res)

    pred = clf.predict(X_te)
    print("\nHeld-out test set (original, imbalanced distribution):")
    print(confusion_matrix(y_te, pred))
    print(classification_report(y_te, pred, digits=3))

    joblib.dump({"model": clf, "scaler": scaler, "features": FEATURES}, "model.pkl")
    print("Saved model.pkl and samples.json")


if __name__ == "__main__":
    main()
