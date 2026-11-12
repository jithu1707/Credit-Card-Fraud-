# Credit Card Fraud Detector

A Flask web app that serves a trained scikit-learn model for single-transaction fraud predictions.

## Run locally

```bash
python -m venv .venv
# Windows PowerShell: .venv\Scripts\Activate.ps1
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python app.py
```

Open http://127.0.0.1:5000.

## Deploy on Render

This repository includes `render.yaml`. In Render, create a **New → Blueprint** and connect this GitHub repository, or create a Web Service with:

- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn app:app`
- Instance: Free

If the project is inside a subfolder of your GitHub repository, set that folder as the Render service's Root Directory. If the project files are at the repository root, leave Root Directory blank.

The app needs `app.py`, `model.pkl`, `samples.json`, `templates/index.html`, and `requirements.txt`. The original training CSV is not needed at runtime. Keep `data/creditcard.csv` out of the Git repository to avoid large-file pushes and unnecessary deployment size. If it was already committed/tracked, remove it from Git's index while retaining your local copy:

```bash
git rm --cached data/creditcard.csv
git add .gitignore
 git commit -m "Exclude training dataset from deployment"
git push
```

The saved model was serialized with scikit-learn 1.9.0, so `requirements.txt` pins that version to avoid model compatibility warnings/errors.

## API

- `GET /` — dashboard
- `GET /healthz` — health check
- `GET /api/samples` — sample transactions for the demo buttons
- `POST /api/predict` — JSON body containing `V1` through `V28` and `Amount`

CSV upload prediction is intentionally not included in this deployment version.

## Important demo note

This is an educational machine-learning demonstration, not a payment-processing or real-world fraud-prevention system. Do not submit real cardholder or sensitive financial information.
