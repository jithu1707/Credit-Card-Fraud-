# Credit Card Fraud Detector

Flask backend + browser frontend around the model from your notebook.

## Setup
```
pip install -r requirements.txt
python train.py --data data/creditcard.csv     # ~1 min; creates model.pkl + samples.json
python app.py                                     # open http://127.0.0.1:5000
```
Use `--model rf` for a Random Forest instead of a Decision Tree.

## Usage
- **Single transaction:** type Amount + V1–V28 (or click "Load normal/fraud sample"), then Check.
- **Upload CSV:** a file with columns V1–V28 and Amount (Time/Class ignored) -> per-row verdicts.

API: `POST /api/predict` (JSON of the 29 fields), `POST /api/predict_csv` (multipart file).
