"""Prediction function shared by the CLI and Streamlit."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from src.features import FEATURES, validate_record

def predict_one(record, model_path='models/risk_model.joblib'):
    values = validate_record(record)
    if not Path(model_path).is_file():
        raise FileNotFoundError('Model not found. First run: python -m src.data && python -m src.train')
    # Load model artifacts ONLY from your own trusted project; joblib uses pickle.
    model = joblib.load(model_path)
    risk = float(model.predict_proba(pd.DataFrame([values], columns=FEATURES))[0, 1])
    return {'risk_probability': round(risk, 4), 'risk_label': 'Higher synthetic risk' if risk >= 0.5 else 'Lower synthetic risk'}

def main():
    parser = argparse.ArgumentParser(description='Predict from a JSON object of feature values')
    parser.add_argument('--json', required=True, help='JSON object containing all five feature fields')
    args = parser.parse_args()
    print(json.dumps(predict_one(json.loads(args.json)), indent=2))

if __name__ == '__main__':
    main()
