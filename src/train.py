"""Train a baseline model, evaluate on a held-out split, and save artifacts."""
import argparse
import json
from pathlib import Path
import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from src.features import FEATURES

def train(data_path='data/synthetic_students.csv', output_dir='models'):
    df = pd.read_csv(data_path)
    needed = FEATURES + ['at_risk']
    if df.empty or any(col not in df for col in needed):
        raise ValueError('Input CSV must include all feature columns and at_risk')
    if len(df) < 30 or df.at_risk.nunique() != 2:
        raise ValueError('Need >=30 rows and both target classes')
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df['at_risk'], test_size=0.2, stratify=df['at_risk'], random_state=42)
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('classifier', RandomForestClassifier(n_estimators=160, min_samples_leaf=4,
                                             random_state=42, n_jobs=-1, class_weight='balanced')),
    ])
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    metrics = {
        'dataset': 'SYNTHETIC: fabricated labels; educational demo only',
        'train_rows': len(X_train), 'test_rows': len(X_test),
        'accuracy': round(float(accuracy_score(y_test, predictions)), 4),
        'precision': round(float(precision_score(y_test, predictions, zero_division=0)), 4),
        'recall': round(float(recall_score(y_test, predictions, zero_division=0)), 4),
        'f1': round(float(f1_score(y_test, predictions, zero_division=0)), 4),
        'roc_auc': round(float(roc_auc_score(y_test, probability)), 4),
        'confusion_matrix': confusion_matrix(y_test, predictions).tolist(),
        'feature_importance': dict(zip(FEATURES, [round(float(v), 4) for v in model.named_steps['classifier'].feature_importances_])),
    }
    dest = Path(output_dir)
    dest.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, dest / 'risk_model.joblib')
    (dest / 'metrics.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    return metrics

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/synthetic_students.csv')
    parser.add_argument('--out', default='models')
    args = parser.parse_args()
    print(json.dumps(train(args.data, args.out), indent=2))

if __name__ == '__main__':
    main()
