import json
from pathlib import Path
import pytest
from src.data import make_dataset
from src.features import FEATURES, validate_record
from src.train import train
from src.predict import predict_one

def test_generation_is_deterministic():
    a = make_dataset(100, seed=123)
    b = make_dataset(100, seed=123)
    assert a.equals(b)
    assert set(FEATURES + ['at_risk']) == set(a.columns)
    assert set(a['at_risk'].unique()).issubset({0, 1})

def test_invalid_inputs():
    valid = dict(attendance_pct=70, study_hours_week=12, previous_score=70,
                 assignments_completed_pct=70, sleep_hours=8)
    assert validate_record(valid) == {key: float(value) for key, value in valid.items()}
    with pytest.raises(ValueError):
        validate_record({**valid, 'sleep_hours': 25})
    with pytest.raises(ValueError):
        validate_record({**valid, 'previous_score': 'high'})

def test_end_to_end(tmp_path):
    csv_path = tmp_path / 'data.csv'
    make_dataset(300, seed=100).to_csv(csv_path, index=False)
    model_dir = tmp_path / 'models'
    metrics = train(csv_path, model_dir)
    assert (model_dir / 'risk_model.joblib').exists()
    assert 0 <= metrics['roc_auc'] <= 1
    assert metrics['train_rows'] + metrics['test_rows'] == 300
    record = dict(attendance_pct=80, study_hours_week=12, previous_score=75,
                  assignments_completed_pct=90, sleep_hours=7)
    result = predict_one(record, model_dir / 'risk_model.joblib')
    assert 0 <= result['risk_probability'] <= 1
