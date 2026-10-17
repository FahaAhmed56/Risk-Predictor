"""Create a reproducible, fully synthetic educational dataset."""
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
from src.features import FEATURES

def make_dataset(n=2500, seed=42):
    if n < 20:
        raise ValueError("n must be at least 20")
    rng = np.random.default_rng(seed)
    attendance = rng.uniform(35, 100, n)
    study = rng.uniform(0, 35, n)
    previous = rng.uniform(20, 100, n)
    assignments = rng.uniform(20, 100, n)
    sleep = rng.uniform(4, 10, n)
    # This rule is invented for a demonstration; it is NOT research or institutional data.
    latent = (0.07 * (70 - attendance) + 0.085 * (12 - study)
              + 0.075 * (65 - previous) + 0.06 * (70 - assignments)
              + 0.3 * abs(sleep - 7.5) + rng.normal(0, 1.8, n))
    probability = 1 / (1 + np.exp(-np.clip(latent / 3, -30, 30)))
    risk = rng.binomial(1, probability)
    frame = pd.DataFrame({"attendance_pct": attendance, "study_hours_week": study,
        "previous_score": previous, "assignments_completed_pct": assignments,
        "sleep_hours": sleep, "at_risk": risk})
    return frame.round({col: 2 for col in FEATURES})

def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sample student data")
    parser.add_argument('--rows', type=int, default=2500)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--output', default='data/synthetic_students.csv')
    args = parser.parse_args()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    data = make_dataset(args.rows, args.seed)
    data.to_csv(out, index=False)
    print(f"Saved {len(data)} synthetic rows to {out} (risk rate: {data.at_risk.mean():.1%})")

if __name__ == '__main__':
    main()
