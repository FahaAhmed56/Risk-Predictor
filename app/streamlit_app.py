"""Friendly web UI for a synthetic-data machine learning demonstration."""
import json
from pathlib import Path
import sys
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.predict import predict_one

st.set_page_config(page_title='Student Support AI | Demo', page_icon='🎓', layout='wide')
st.title('🎓 Student Support AI')
st.info('Educational demonstration only. All training data and labels are synthetic. Do not use these predictions to evaluate or penalize real students.')
st.write('Explore how a machine-learning classifier turns five inputs into an illustrative risk estimate.')

left, right = st.columns([1, 1])
with left:
    st.subheader('Student inputs')
    attendance = st.slider('Attendance (%)', 0, 100, 82)
    study = st.slider('Study hours per week', 0, 80, 14)
    previous = st.slider('Previous score (%)', 0, 100, 72)
    assignments = st.slider('Assignments completed (%)', 0, 100, 87)
    sleep = st.slider('Average sleep (hours)', 0.0, 24.0, 7.5, 0.5)
    submitted = st.button('Estimate demo risk', type='primary', use_container_width=True)
record = dict(attendance_pct=attendance, study_hours_week=study, previous_score=previous,
              assignments_completed_pct=assignments, sleep_hours=sleep)
with right:
    st.subheader('Model output')
    if submitted:
        try:
            result = predict_one(record, str(ROOT / 'models/risk_model.joblib'))
            st.metric('Illustrative risk probability', f"{result['risk_probability']:.1%}")
            st.write(result['risk_label'])
            st.progress(result['risk_probability'])
            st.caption('Probability is derived from simulated data. It is not calibrated for real learners.')
        except (FileNotFoundError, ValueError) as exc:
            st.error(str(exc))
    else:
        st.write('Set inputs and select **Estimate demo risk**.')

st.divider()
metrics_file = ROOT / 'models/metrics.json'
if metrics_file.exists():
    metrics = json.loads(metrics_file.read_text(encoding='utf-8'))
    st.subheader('Held-out evaluation on synthetic samples')
    cols = st.columns(4)
    for col, key in zip(cols, ['accuracy','precision','recall','roc_auc']):
        col.metric(key.replace('_',' ').title(), f"{metrics[key]:.1%}")
    importance = pd.Series(metrics['feature_importance']).sort_values(ascending=False)
    st.write('**Feature importance (model association, not causality)**')
    st.bar_chart(importance)
st.caption('Technical stack: Python · scikit-learn · pandas · Streamlit')
