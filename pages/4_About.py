"""About page for the Student Health Risk Prediction application."""

from __future__ import annotations

import streamlit as st

from utils.constants import APP_SHORT_TITLE, APP_TITLE, MODEL_METRICS, RAW_FEATURES
from utils.helpers import load_css, metric_card


st.set_page_config(page_title=f"{APP_SHORT_TITLE} | About", page_icon="A", layout="wide")
load_css()

st.title("About")
st.markdown(
    f"""
    <div class="panel">
        <h3>{APP_TITLE}</h3>
        <p>
            This portfolio project demonstrates an end-to-end machine learning
            application for student health risk prediction. It combines feature
            engineering, a saved XGBoost classifier, interactive dashboards,
            CSV batch scoring, explainability, and PDF reporting.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

cols = st.columns(3)
with cols[0]:
    metric_card("Accuracy", f"{MODEL_METRICS['Accuracy']:.2f}%", "Model performance")
with cols[1]:
    metric_card("Balanced Accuracy", f"{MODEL_METRICS['Balanced Accuracy']:.2f}%", "Class balance")
with cols[2]:
    metric_card("Macro F1", f"{MODEL_METRICS['Macro F1']:.2f}%", "Portfolio headline")

st.markdown("### Application Features")
st.markdown(
    """
    - Individual student risk prediction with confidence and probability charts.
    - Automatic feature engineering for BMI, sleep, steps, exercise, hydration, and interaction signals.
    - Personalized recommendations based on sleep, stress, exercise, water, BMI, activity, and smoking or alcohol exposure.
    - Model insight page with feature importance, confusion matrix, ROC, precision recall, and SHAP explanations.
    - Dataset insight page with missing values, correlation, class distribution, distributions, boxplots, and outlier analysis.
    - Batch prediction workflow with CSV upload, summary charts, and downloadable predictions.
    - PDF report generation for individual predictions.
    """
)

st.markdown("### Model Input Features")
st.code("\n".join(RAW_FEATURES))

st.markdown("### Deployment Targets")
st.markdown(
    """
    The project is structured for Streamlit Cloud, Render, and Hugging Face Spaces.
    Install dependencies from `requirements.txt`, keep the saved artifacts in `model/`,
    and launch with `streamlit run app.py`.
    """
)
