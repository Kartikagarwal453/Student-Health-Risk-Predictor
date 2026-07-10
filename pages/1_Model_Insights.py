"""Model diagnostics and explainability page."""

from __future__ import annotations

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from utils.charts import (
    confusion_matrix_chart,
    feature_importance_chart,
    precision_recall_chart,
    roc_curve_chart,
)
from utils.constants import (
    APP_SHORT_TITLE,
    FEATURED_TRAIN_DATA_PATH,
    MODEL_METRICS,
    MODEL_NAME,
    RAW_FEATURES,
    TARGET_COLUMN,
)
from utils.feature_engineering import create_features
from utils.helpers import load_css, metric_card
from utils.predictor import HealthPredictor


LOGGER = logging.getLogger(__name__)

st.set_page_config(page_title=f"{APP_SHORT_TITLE} | Model Insights", page_icon="M", layout="wide")
load_css()


@st.cache_resource(show_spinner=False)
def load_predictor() -> HealthPredictor:
    """Load model artifacts."""

    return HealthPredictor()


@st.cache_data(show_spinner=False)
def load_model_sample(rows: int = 5000) -> pd.DataFrame:
    """Load a bounded sample for model diagnostics."""

    return pd.read_csv(FEATURED_TRAIN_DATA_PATH, nrows=rows)


def build_feature_importance(predictor: HealthPredictor) -> pd.DataFrame:
    """Return model feature importances with readable feature names."""

    importances = getattr(predictor.model, "feature_importances_", None)
    if importances is None:
        raise ValueError("The loaded model does not expose feature_importances_.")

    try:
        names = predictor.preprocessor.get_feature_names_out()
    except Exception:
        names = np.array([f"feature_{index}" for index in range(len(importances))])

    if len(names) != len(importances):
        names = np.array([f"feature_{index}" for index in range(len(importances))])

    cleaned_names = [
        str(name).replace("num__", "").replace("cat__", "").replace("_", " ")
        for name in names
    ]
    return pd.DataFrame({"Feature": cleaned_names, "Importance": importances})


def predict_sample(predictor: HealthPredictor, frame: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Predict labels and probabilities for a diagnostic sample."""

    featured = create_features(frame[RAW_FEATURES])
    transformed = predictor.preprocessor.transform(featured)
    encoded_true = predictor.encoder.transform(frame[TARGET_COLUMN])
    encoded_pred = predictor.model.predict(transformed).astype(int)
    probabilities = predictor.model.predict_proba(transformed)
    return encoded_true, encoded_pred, probabilities


st.title("Model Insights")
st.caption("Architecture, validation metrics, feature importance, performance curves, and SHAP explainability.")

metrics = st.columns(5)
with metrics[0]:
    metric_card("Model", MODEL_NAME, "Extreme Gradient Boosting")
with metrics[1]:
    metric_card("Accuracy", f"{MODEL_METRICS['Accuracy']:.2f}%", "Validation")
with metrics[2]:
    metric_card("Balanced Accuracy", f"{MODEL_METRICS['Balanced Accuracy']:.2f}%", "Imbalance-aware")
with metrics[3]:
    metric_card("Macro Precision", f"{MODEL_METRICS['Macro Precision']:.2f}%", "All classes")
with metrics[4]:
    metric_card("Macro F1", f"{MODEL_METRICS['Macro F1']:.2f}%", "Final score")

st.markdown(
    """
    <div class="panel">
        <h3>Model Architecture</h3>
        <p>
            The prediction service applies deterministic feature engineering, transforms
            numeric and categorical features through the saved preprocessing pipeline,
            and classifies health condition using an XGBoost multi-class classifier.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

try:
    predictor = load_predictor()
    sample = load_model_sample()
except Exception as exc:
    st.warning(f"Model diagnostics require the saved model dependencies to load successfully: {exc}")
    st.stop()

try:
    importance = build_feature_importance(predictor)
    st.plotly_chart(feature_importance_chart(importance), use_container_width=True)
except Exception as exc:
    st.warning(f"Feature importance could not be rendered: {exc}")

try:
    with st.spinner("Generating validation diagnostics from a bounded dataset sample..."):
        y_true, y_pred, probabilities = predict_sample(predictor, sample)
    labels = list(predictor.classes)

    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(
            confusion_matrix_chart(
                predictor.encoder.inverse_transform(y_true),
                predictor.encoder.inverse_transform(y_pred),
                labels,
            ),
            use_container_width=True,
        )
    with right:
        st.plotly_chart(roc_curve_chart(y_true, probabilities, labels), use_container_width=True)
    st.plotly_chart(precision_recall_chart(y_true, probabilities, labels), use_container_width=True)
except Exception as exc:
    LOGGER.exception("Diagnostic chart generation failed.")
    st.warning(f"Confusion matrix and curve generation failed: {exc}")

st.markdown("### Explainable AI")
try:
    import shap

    explain_sample = sample[RAW_FEATURES].head(150)
    featured = create_features(explain_sample)
    transformed = predictor.preprocessor.transform(featured)
    feature_names = predictor.preprocessor.get_feature_names_out()
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()

    explainer = shap.TreeExplainer(predictor.model)
    shap_values = explainer.shap_values(transformed)

    st.markdown("#### Global SHAP Summary")
    plt.figure()
    if isinstance(shap_values, list):
        shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False, max_display=15)
    else:
        shap.summary_plot(shap_values, transformed, feature_names=feature_names, show=False, max_display=15)
    st.pyplot(plt.gcf(), clear_figure=True)

    st.markdown("#### Local SHAP Waterfall")
    row_index = st.slider("Student row for local explanation", 0, len(explain_sample) - 1, 0)
    predicted_class_index = int(np.argmax(predictor.model.predict_proba(transformed[[row_index]])[0]))
    if isinstance(shap_values, list):
        values = shap_values[predicted_class_index][row_index]
        base_value = explainer.expected_value[predicted_class_index]
    else:
        values = shap_values[row_index, :, predicted_class_index]
        base_value = explainer.expected_value[predicted_class_index]
    explanation = shap.Explanation(
        values=values,
        base_values=base_value,
        data=transformed[row_index],
        feature_names=feature_names,
    )
    shap.plots.waterfall(explanation, show=False, max_display=12)
    st.pyplot(plt.gcf(), clear_figure=True)
except Exception as exc:
    st.info(f"SHAP explanations are available when compatible SHAP/XGBoost dependencies load successfully. Details: {exc}")
