"""CSV batch prediction page."""

from __future__ import annotations

from io import StringIO

import pandas as pd
import streamlit as st

from utils.charts import prediction_pie_chart
from utils.constants import RAW_FEATURES
from utils.helpers import load_css, metric_card, page_intro, render_footer, render_header, render_sidebar
from utils.predictor import HealthPredictor


load_css()
render_sidebar()
render_header()


@st.cache_resource(show_spinner=False)
def load_predictor() -> HealthPredictor:
    """Load the model service."""

    return HealthPredictor()


def template_csv() -> str:
    """Return a one-row upload template CSV."""

    template = pd.DataFrame(
        [
            {
                "sleep_duration": 7.5,
                "heart_rate": 74,
                "bmi": 22.5,
                "calorie_expenditure": 2200,
                "step_count": 8500,
                "exercise_duration": 45,
                "water_intake": 2.5,
                "diet_type": "balanced",
                "stress_level": "low",
                "sleep_quality": "good",
                "physical_activity_level": "active",
                "smoking_alcohol": "no",
                "gender": "female",
            }
        ]
    )
    buffer = StringIO()
    template.to_csv(buffer, index=False)
    return buffer.getvalue()


page_intro("Batch Prediction", "Upload a CSV, score student records securely, and export prediction-ready results.")

with st.expander("Required CSV columns", expanded=False):
    st.code(", ".join(RAW_FEATURES))
    st.download_button(
        "Download CSV Template",
        data=template_csv(),
        file_name="student_health_batch_template.csv",
        mime="text/csv",
    )

uploaded = st.file_uploader("Upload student CSV", type=["csv"])

if uploaded is None:
    st.markdown(
        """
        <div class="empty-state">
            Upload a CSV containing the raw model input features to start batch scoring.
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.stop()

try:
    input_df = pd.read_csv(uploaded)
except Exception as exc:
    st.error(f"Could not read uploaded CSV: {exc}")
    st.stop()

missing = [column for column in RAW_FEATURES if column not in input_df.columns]
if missing:
    st.error(f"The upload is missing required columns: {', '.join(missing)}")
    st.stop()

try:
    predictor = load_predictor()
except RuntimeError as exc:
    st.error(str(exc))
    st.stop()

with st.spinner("Scoring uploaded students..."):
    try:
        predictions = predictor.predict_batch(input_df)
    except Exception as exc:
        st.error(f"Batch prediction failed: {exc}")
        st.stop()

c1, c2, c3 = st.columns(3)
with c1:
    metric_card("Rows Scored", f"{len(predictions):,}", "Uploaded students")
with c2:
    metric_card(
        "Average Confidence",
        f"{predictions['prediction_confidence'].mean() * 100:.2f}%",
        "Model probability",
    )
with c3:
    metric_card(
        "Average Lifestyle Score",
        f"{predictions['lifestyle_score'].mean():.1f}/100",
        "Batch average",
    )

left, right = st.columns([0.8, 1.2], gap="large")
with left:
    st.plotly_chart(
        prediction_pie_chart(predictions["predicted_health_condition"]),
        width="stretch",
    )
with right:
    counts = predictions["predicted_health_condition"].value_counts().reset_index()
    counts.columns = ["Prediction", "Count"]
    st.dataframe(counts, width="stretch", hide_index=True)

st.markdown("### Prediction Output")
st.dataframe(predictions, width="stretch", hide_index=True)
st.download_button(
    "Download Prediction CSV",
    data=predictions.to_csv(index=False).encode("utf-8"),
    file_name="student_health_batch_predictions.csv",
    mime="text/csv",
    width="stretch",
)
render_footer()
