"""Streamlit entry point for the Student Health Risk Prediction System."""

from __future__ import annotations

import logging
from datetime import datetime

import pandas as pd
import streamlit as st

from utils.charts import confidence_gauge, lifestyle_radar, probability_chart
from utils.constants import (
    APP_SHORT_TITLE,
    APP_TITLE,
    CATEGORY_OPTIONS,
    MODEL_METRICS,
    MODEL_NAME,
    NUMERIC_DEFAULTS,
    RAW_FEATURES,
)
from utils.helpers import (
    confidence_label,
    load_css,
    load_metadata,
    metric_card,
    prediction_badge_class,
)
from utils.predictor import HealthPredictor
from utils.recommendations import generate_recommendations
from utils.report_generator import generate_prediction_pdf


logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
LOGGER = logging.getLogger(__name__)

st.set_page_config(
    page_title=APP_SHORT_TITLE,
    page_icon="H",
    layout="wide",
    initial_sidebar_state="expanded",
)
load_css()


@st.cache_resource(show_spinner=False)
def get_predictor() -> HealthPredictor:
    """Load model artifacts once per Streamlit session."""

    return HealthPredictor()


def initialize_state() -> None:
    """Initialize Streamlit session state."""

    st.session_state.setdefault("prediction_history", [])


def build_sidebar() -> None:
    """Render consistent sidebar project information."""

    st.sidebar.markdown(
        """
        <div class="sidebar-brand">
            <div class="brand-icon">H</div>
            <div>
                <strong>Student Health AI</strong>
                <span>Risk intelligence dashboard</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.sidebar.info(
        "Use the prediction page for individual students, then open the insights "
        "pages from the Streamlit navigation."
    )
    st.sidebar.caption("Portfolio-grade ML application powered by XGBoost.")


def render_home() -> None:
    """Render the landing dashboard section."""

    metadata = load_metadata()
    history = st.session_state["prediction_history"]
    average_score = (
        sum(item["lifestyle_score"] for item in history) / len(history)
        if history
        else 0
    )

    st.markdown(
        """
        <section class="hero">
            <div>
                <p class="eyebrow">Machine Learning Health Risk Screening</p>
                <h1>AI Powered Student Health Risk Prediction System</h1>
                <p>
                    Predict fit, at-risk, and unhealthy student profiles from lifestyle,
                    health, and personal indicators using a trained XGBoost classifier.
                </p>
            </div>
        </section>
        """,
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        metric_card("Model", MODEL_NAME, f"v{metadata.get('version', '1.0.0')}")
    with col2:
        metric_card("Accuracy", f"{MODEL_METRICS['Accuracy']:.2f}%", "Validation")
    with col3:
        metric_card("Balanced Accuracy", f"{MODEL_METRICS['Balanced Accuracy']:.2f}%", "Macro view")
    with col4:
        metric_card("Macro F1", f"{MODEL_METRICS['Macro F1']:.2f}%", "Class weighted")
    with col5:
        metric_card("Students Predicted", str(len(history)), "This session")

    st.markdown("### Dashboard")
    left, right = st.columns([1.1, 0.9], gap="large")
    with left:
        st.markdown(
            """
            <div class="panel">
                <h3>Project Description</h3>
                <p>
                    This application transforms student lifestyle and health inputs into
                    engineered behavioral signals, applies the saved preprocessing pipeline,
                    and returns a calibrated risk-class prediction with practical
                    recommendations and report generation.
                </p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with right:
        metric_card(
            "Average Lifestyle Score",
            f"{average_score:.1f}/100" if history else "No predictions",
            "Calculated from session history",
        )

    if history:
        st.markdown("### Prediction History")
        history_df = pd.DataFrame(history)
        st.dataframe(history_df, use_container_width=True, hide_index=True)
    else:
        st.markdown(
            """
            <div class="empty-state">
                Prediction history appears here after the first student assessment.
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_prediction_page() -> None:
    """Render the individual prediction workflow."""

    st.markdown("## Individual Student Prediction")
    st.caption("Enter student information, run the trained model, and export a PDF summary.")

    try:
        predictor = get_predictor()
    except RuntimeError as exc:
        st.error(str(exc))
        st.stop()

    with st.form("student_prediction_form"):
        lifestyle, health, personal = st.columns(3, gap="large")
        with lifestyle:
            st.markdown("#### Lifestyle")
            sleep_duration = st.slider("Sleep duration (hours)", 3.0, 11.0, NUMERIC_DEFAULTS["sleep_duration"], 0.5)
            step_count = st.number_input("Daily steps", 0, 30000, NUMERIC_DEFAULTS["step_count"], step=500)
            exercise_duration = st.slider("Exercise duration (minutes)", 0, 240, NUMERIC_DEFAULTS["exercise_duration"], 5)
            water_intake = st.slider("Water intake (liters)", 0.5, 6.0, NUMERIC_DEFAULTS["water_intake"], 0.1)
            physical_activity_level = st.selectbox(
                "Physical activity level",
                CATEGORY_OPTIONS["physical_activity_level"],
            )
        with health:
            st.markdown("#### Health")
            heart_rate = st.number_input("Resting heart rate", 40, 140, NUMERIC_DEFAULTS["heart_rate"])
            bmi = st.number_input("BMI", 12.0, 45.0, NUMERIC_DEFAULTS["bmi"], step=0.1)
            calorie_expenditure = st.number_input(
                "Calorie expenditure",
                800,
                5000,
                NUMERIC_DEFAULTS["calorie_expenditure"],
                step=50,
            )
            stress_level = st.selectbox("Stress level", CATEGORY_OPTIONS["stress_level"])
            sleep_quality = st.selectbox("Sleep quality", CATEGORY_OPTIONS["sleep_quality"])
        with personal:
            st.markdown("#### Personal Information")
            diet_type = st.selectbox("Diet type", CATEGORY_OPTIONS["diet_type"])
            smoking_alcohol = st.selectbox("Smoking or alcohol exposure", CATEGORY_OPTIONS["smoking_alcohol"])
            gender = st.selectbox("Gender", CATEGORY_OPTIONS["gender"])

        submitted = st.form_submit_button("Run Prediction", use_container_width=True)

    student = {
        "sleep_duration": sleep_duration,
        "heart_rate": heart_rate,
        "bmi": bmi,
        "calorie_expenditure": calorie_expenditure,
        "step_count": step_count,
        "exercise_duration": exercise_duration,
        "water_intake": water_intake,
        "diet_type": diet_type,
        "stress_level": stress_level,
        "sleep_quality": sleep_quality,
        "physical_activity_level": physical_activity_level,
        "smoking_alcohol": smoking_alcohol,
        "gender": gender,
    }

    if submitted:
        with st.spinner("Engineering features and running XGBoost prediction..."):
            try:
                result = predictor.predict_one(student)
            except Exception as exc:
                LOGGER.exception("Prediction failed.")
                st.error(f"Prediction failed: {exc}")
                st.stop()

        st.session_state["prediction_history"].append(
            {
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                "prediction": result.prediction,
                "confidence": f"{result.confidence * 100:.2f}%",
                "lifestyle_score": result.lifestyle_score,
            }
        )
        st.session_state["latest_prediction"] = {
            "student": student,
            "result": result,
            "recommendations": generate_recommendations(student),
        }

    latest = st.session_state.get("latest_prediction")
    if not latest:
        st.markdown(
            """
            <div class="empty-state">
                Complete the form and run a prediction to see model output,
                probabilities, recommendations, and report generation.
            </div>
            """,
            unsafe_allow_html=True,
        )
        return

    result = latest["result"]
    recommendations = latest["recommendations"]
    classes = list(result.probabilities.keys())
    probabilities = list(result.probabilities.values())

    st.markdown("### Prediction Result")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(
            f'<div class="prediction-badge {prediction_badge_class(result.prediction)}">'
            f"{result.prediction.upper()}</div>",
            unsafe_allow_html=True,
        )
    with c2:
        metric_card("Confidence", f"{result.confidence * 100:.2f}%", confidence_label(result.confidence))
    with c3:
        metric_card("Lifestyle Score", f"{result.lifestyle_score}/100", "Behavioral score")

    chart_left, chart_right = st.columns(2, gap="large")
    with chart_left:
        st.plotly_chart(confidence_gauge(result.confidence), use_container_width=True)
    with chart_right:
        st.plotly_chart(probability_chart(classes, probabilities), use_container_width=True)
    st.plotly_chart(lifestyle_radar(latest["student"]), use_container_width=True)

    st.markdown("### General Recommendations")
    for item in recommendations:
        st.markdown(
            f"""
            <div class="recommendation">
                <strong>{item['priority']} - {item['area']}</strong>
                <p>{item['recommendation']}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    try:
        pdf_bytes = generate_prediction_pdf(
            latest["student"],
            result.prediction,
            result.confidence,
            result.probabilities,
            result.lifestyle_score,
            recommendations,
        )
        st.download_button(
            "Download PDF Report",
            data=pdf_bytes,
            file_name="student_health_prediction_report.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    except RuntimeError as exc:
        st.warning(str(exc))


def main() -> None:
    """Run the Streamlit app."""

    initialize_state()
    build_sidebar()
    tabs = st.tabs(["Home Dashboard", "Prediction"])
    with tabs[0]:
        render_home()
    with tabs[1]:
        render_prediction_page()


if __name__ == "__main__":
    main()
