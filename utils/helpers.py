"""Reusable UI and scoring helpers for the Streamlit app."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from utils.constants import METADATA_PATH, MODEL_METRICS


def load_css(css_path: str | Path = "style.css") -> None:
    """Inject the project stylesheet into Streamlit."""

    path = Path(css_path)
    if path.exists():
        st.markdown(f"<style>{path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def load_metadata() -> dict[str, Any]:
    """Load model metadata with safe defaults."""

    if METADATA_PATH.exists():
        return json.loads(METADATA_PATH.read_text(encoding="utf-8"))
    return {
        "model": "XGBoost",
        "accuracy": MODEL_METRICS["Accuracy"],
        "balanced_accuracy": MODEL_METRICS["Balanced Accuracy"],
        "macro_f1": MODEL_METRICS["Macro F1"],
        "version": "1.0.0",
    }


def metric_card(title: str, value: str, subtitle: str = "") -> None:
    """Render a compact metric card using custom CSS."""

    subtitle_html = f"<span>{subtitle}</span>" if subtitle else ""
    st.markdown(
        f"""
        <div class="metric-card">
            <p>{title}</p>
            <h3>{value}</h3>
            {subtitle_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def calculate_lifestyle_score(student: dict[str, Any] | pd.Series) -> int:
    """Calculate a 0-100 lifestyle score from interpretable health habits."""

    score = 100
    record = dict(student)

    if float(record.get("sleep_duration", 0)) < 7:
        score -= 15
    if record.get("sleep_quality") == "poor":
        score -= 10
    if record.get("stress_level") == "high":
        score -= 20
    elif record.get("stress_level") == "medium":
        score -= 8
    if float(record.get("exercise_duration", 0)) < 30:
        score -= 15
    if float(record.get("step_count", 0)) < 5000:
        score -= 15
    if float(record.get("water_intake", 0)) < 2:
        score -= 10
    bmi = float(record.get("bmi", 0))
    if bmi < 18.5 or bmi >= 30:
        score -= 10
    if record.get("smoking_alcohol") == "yes":
        score -= 15
    elif record.get("smoking_alcohol") == "occasional":
        score -= 6

    return int(max(min(score, 100), 0))


def confidence_label(confidence: float) -> str:
    """Convert a probability into a reader-friendly confidence band."""

    percent = confidence * 100
    if percent >= 90:
        return "Very high"
    if percent >= 75:
        return "High"
    if percent >= 60:
        return "Moderate"
    return "Low"


def prediction_badge_class(prediction: str) -> str:
    """Return the CSS class for a prediction label."""

    if prediction == "fit":
        return "badge-fit"
    if prediction == "at-risk":
        return "badge-risk"
    return "badge-unhealthy"
