"""Feature engineering used by the trained health risk model."""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from utils.constants import MODEL_FEATURES, RAW_FEATURES


LOGGER = logging.getLogger(__name__)


def validate_input_columns(df: pd.DataFrame) -> None:
    """Raise a helpful error when required raw model inputs are missing."""

    missing = [column for column in RAW_FEATURES if column not in df.columns]
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing required input column(s): {missing_text}")


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create all engineered features required by the saved preprocessor.

    Parameters
    ----------
    df:
        Raw student records containing the 13 input features.

    Returns
    -------
    pandas.DataFrame
        Data with raw and engineered model columns in the expected order.
    """

    validate_input_columns(df)
    data = df.copy()

    numeric_columns = [
        "sleep_duration",
        "heart_rate",
        "bmi",
        "calorie_expenditure",
        "step_count",
        "exercise_duration",
        "water_intake",
    ]
    for column in numeric_columns:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data["bmi_category"] = pd.cut(
        data["bmi"],
        bins=[0, 18.5, 25, 30, np.inf],
        labels=["underweight", "normal", "overweight", "obese"],
        include_lowest=True,
    ).astype(str)

    data["sleep_category"] = pd.cut(
        data["sleep_duration"],
        bins=[0, 6, 8, np.inf],
        labels=["insufficient", "recommended", "long"],
        include_lowest=True,
    ).astype(str)

    data["step_category"] = pd.cut(
        data["step_count"],
        bins=[0, 5000, 10000, np.inf],
        labels=["low", "moderate", "high"],
        include_lowest=True,
    ).astype(str)

    data["exercise_category"] = pd.cut(
        data["exercise_duration"],
        bins=[-1, 20, 45, np.inf],
        labels=["low", "moderate", "high"],
        include_lowest=True,
    ).astype(str)

    data["hydration_category"] = pd.cut(
        data["water_intake"],
        bins=[0, 1.5, 2.5, np.inf],
        labels=["low", "adequate", "high"],
        include_lowest=True,
    ).astype(str)

    data["activity_score"] = data["step_count"] * data["exercise_duration"]
    data["sleep_activity_interaction"] = (
        data["sleep_duration"] * data["exercise_duration"]
    )

    LOGGER.debug("Created engineered features for %s row(s).", len(data))
    return data[MODEL_FEATURES]
