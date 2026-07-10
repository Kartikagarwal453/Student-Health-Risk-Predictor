"""Model loading and prediction services."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd

from utils.constants import ENCODER_PATH, MODEL_PATH, PREPROCESSOR_PATH, RAW_FEATURES
from utils.feature_engineering import create_features
from utils.helpers import calculate_lifestyle_score


LOGGER = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Structured output for a single student prediction."""

    prediction: str
    confidence: float
    probabilities: dict[str, float]
    lifestyle_score: int


class HealthPredictor:
    """Thin service wrapper around the saved preprocessor, model, and encoder."""

    def __init__(self) -> None:
        try:
            self.model = joblib.load(MODEL_PATH)
            self.preprocessor = joblib.load(PREPROCESSOR_PATH)
            self.encoder = joblib.load(ENCODER_PATH)
        except Exception as exc:
            LOGGER.exception("Unable to load model artifacts.")
            raise RuntimeError(
                "Model artifacts could not be loaded. Confirm dependencies are installed "
                "and the model/ directory contains the saved .pkl files."
            ) from exc

    @property
    def classes(self) -> np.ndarray:
        """Return target class labels in model order."""

        return self.encoder.classes_

    def _prepare(self, records: pd.DataFrame) -> Any:
        featured = create_features(records)
        return self.preprocessor.transform(featured)

    def predict_one(self, student: dict[str, Any]) -> PredictionResult:
        """Predict health risk for one student."""

        frame = pd.DataFrame([student], columns=RAW_FEATURES)
        transformed = self._prepare(frame)
        encoded_prediction = self.model.predict(transformed).astype(int)
        probabilities = self.model.predict_proba(transformed)[0]
        prediction = self.encoder.inverse_transform(encoded_prediction)[0]
        probability_map = {
            str(label): float(probabilities[index])
            for index, label in enumerate(self.classes)
        }

        return PredictionResult(
            prediction=str(prediction),
            confidence=float(np.max(probabilities)),
            probabilities=probability_map,
            lifestyle_score=calculate_lifestyle_score(student),
        )

    def predict(self, student: dict[str, Any]) -> tuple[str, np.ndarray, np.ndarray]:
        """Backward-compatible tuple prediction API."""

        result = self.predict_one(student)
        probabilities = np.array([result.probabilities[label] for label in self.classes])
        return result.prediction, probabilities, self.classes

    def predict_batch(self, data: pd.DataFrame) -> pd.DataFrame:
        """Predict health risk for a batch of uploaded student records."""

        records = data.copy()
        missing = [column for column in RAW_FEATURES if column not in records.columns]
        if missing:
            raise ValueError(f"Uploaded file is missing: {', '.join(missing)}")

        transformed = self._prepare(records[RAW_FEATURES])
        encoded_predictions = self.model.predict(transformed).astype(int)
        probabilities = self.model.predict_proba(transformed)
        labels = self.encoder.inverse_transform(encoded_predictions)

        output = records.copy()
        output["predicted_health_condition"] = labels
        output["prediction_confidence"] = probabilities.max(axis=1)
        output["lifestyle_score"] = records.apply(calculate_lifestyle_score, axis=1)

        for index, class_name in enumerate(self.classes):
            output[f"probability_{class_name}"] = probabilities[:, index]

        return output
