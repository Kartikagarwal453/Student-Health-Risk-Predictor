"""Project-wide constants for the Student Health Risk application."""

from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "model"
DATASET_DIR = BASE_DIR / "dataset"

APP_TITLE = "AI Powered Student Health Risk Prediction System"
APP_SHORT_TITLE = "Student Health AI"
APP_TAGLINE = "Predict • Analyze • Improve"
APP_VERSION = "1.0.0"
LOGO_PATH = BASE_DIR / "assets" / "logo.png"
MODEL_NAME = "XGBoost Classifier"
MODEL_VERSION = "1.0.0"
AUTHOR = "Kartik Agarwal"

MODEL_PATH = MODEL_DIR / "student_health_xgboost.pkl"
PREPROCESSOR_PATH = MODEL_DIR / "student_health_preprocessor.pkl"
ENCODER_PATH = MODEL_DIR / "health_label_encoder.pkl"
METADATA_PATH = MODEL_DIR / "model_metadata.json"

TRAIN_DATA_PATH = DATASET_DIR / "train.csv"
FEATURED_TRAIN_DATA_PATH = DATASET_DIR / "train_featured.csv"
FULL_DATA_PATH = DATASET_DIR / "student_health_dataset_50k.csv"

RAW_FEATURES = [
    "sleep_duration",
    "heart_rate",
    "bmi",
    "calorie_expenditure",
    "step_count",
    "exercise_duration",
    "water_intake",
    "diet_type",
    "stress_level",
    "sleep_quality",
    "physical_activity_level",
    "smoking_alcohol",
    "gender",
]

ENGINEERED_FEATURES = [
    "bmi_category",
    "sleep_category",
    "step_category",
    "exercise_category",
    "hydration_category",
    "activity_score",
    "sleep_activity_interaction",
]

MODEL_FEATURES = RAW_FEATURES + ENGINEERED_FEATURES
TARGET_COLUMN = "health_condition"
ID_COLUMN = "id"

TARGET_CLASSES = ["at-risk", "fit", "unhealthy"]

MODEL_METRICS = {
    "Accuracy": 96.59,
    "Balanced Accuracy": 86.01,
    "Macro F1": 90.48,
    "Macro Precision": 91.20,
    "Macro Recall": 86.01,
}

CATEGORY_OPTIONS = {
    "diet_type": ["balanced", "veg", "non-veg"],
    "stress_level": ["low", "medium", "high"],
    "sleep_quality": ["good", "average", "poor"],
    "physical_activity_level": ["active", "moderate", "sedentary"],
    "smoking_alcohol": ["no", "occasional", "yes"],
    "gender": ["female", "male", "other"],
}

NUMERIC_DEFAULTS = {
    "sleep_duration": 7.5,
    "heart_rate": 74,
    "bmi": 22.5,
    "calorie_expenditure": 2200,
    "step_count": 8500,
    "exercise_duration": 45,
    "water_intake": 2.5,
}

CLASS_COLORS = {
    "fit": "#16a34a",
    "at-risk": "#f59e0b",
    "unhealthy": "#dc2626",
}
