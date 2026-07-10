# AI Powered Student Health Risk Prediction System

A production-ready Streamlit machine learning application that predicts student health risk as `fit`, `at-risk`, or `unhealthy` from lifestyle, health, and personal indicators.

## Overview

This project packages a trained XGBoost classifier into an interactive portfolio-quality application. It supports single-student prediction, CSV batch prediction, dataset exploration, model diagnostics, explainable AI, recommendation generation, and downloadable PDF reports.

## Features

- Modern Streamlit dashboard with custom CSS and responsive layout.
- XGBoost prediction service using saved model, preprocessor, and label encoder artifacts.
- Automatic feature engineering before every prediction.
- Individual prediction workflow with confidence gauge, probability chart, lifestyle radar, and PDF report.
- Personalized recommendation engine for sleep, stress, exercise, hydration, BMI, activity, and smoking or alcohol exposure.
- Batch CSV upload with automatic scoring, summary metrics, pie chart, and downloadable prediction CSV.
- Model insights page with architecture, metrics, feature importance, confusion matrix, ROC curve, precision recall curve, and SHAP explainability.
- Dataset insights page with missing values, correlation heatmap, class distribution, feature distributions, categorical analysis, boxplots, and outlier analysis.

## Dataset

Dataset: Student Health Risk Dataset  
Target column: `health_condition`

Target classes:

- `fit`
- `at-risk`
- `unhealthy`

Input features:

```text
sleep_duration
heart_rate
bmi
calorie_expenditure
step_count
exercise_duration
water_intake
diet_type
stress_level
sleep_quality
physical_activity_level
smoking_alcohol
gender
```

## Feature Engineering

The application creates these model-required engineered features automatically:

- `bmi_category`
- `sleep_category`
- `step_category`
- `exercise_category`
- `hydration_category`
- `activity_score`
- `sleep_activity_interaction`

## EDA

The Dataset Insights page provides:

- Missing value audit.
- Target class distribution.
- Numerical histograms with boxplot margins.
- Categorical feature distributions by class.
- Correlation heatmap.
- Boxplots and IQR-based outlier analysis.

## Model Building

Final model: XGBoost Classifier  
Saved artifacts:

```text
model/student_health_xgboost.pkl
model/student_health_preprocessor.pkl
model/health_label_encoder.pkl
model/model_metadata.json
```

## Hyperparameter Tuning

The final model artifact is packaged for deployment after experimentation and tuning in the project notebooks. The app is intentionally inference-focused, so retraining code is separated from the production prediction path.

## Results

| Metric | Score |
| --- | ---: |
| Accuracy | 96.59% |
| Balanced Accuracy | 86.01% |
| Macro F1 | 90.48% |

## Screenshots

Add deployment screenshots to the `screenshots/` folder after launching the app.

## Installation

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## Usage

```bash
streamlit run app.py
```

Open the local Streamlit URL in your browser. Use:

- Home Dashboard for metrics and session prediction history.
- Prediction for individual student scoring and PDF reports.
- Model Insights for diagnostics and SHAP explanations.
- Dataset Insights for EDA.
- Batch Prediction for CSV scoring.

## Deployment

### Streamlit Cloud

1. Push the repository to GitHub.
2. Create a Streamlit Cloud app from the repository.
3. Set the main file path to `app.py`.
4. Ensure the `model/` artifacts are included.

### Render

Use this start command:

```bash
streamlit run app.py --server.port $PORT --server.address 0.0.0.0
```

### Hugging Face Spaces

Create a Streamlit Space, upload the repository, and keep `requirements.txt` at the project root.

## Future Work

- Add model retraining pipeline with experiment tracking.
- Add authentication for institution-level usage.
- Store prediction history in a database.
- Add fairness analysis across demographic groups.
- Add monitoring for drift and prediction confidence trends.
