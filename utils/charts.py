"""Plotly visualizations for prediction, model, and dataset insights."""

from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import auc, confusion_matrix, precision_recall_curve, roc_curve
from sklearn.preprocessing import label_binarize

from utils.constants import CLASS_COLORS, PLOTLY_TEMPLATE


def _layout(fig: go.Figure, height: int = 420) -> go.Figure:
    """Apply consistent layout defaults."""

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=height,
        margin=dict(t=70, b=35, l=35, r=30),
        font=dict(family="Inter, Segoe UI, sans-serif"),
    )
    return fig


def confidence_gauge(confidence: float) -> go.Figure:
    """Create a prediction confidence gauge."""

    value = round(confidence * 100, 2)
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=value,
            number={"suffix": "%", "font": {"size": 34}},
            title={"text": "Prediction Confidence"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "#2563eb"},
                "steps": [
                    {"range": [0, 50], "color": "#fee2e2"},
                    {"range": [50, 75], "color": "#fef3c7"},
                    {"range": [75, 100], "color": "#dcfce7"},
                ],
            },
        )
    )
    return _layout(fig, height=330)


def probability_chart(classes: Iterable[str], probabilities: Iterable[float]) -> go.Figure:
    """Create a horizontal probability bar chart."""

    frame = pd.DataFrame(
        {
            "Class": list(classes),
            "Probability": [float(value) * 100 for value in probabilities],
        }
    ).sort_values("Probability")
    fig = px.bar(
        frame,
        x="Probability",
        y="Class",
        orientation="h",
        color="Class",
        color_discrete_map=CLASS_COLORS,
        text=frame["Probability"].map(lambda value: f"{value:.1f}%"),
        title="Prediction Probability by Class",
    )
    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_xaxes(range=[0, 105], ticksuffix="%")
    return _layout(fig, height=350)


def lifestyle_radar(student: dict) -> go.Figure:
    """Create a radar chart for lifestyle quality dimensions."""

    stress_map = {"low": 0.9, "medium": 0.55, "high": 0.2}
    bmi_score = 1 - min(abs(float(student["bmi"]) - 22) / 15, 1)
    values = [
        min(float(student["sleep_duration"]) / 8, 1),
        min(float(student["exercise_duration"]) / 60, 1),
        min(float(student["water_intake"]) / 3, 1),
        min(float(student["step_count"]) / 10000, 1),
        stress_map.get(student["stress_level"], 0.5),
        bmi_score,
    ]
    categories = ["Sleep", "Exercise", "Hydration", "Steps", "Stress", "BMI"]
    values.append(values[0])
    categories.append(categories[0])

    fig = go.Figure(
        go.Scatterpolar(
            r=values,
            theta=categories,
            fill="toself",
            line=dict(color="#2563eb", width=3),
            fillcolor="rgba(37, 99, 235, 0.20)",
        )
    )
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=False,
        title="Lifestyle Radar",
    )
    return _layout(fig, height=430)


def feature_importance_chart(frame: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Create a feature importance chart from Feature/Importance columns."""

    data = frame.sort_values("Importance", ascending=False).head(top_n)
    data = data.sort_values("Importance")
    fig = px.bar(
        data,
        x="Importance",
        y="Feature",
        orientation="h",
        color="Importance",
        color_continuous_scale="Viridis",
        title=f"Top {len(data)} Feature Importances",
    )
    fig.update_layout(coloraxis_showscale=False)
    return _layout(fig, height=620)


def confusion_matrix_chart(y_true: Iterable, y_pred: Iterable, labels: list[str]) -> go.Figure:
    """Create an interactive confusion matrix."""

    cm = confusion_matrix(y_true, y_pred, labels=labels)
    fig = px.imshow(
        cm,
        x=labels,
        y=labels,
        text_auto=True,
        color_continuous_scale="Blues",
        title="Interactive Confusion Matrix",
    )
    fig.update_xaxes(title="Predicted")
    fig.update_yaxes(title="Actual")
    return _layout(fig, height=520)


def roc_curve_chart(y_true: Iterable[int], probabilities: np.ndarray, class_names: list[str]) -> go.Figure:
    """Create one-vs-rest ROC curves."""

    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    fig = go.Figure()
    for index, name in enumerate(class_names):
        fpr, tpr, _ = roc_curve(y_true_bin[:, index], probabilities[:, index])
        fig.add_trace(
            go.Scatter(
                x=fpr,
                y=tpr,
                mode="lines",
                name=f"{name} AUC {auc(fpr, tpr):.3f}",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(dash="dash", color="#64748b"),
            name="Chance",
        )
    )
    fig.update_xaxes(title="False Positive Rate")
    fig.update_yaxes(title="True Positive Rate")
    fig.update_layout(title="ROC Curve")
    return _layout(fig, height=560)


def precision_recall_chart(
    y_true: Iterable[int],
    probabilities: np.ndarray,
    class_names: list[str],
) -> go.Figure:
    """Create one-vs-rest precision recall curves."""

    y_true_bin = label_binarize(y_true, classes=np.arange(len(class_names)))
    fig = go.Figure()
    for index, name in enumerate(class_names):
        precision, recall, _ = precision_recall_curve(
            y_true_bin[:, index],
            probabilities[:, index],
        )
        fig.add_trace(go.Scatter(x=recall, y=precision, mode="lines", name=name))
    fig.update_xaxes(title="Recall")
    fig.update_yaxes(title="Precision")
    fig.update_layout(title="Precision Recall Curve")
    return _layout(fig, height=560)


def class_distribution_chart(series: pd.Series) -> go.Figure:
    """Create a class distribution donut chart."""

    counts = series.value_counts().rename_axis("Class").reset_index(name="Count")
    fig = px.pie(
        counts,
        names="Class",
        values="Count",
        hole=0.55,
        color="Class",
        color_discrete_map=CLASS_COLORS,
        title="Class Distribution",
    )
    return _layout(fig, height=430)


def correlation_heatmap(df: pd.DataFrame) -> go.Figure:
    """Create a numeric correlation heatmap."""

    corr = df.corr(numeric_only=True)
    fig = px.imshow(
        corr,
        text_auto=".2f",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlation Heatmap",
    )
    return _layout(fig, height=680)


def missing_values_chart(df: pd.DataFrame) -> go.Figure:
    """Create a missing-values bar chart, or a no-missing annotation."""

    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values()
    if missing.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No missing values detected",
            x=0.5,
            y=0.5,
            showarrow=False,
            font=dict(size=22, color="#16a34a"),
        )
        fig.update_xaxes(visible=False)
        fig.update_yaxes(visible=False)
        return _layout(fig, height=260)

    fig = px.bar(
        x=missing.values,
        y=missing.index,
        orientation="h",
        labels={"x": "Missing rows", "y": "Feature"},
        color=missing.values,
        color_continuous_scale="Reds",
        title="Missing Values",
    )
    fig.update_layout(coloraxis_showscale=False)
    return _layout(fig, height=430)


def numeric_distribution_chart(df: pd.DataFrame, feature: str, target: str) -> go.Figure:
    """Create a histogram for a selected numeric feature."""

    fig = px.histogram(
        df,
        x=feature,
        color=target,
        marginal="box",
        nbins=40,
        color_discrete_map=CLASS_COLORS,
        title=f"{feature.replace('_', ' ').title()} Distribution",
    )
    return _layout(fig, height=460)


def categorical_distribution_chart(df: pd.DataFrame, feature: str, target: str) -> go.Figure:
    """Create a grouped categorical count chart."""

    counts = df.groupby([feature, target], observed=False).size().reset_index(name="Count")
    fig = px.bar(
        counts,
        x=feature,
        y="Count",
        color=target,
        barmode="group",
        color_discrete_map=CLASS_COLORS,
        title=f"{feature.replace('_', ' ').title()} Distribution",
    )
    return _layout(fig, height=460)


def boxplot_chart(df: pd.DataFrame, feature: str, target: str) -> go.Figure:
    """Create a class-level boxplot for an input feature."""

    fig = px.box(
        df,
        x=target,
        y=feature,
        color=target,
        points="outliers",
        color_discrete_map=CLASS_COLORS,
        title=f"{feature.replace('_', ' ').title()} by Health Condition",
    )
    return _layout(fig, height=470)


def prediction_pie_chart(series: pd.Series) -> go.Figure:
    """Create a pie chart for batch prediction counts."""

    return class_distribution_chart(series)
