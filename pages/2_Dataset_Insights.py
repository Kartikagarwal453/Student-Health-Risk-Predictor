"""Dataset exploration page."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from utils.charts import (
    boxplot_chart,
    categorical_distribution_chart,
    class_distribution_chart,
    correlation_heatmap,
    missing_values_chart,
    numeric_distribution_chart,
)
from utils.constants import APP_SHORT_TITLE, RAW_FEATURES, TARGET_COLUMN, TRAIN_DATA_PATH
from utils.helpers import load_css, metric_card


st.set_page_config(page_title=f"{APP_SHORT_TITLE} | Dataset Insights", page_icon="D", layout="wide")
load_css()


@st.cache_data(show_spinner=False)
def load_dataset(rows: int = 50000) -> pd.DataFrame:
    """Load a bounded dataset sample for interactive EDA."""

    return pd.read_csv(TRAIN_DATA_PATH, nrows=rows)


st.title("Dataset Insights")
st.caption("Explore quality, distributions, relationships, and outliers in the student health dataset.")

df = load_dataset()
numeric_features = df[RAW_FEATURES].select_dtypes(include="number").columns.tolist()
categorical_features = [feature for feature in RAW_FEATURES if feature not in numeric_features]

c1, c2, c3, c4 = st.columns(4)
with c1:
    metric_card("Rows Reviewed", f"{len(df):,}", "Bounded sample")
with c2:
    metric_card("Features", str(len(RAW_FEATURES)), "Model inputs")
with c3:
    metric_card("Classes", str(df[TARGET_COLUMN].nunique()), "Target labels")
with c4:
    metric_card("Missing Cells", f"{int(df.isna().sum().sum()):,}", "Data quality")

tab_quality, tab_distribution, tab_relationships, tab_outliers = st.tabs(
    ["Data Quality", "Distributions", "Relationships", "Outliers"]
)

with tab_quality:
    st.plotly_chart(missing_values_chart(df), width="stretch")
    st.dataframe(df.head(50), width="stretch", hide_index=True)

with tab_distribution:
    left, right = st.columns(2, gap="large")
    with left:
        st.plotly_chart(class_distribution_chart(df[TARGET_COLUMN]), use_container_width=True)
        selected_numeric = st.selectbox("Numerical feature", numeric_features)
        st.plotly_chart(
            numeric_distribution_chart(df, selected_numeric, TARGET_COLUMN),
            width="stretch",
        )
    with right:
        selected_category = st.selectbox("Categorical feature", categorical_features)
        st.plotly_chart(
            categorical_distribution_chart(df, selected_category, TARGET_COLUMN),
            width="stretch",
        )

with tab_relationships:
    st.plotly_chart(correlation_heatmap(df[numeric_features]), width="stretch")

with tab_outliers:
    outlier_feature = st.selectbox("Feature for boxplot", numeric_features, key="outlier_feature")
    q1 = df[outlier_feature].quantile(0.25)
    q3 = df[outlier_feature].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    outliers = df[(df[outlier_feature] < lower) | (df[outlier_feature] > upper)]

    metric_card("Outlier Count", f"{len(outliers):,}", f"IQR rule for {outlier_feature}")
    st.plotly_chart(boxplot_chart(df, outlier_feature, TARGET_COLUMN), width="stretch")
    st.dataframe(outliers.head(100), width="stretch", hide_index=True)
