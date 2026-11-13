
import streamlit as st
import pandas as pd
import plotly.express as px

from src.feature_engineering import load_data


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Disruption Detection",
    page_icon="⚠️",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/processed/processed_data.csv"
ANOMALY_DATA_PATH = "data/processed/anomaly_results.csv"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_supply_chain_data():
    """Load processed supply-chain data."""
    return load_data()


@st.cache_data
def load_anomaly_data():
    """Load saved Isolation Forest results."""
    return pd.read_csv(ANOMALY_DATA_PATH)


df = load_supply_chain_data()
anomaly_df = load_anomaly_data()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("⚠️ Disruption & Anomaly Detection")

st.markdown(
    """
    This page uses the Isolation Forest anomaly-detection model
    to identify unusual supply-chain order patterns.

    An anomalous order is not automatically a confirmed real-world
    disruption. It represents an observation whose characteristics
    differ from the normal patterns found in the dataset.
    """
)

st.divider()


# ============================================================
# ANOMALY OVERVIEW
# ============================================================

st.subheader(
    "Anomaly Detection Overview"
)

total_orders = len(anomaly_df)

total_anomalies = int(
    anomaly_df["anomaly_label"].sum()
)

normal_orders = (
    total_orders - total_anomalies
)

anomaly_rate = (
    total_anomalies
    / total_orders
    * 100
)


col1, col2, col3 = st.columns(3)

with col1:

    st.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

with col2:

    st.metric(
        "Anomalous Orders",
        f"{total_anomalies:,}"
    )

with col3:

    st.metric(
        "Anomaly Rate",
        f"{anomaly_rate:.2f}%"
    )


# ============================================================
# NORMAL VS ANOMALY
# ============================================================

st.subheader(
    "Normal vs Anomalous Orders"
)

pattern_counts = pd.DataFrame(
    {
        "Pattern": [
            "Normal",
            "Anomaly"
        ],
        "Orders": [
            normal_orders,
            total_anomalies
        ]
    }
)


fig_pattern = px.bar(
    pattern_counts,
    x="Pattern",
    y="Orders",
    title="Normal vs Anomalous Order Distribution",
    text="Orders"
)

fig_pattern.update_traces(
    texttemplate="%{text:,}"
)

st.plotly_chart(
    fig_pattern,
    width="stretch"
)


# ============================================================
# ANOMALY SCORE DISTRIBUTION
# ============================================================

st.subheader(
    "Anomaly Score Distribution"
)

fig_score = px.histogram(
    anomaly_df,
    x="anomaly_score",
    nbins=50,
    title="Distribution of Anomaly Scores"
)

fig_score.add_vline(
    x=0,
    line_dash="dash"
)

st.plotly_chart(
    fig_score,
    width="stretch"
)


st.markdown(
    """
    **Interpretation:**

    The anomaly score represents how unusual an observation is
    according to the Isolation Forest model.

    Higher scores in the saved anomaly results indicate stronger
    anomaly characteristics after the score transformation used
    by the project.
    """
)


# ============================================================
# MOST ANOMALOUS ORDERS
# ============================================================

st.subheader(
    "Most Anomalous Orders"
)

top_anomalies = (
    anomaly_df[
        anomaly_df["anomaly_label"] == 1
    ]
    .sort_values(
        "anomaly_score",
        ascending=False
    )
    .head(20)
    .copy()
)


top_anomalies[
    "anomaly_score"
] = (
    top_anomalies[
        "anomaly_score"
    ].round(4)
)


st.dataframe(
    top_anomalies,
    width="stretch"
)


# ============================================================
# ANOMALY FEATURE ANALYSIS
# ============================================================

st.subheader(
    "Anomaly Characteristics"
)

numeric_features = [
    column
    for column in anomaly_df.columns
    if column not in [
        "anomaly_label",
        "anomaly_score"
    ]
]


selected_feature = st.selectbox(
    "Select a feature to compare normal and anomalous orders:",
    numeric_features
)


feature_comparison = (
    anomaly_df
    .groupby("anomaly_label")[
        selected_feature
    ]
    .mean()
    .reset_index()
)


feature_comparison[
    "Pattern"
] = (
    feature_comparison[
        "anomaly_label"
    ]
    .map(
        {
            0: "Normal",
            1: "Anomaly"
        }
    )
)


fig_feature = px.bar(
    feature_comparison,
    x="Pattern",
    y=selected_feature,
    title=f"Average {selected_feature}: Normal vs Anomaly"
)

st.plotly_chart(
    fig_feature,
    width="stretch"
)


# ============================================================
# ANOMALY FEATURE STATISTICS
# ============================================================

st.subheader(
    "Selected Feature Statistics"
)

normal_values = (
    anomaly_df[
        anomaly_df["anomaly_label"] == 0
    ][selected_feature]
)

anomaly_values = (
    anomaly_df[
        anomaly_df["anomaly_label"] == 1
    ][selected_feature]
)


stat_col1, stat_col2 = st.columns(2)

with stat_col1:

    st.markdown("### Normal Orders")

    st.metric(
        "Average",
        f"{normal_values.mean():.2f}"
    )

    st.metric(
        "Median",
        f"{normal_values.median():.2f}"
    )


with stat_col2:

    st.markdown("### Anomalous Orders")

    st.metric(
        "Average",
        f"{anomaly_values.mean():.2f}"
    )

    st.metric(
        "Median",
        f"{anomaly_values.median():.2f}"
    )


# ============================================================
# HIGH-RISK ANOMALOUS ORDERS
# ============================================================

st.subheader(
    "Anomalous Orders with Additional Risk Signals"
)


# The anomaly dataset has the anomaly result while the processed
# dataset contains the late-delivery-risk target.
#
# Both datasets were generated from the same processed dataset
# without reordering, so the row positions can be aligned here.

combined_df = anomaly_df.copy()

combined_df[
    "Late_delivery_risk"
] = df[
    "Late_delivery_risk"
].values


high_risk_anomalies = (
    combined_df[
        (combined_df["anomaly_label"] == 1)
        &
        (combined_df["Late_delivery_risk"] == 1)
    ]
    .sort_values(
        "anomaly_score",
        ascending=False
    )
    .head(20)
)


st.dataframe(
    high_risk_anomalies,
    width="stretch"
)


# ============================================================
# INTERPRETATION
# ============================================================

st.subheader(
    "Disruption Intelligence Interpretation"
)

st.markdown(
    """
    ### What this module identifies

    The Isolation Forest model identifies unusual combinations
    of supply-chain order characteristics.

    These may include unusual:

    - Profitability
    - Sales values
    - Product prices
    - Order quantities
    - Scheduled shipping durations
    - Time-related order patterns

    ### Important distinction

    An anomaly is not equivalent to a confirmed disruption.

    The current dataset does not contain direct disruption-event
    labels such as:

    - Transportation failure
    - Supplier shutdown
    - Warehouse outage
    - Severe weather event
    - Port disruption
    - Stockout event

    Therefore, this module should be described as **anomaly and
    disruption-signal detection**, rather than confirmed disruption
    detection.

    The detected anomalies can be used as an early-warning signal
    for further investigation.
    """
)


# ============================================================
# SYSTEM ROLE
# ============================================================

st.info(
    """
    **System role:**

    Historical Supply-Chain Data
    → Isolation Forest
    → Anomaly Score
    → Anomaly Classification
    → Disruption/Risk Signal
    → Further Investigation
    """
)
