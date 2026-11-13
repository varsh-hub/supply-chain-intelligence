
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from src.feature_engineering import load_data, prepare_features
from src.recommendations import generate_recommendation


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/risk_model.pkl"
ANOMALY_DATA_PATH = "data/processed/anomaly_results.csv"


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Supply Chain Intelligence",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# LOAD ANOMALY DATA
# ============================================================

@st.cache_data
def load_anomaly_data():
    """Load saved anomaly-detection results."""
    return pd.read_csv(ANOMALY_DATA_PATH)


# ============================================================
# LOAD ML MODEL
# ============================================================

@st.cache_resource
def load_model():
    """Load the trained Random Forest model."""
    return joblib.load(MODEL_PATH)


# ============================================================
# LOAD APPLICATION DATA
# ============================================================

df = load_data()
model = load_model()
anomaly_df = load_anomaly_data()


# ============================================================
# PAGE HEADER
# ============================================================

st.title(
    "📦 Supply Chain Risk & Disruption Intelligence"
)

st.markdown(
    """
    Machine-learning based monitoring of supply-chain
    delivery risk and unusual order patterns.
    """
)


# ============================================================
# HISTORICAL KPIs
# ============================================================

total_orders = len(df)

late_risk_orders = int(
    df["Late_delivery_risk"].sum()
)

late_risk_percentage = (
    late_risk_orders / total_orders * 100
)

total_products = (
    df["Product Card Id"].nunique()
)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

with col2:
    st.metric(
        "Historical Late-Risk Orders",
        f"{late_risk_orders:,}"
    )

with col3:
    st.metric(
        "Historical Late-Risk %",
        f"{late_risk_percentage:.2f}%"
    )

with col4:
    st.metric(
        "Products",
        f"{total_products:,}"
    )


st.divider()


# ============================================================
# MACHINE LEARNING RISK PREDICTION
# ============================================================

st.subheader(
    "🤖 Machine Learning Risk Assessment"
)

st.markdown(
    """
    The Random Forest model estimates the probability that
    an order belongs to the late-delivery-risk class.
    """
)


# Prepare features using the same feature-engineering logic
X, y = prepare_features(df)


# Generate ML risk probabilities
risk_probabilities = (
    model.predict_proba(X)[:, 1]
)


# ============================================================
# RISK CLASSIFICATION
# ============================================================

def classify_risk(probability):

    if probability < 0.40:
        return "LOW"

    elif probability < 0.70:
        return "MEDIUM"

    else:
        return "HIGH"


risk_levels = [
    classify_risk(probability)
    for probability in risk_probabilities
]


# Add predictions to a copy of the dataframe
prediction_df = df.copy()

prediction_df[
    "ML Risk Probability"
] = risk_probabilities

prediction_df[
    "ML Risk Level"
] = risk_levels


# ============================================================
# ML RISK KPIs
# ============================================================

low_risk = (
    prediction_df["ML Risk Level"] == "LOW"
).sum()

medium_risk = (
    prediction_df["ML Risk Level"] == "MEDIUM"
).sum()

high_risk = (
    prediction_df["ML Risk Level"] == "HIGH"
).sum()


risk_col1, risk_col2, risk_col3 = st.columns(3)

with risk_col1:
    st.metric(
        "LOW Risk",
        f"{low_risk:,}"
    )

with risk_col2:
    st.metric(
        "MEDIUM Risk",
        f"{medium_risk:,}"
    )

with risk_col3:
    st.metric(
        "HIGH Risk",
        f"{high_risk:,}"
    )


# ============================================================
# ML RISK DISTRIBUTION
# ============================================================

risk_distribution = (
    prediction_df["ML Risk Level"]
    .value_counts()
    .reindex(
        ["LOW", "MEDIUM", "HIGH"],
        fill_value=0
    )
    .reset_index()
)

risk_distribution.columns = [
    "Risk Level",
    "Orders"
]


fig_ml = px.bar(
    risk_distribution,
    x="Risk Level",
    y="Orders",
    title="Machine Learning Risk Distribution",
    text="Orders"
)

fig_ml.update_traces(
    texttemplate="%{text:,}"
)

st.plotly_chart(
    fig_ml,
    width="stretch"
)


# ============================================================
# ANOMALY DETECTION
# ============================================================

st.subheader(
    "🔎 Anomaly Detection"
)

st.markdown(
    """
    The anomaly detector identifies unusual combinations of
    operational and financial order characteristics.

    An anomaly does not necessarily represent a confirmed
    real-world supply-chain disruption.
    """
)


total_anomalies = int(
    anomaly_df["anomaly_label"].sum()
)

anomaly_percentage = (
    total_anomalies
    / len(anomaly_df)
    * 100
)


anomaly_col1, anomaly_col2 = st.columns(2)

with anomaly_col1:
    st.metric(
        "Anomalous Orders",
        f"{total_anomalies:,}"
    )

with anomaly_col2:
    st.metric(
        "Anomaly Rate",
        f"{anomaly_percentage:.2f}%"
    )


# ============================================================
# ANOMALY DISTRIBUTION
# ============================================================

anomaly_counts = (
    anomaly_df["anomaly_label"]
    .map({
        0: "Normal",
        1: "Anomaly"
    })
    .value_counts()
    .reindex(
        ["Normal", "Anomaly"],
        fill_value=0
    )
    .reset_index()
)

anomaly_counts.columns = [
    "Pattern",
    "Orders"
]


fig_anomaly = px.bar(
    anomaly_counts,
    x="Pattern",
    y="Orders",
    title="Normal vs Anomalous Order Patterns",
    text="Orders"
)

fig_anomaly.update_traces(
    texttemplate="%{text:,}"
)

st.plotly_chart(
    fig_anomaly,
    width="stretch"
)


# ============================================================
# ANOMALOUS ORDER PATTERNS
# ============================================================

st.subheader(
    "⚠️ Anomalous Order Patterns"
)


anomaly_display = (
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


anomaly_display[
    "anomaly_score"
] = (
    anomaly_display[
        "anomaly_score"
    ].round(4)
)


st.dataframe(
    anomaly_display,
    width="stretch"
)


# ============================================================
# RECOMMENDATION ENGINE
# ============================================================

st.subheader(
    "💡 Risk-Based Recommendations"
)

st.markdown(
    """
    The recommendation engine converts detected risk and
    operational signals into explainable decision-support guidance.
    """
)


# Select medium- and high-risk orders
recommendation_orders = (
    prediction_df[
        prediction_df["ML Risk Level"].isin(
            ["HIGH", "MEDIUM"]
        )
    ]
    .head(20)
    .copy()
)


if len(recommendation_orders) > 0:

    selected_order = st.selectbox(
        "Select an order to view recommendations:",
        recommendation_orders["Order Id"].tolist()
    )

    selected_row = recommendation_orders[
        recommendation_orders["Order Id"] == selected_order
    ].iloc[0]


    # --------------------------------------------------------
    # Match anomaly result with selected order
    # --------------------------------------------------------

    selected_index = selected_row.name

    selected_anomaly = anomaly_df.iloc[
        selected_index
    ]


    # --------------------------------------------------------
    # Generate recommendations
    # --------------------------------------------------------

    recommendations = generate_recommendation(
        risk_level=selected_row[
            "ML Risk Level"
        ],

        risk_probability=selected_row[
            "ML Risk Probability"
        ],

        anomaly_label=int(
            selected_anomaly[
                "anomaly_label"
            ]
        ),

        shipping_mode=selected_row[
            "Shipping Mode"
        ],

        scheduled_shipping_days=selected_row[
            "Days for shipment (scheduled)"
        ],

        order_quantity=selected_row[
            "Order Item Quantity"
        ],

        order_profit=selected_row[
            "Order Profit Per Order"
        ]
    )


    # --------------------------------------------------------
    # Selected order information
    # --------------------------------------------------------

    st.markdown(
        f"""
        **Order ID:** {selected_order}

        **ML Risk Probability:** \
        {selected_row["ML Risk Probability"]:.2f}

        **Risk Level:** \
        {selected_row["ML Risk Level"]}

        **Anomaly Status:** \
        {"Anomalous" if selected_anomaly["anomaly_label"] == 1 else "Normal"}
        """
    )


    # --------------------------------------------------------
    # Recommended actions
    # --------------------------------------------------------

    st.markdown(
        "### Recommended Actions"
    )


    for recommendation in recommendations:

        st.info(
            f"💡 {recommendation}"
        )

else:

    st.info(
        "No medium- or high-risk orders are currently available."
    )


# ============================================================
# HISTORICAL DELIVERY RISK
# ============================================================

st.subheader(
    "Historical Delivery Risk"
)


risk_counts = (
    df["Late_delivery_risk"]
    .value_counts()
    .rename(
        {
            0: "No Late Risk",
            1: "Late Risk"
        }
    )
    .reset_index()
)

risk_counts.columns = [
    "Risk Status",
    "Orders"
]


fig = px.bar(
    risk_counts,
    x="Risk Status",
    y="Orders",
    title="Historical Late-Delivery Risk"
)

st.plotly_chart(
    fig,
    width="stretch"
)


# ============================================================
# SHIPPING MODE ANALYSIS
# ============================================================

st.subheader(
    "Late-Delivery Risk by Shipping Mode"
)


shipping_risk = (
    df.groupby("Shipping Mode")[
        "Late_delivery_risk"
    ]
    .mean()
    .reset_index()
)


shipping_risk[
    "Risk Percentage"
] = (
    shipping_risk[
        "Late_delivery_risk"
    ]
    * 100
)


fig_shipping = px.bar(
    shipping_risk,
    x="Shipping Mode",
    y="Risk Percentage",
    title="Historical Late-Delivery Risk by Shipping Mode",
    text="Risk Percentage"
)

fig_shipping.update_traces(
    texttemplate="%{text:.2f}%"
)

st.plotly_chart(
    fig_shipping,
    width="stretch"
)


# ============================================================
# HIGH-RISK ORDERS
# ============================================================

st.subheader(
    "🚨 High-Risk Orders"
)


high_risk_orders = (
    prediction_df[
        prediction_df["ML Risk Level"] == "HIGH"
    ]
    .copy()
)


display_columns = [
    "Order Id",
    "Product Name",
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "ML Risk Probability",
    "ML Risk Level"
]


high_risk_display = (
    high_risk_orders[
        display_columns
    ]
    .head(20)
    .copy()
)


high_risk_display[
    "ML Risk Probability"
] = (
    high_risk_display[
        "ML Risk Probability"
    ].round(3)
)


st.dataframe(
    high_risk_display,
    width="stretch"
)


# ============================================================
# RECENT ORDERS
# ============================================================

st.subheader(
    "Recent Orders"
)


recent_columns = [
    "Order Id",
    "Product Name",
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "delivery_delay_days",
    "Late_delivery_risk",
    "ML Risk Probability",
    "ML Risk Level"
]


recent_orders = (
    prediction_df[
        recent_columns
    ]
    .head(20)
    .copy()
)


recent_orders[
    "ML Risk Probability"
] = (
    recent_orders[
        "ML Risk Probability"
    ].round(3)
)


st.dataframe(
    recent_orders,
    width="stretch"
)
