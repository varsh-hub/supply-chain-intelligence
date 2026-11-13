
import streamlit as st
import pandas as pd
import joblib

from src.feature_engineering import (
    load_data,
    prepare_features
)

from src.recommendations import (
    generate_recommendation
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Recommendations",
    page_icon="💡",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/risk_model.pkl"
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
    """Load saved anomaly-detection results."""
    return pd.read_csv(ANOMALY_DATA_PATH)


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_risk_model():
    """Load the saved Random Forest model."""
    return joblib.load(MODEL_PATH)


df = load_supply_chain_data()
anomaly_df = load_anomaly_data()
model = load_risk_model()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("💡 Supply Chain Recommendations")

st.markdown(
    """
    This page converts machine-learning risk predictions,
    anomaly-detection results, and operational signals into
    explainable decision-support recommendations.
    """
)

st.divider()


# ============================================================
# PREPARE ML PREDICTIONS
# ============================================================

X, y = prepare_features(df)

risk_probability = (
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


risk_level = [
    classify_risk(probability)
    for probability in risk_probability
]


prediction_df = df.copy()

prediction_df[
    "ML Risk Probability"
] = risk_probability

prediction_df[
    "ML Risk Level"
] = risk_level


# ============================================================
# ADD ANOMALY INFORMATION
# ============================================================

# The anomaly results were generated from the same processed
# dataset without changing row order.

prediction_df[
    "Anomaly Label"
] = anomaly_df[
    "anomaly_label"
].values

prediction_df[
    "Anomaly Score"
] = anomaly_df[
    "anomaly_score"
].values


prediction_df[
    "Anomaly Status"
] = prediction_df[
    "Anomaly Label"
].map(
    {
        0: "Normal",
        1: "Anomalous"
    }
)


# ============================================================
# PAGE OVERVIEW
# ============================================================

total_orders = len(prediction_df)

high_risk_orders = (
    prediction_df[
        prediction_df["ML Risk Level"] == "HIGH"
    ]
)

medium_risk_orders = (
    prediction_df[
        prediction_df["ML Risk Level"] == "MEDIUM"
    ]
)

anomalous_orders = (
    prediction_df[
        prediction_df["Anomaly Label"] == 1
    ]
)


col1, col2, col3, col4 = st.columns(4)


with col1:

    st.metric(
        "Total Orders",
        f"{total_orders:,}"
    )


with col2:

    st.metric(
        "High Risk",
        f"{len(high_risk_orders):,}"
    )


with col3:

    st.metric(
        "Medium Risk",
        f"{len(medium_risk_orders):,}"
    )


with col4:

    st.metric(
        "Anomalous",
        f"{len(anomalous_orders):,}"
    )


# ============================================================
# ORDER SELECTION
# ============================================================

st.subheader(
    "🔍 Select an Order"
)

st.markdown(
    """
    Select a medium- or high-risk order to generate
    explainable recommendations.
    """
)


recommendation_orders = (
    prediction_df[
        prediction_df["ML Risk Level"].isin(
            [
                "HIGH",
                "MEDIUM"
            ]
        )
    ]
    .sort_values(
        "ML Risk Probability",
        ascending=False
    )
    .head(100)
    .copy()
)


if len(recommendation_orders) == 0:

    st.info(
        "No medium- or high-risk orders are currently available."
    )

    st.stop()


# ============================================================
# ORDER SELECTOR
# ============================================================

selected_order = st.selectbox(
    "Select an Order ID:",
    recommendation_orders[
        "Order Id"
    ].tolist()
)


selected_row = recommendation_orders[
    recommendation_orders[
        "Order Id"
    ] == selected_order
].iloc[0]


# ============================================================
# SELECTED ORDER INFORMATION
# ============================================================

st.subheader(
    "Selected Order Risk Profile"
)


info_col1, info_col2, info_col3 = st.columns(3)


with info_col1:

    st.metric(
        "ML Risk Probability",
        f"{selected_row['ML Risk Probability']:.3f}"
    )


with info_col2:

    st.metric(
        "Risk Level",
        selected_row[
            "ML Risk Level"
        ]
    )


with info_col3:

    st.metric(
        "Anomaly Status",
        selected_row[
            "Anomaly Status"
        ]
    )


# ============================================================
# ORDER DETAILS
# ============================================================

detail_col1, detail_col2 = st.columns(2)


with detail_col1:

    st.markdown(
        "### Order Information"
    )

    st.write(
        f"**Order ID:** "
        f"{selected_order}"
    )

    st.write(
        f"**Product:** "
        f"{selected_row['Product Name']}"
    )

    st.write(
        f"**Shipping Mode:** "
        f"{selected_row['Shipping Mode']}"
    )

    st.write(
        f"**Scheduled Shipping Days:** "
        f"{selected_row['Days for shipment (scheduled)']}"
    )


with detail_col2:

    st.markdown(
        "### Operational Information"
    )

    st.write(
        f"**Order Quantity:** "
        f"{selected_row['Order Item Quantity']}"
    )

    st.write(
        f"**Order Profit:** "
        f"{selected_row['Order Profit Per Order']:.2f}"
    )

    st.write(
        f"**Anomaly Score:** "
        f"{selected_row['Anomaly Score']:.4f}"
    )


# ============================================================
# GENERATE RECOMMENDATIONS
# ============================================================

recommendations = generate_recommendation(
    risk_level=selected_row[
        "ML Risk Level"
    ],

    risk_probability=selected_row[
        "ML Risk Probability"
    ],

    anomaly_label=int(
        selected_row[
            "Anomaly Label"
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


# ============================================================
# RECOMMENDED ACTIONS
# ============================================================

st.subheader(
    "💡 Recommended Actions"
)

for recommendation in recommendations:

    st.info(
        f"💡 {recommendation}"
    )


# ============================================================
# RISK DECISION LOGIC
# ============================================================

st.subheader(
    "Why These Recommendations Were Generated"
)

logic_items = []


if selected_row["ML Risk Level"] == "HIGH":

    logic_items.append(
        "The machine-learning model classified this order "
        "as HIGH risk."
    )

elif selected_row["ML Risk Level"] == "MEDIUM":

    logic_items.append(
        "The machine-learning model classified this order "
        "as MEDIUM risk."
    )

else:

    logic_items.append(
        "The machine-learning model classified this order "
        "as LOW risk."
    )


if selected_row["Anomaly Label"] == 1:

    logic_items.append(
        "The Isolation Forest model identified an unusual "
        "order pattern."
    )

else:

    logic_items.append(
        "The order was not classified as anomalous."
    )


if selected_row["Shipping Mode"] == "First Class":

    logic_items.append(
        "The order uses First Class shipping, which showed "
        "elevated historical late-delivery risk."
    )

elif selected_row["Shipping Mode"] == "Second Class":

    logic_items.append(
        "The order uses Second Class shipping, which showed "
        "elevated historical late-delivery risk."
    )


if selected_row[
    "Days for shipment (scheduled)"
] <= 2:

    logic_items.append(
        "The scheduled shipping window is short, so "
        "proactive monitoring is recommended."
    )


if selected_row[
    "Order Item Quantity"
] >= 5:

    logic_items.append(
        "The order quantity is relatively high and may "
        "require fulfillment-capacity review."
    )


if selected_row[
    "Order Profit Per Order"
] < 0:

    logic_items.append(
        "The order has negative profitability and may "
        "require review of fulfillment or shipping costs."
    )


for item in logic_items:

    st.write(
        f"• {item}"
    )


# ============================================================
# SYSTEM FLOW
# ============================================================

st.subheader(
    "Recommendation Engine Architecture"
)

st.markdown(
    """
    **Supply-Chain Data**

    ↓

    **Random Forest Prediction**

    ↓

    **Late-Delivery Risk Probability**

    +

    **Isolation Forest Anomaly Detection**

    ↓

    **Risk Engine**

    ↓

    **Rule-Based Recommendation Engine**

    ↓

    **Actionable Decision Support**
    """
)


# ============================================================
# IMPORTANT LIMITATION
# ============================================================

st.warning(
    """
    These recommendations are analytical decision-support
    suggestions. They are not guaranteed operational actions.

    The current dataset does not contain real-time information
    about inventory availability, supplier capacity, transportation
    incidents, weather, warehouse congestion, or confirmed
    disruption events.
    """
)
