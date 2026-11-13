
import streamlit as st
import pandas as pd
import plotly.express as px
import joblib

from src.feature_engineering import load_data, prepare_features


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="ML Predictions",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# FILE PATHS
# ============================================================

DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/risk_model.pkl"


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_supply_chain_data():
    """Load processed supply-chain data."""
    return load_data()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_risk_model():
    """Load the saved Random Forest model."""
    return joblib.load(MODEL_PATH)


df = load_supply_chain_data()
model = load_risk_model()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🤖 Machine Learning Predictions")

st.markdown(
    """
    This page uses the trained Random Forest classification model
    to estimate the probability of late-delivery risk for supply-chain
    orders.
    """
)

st.divider()


# ============================================================
# PREPARE FEATURES
# ============================================================

X, y = prepare_features(df)


# ============================================================
# GENERATE PREDICTIONS
# ============================================================

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
# PREDICTION KPIs
# ============================================================

total_orders = len(prediction_df)

low_risk = (
    prediction_df["ML Risk Level"] == "LOW"
).sum()

medium_risk = (
    prediction_df["ML Risk Level"] == "MEDIUM"
).sum()

high_risk = (
    prediction_df["ML Risk Level"] == "HIGH"
).sum()


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Predictions",
        f"{total_orders:,}"
    )

with col2:

    st.metric(
        "LOW Risk",
        f"{low_risk:,}"
    )

with col3:

    st.metric(
        "MEDIUM Risk",
        f"{medium_risk:,}"
    )

with col4:

    st.metric(
        "HIGH Risk",
        f"{high_risk:,}"
    )


# ============================================================
# RISK DISTRIBUTION
# ============================================================

st.subheader(
    "Predicted Risk Distribution"
)


risk_distribution = (
    prediction_df[
        "ML Risk Level"
    ]
    .value_counts()
    .reindex(
        [
            "LOW",
            "MEDIUM",
            "HIGH"
        ],
        fill_value=0
    )
    .reset_index()
)

risk_distribution.columns = [
    "Risk Level",
    "Orders"
]


fig_risk = px.bar(
    risk_distribution,
    x="Risk Level",
    y="Orders",
    title="Machine Learning Risk Levels",
    text="Orders"
)

fig_risk.update_traces(
    texttemplate="%{text:,}"
)

st.plotly_chart(
    fig_risk,
    width="stretch"
)


# ============================================================
# PROBABILITY DISTRIBUTION
# ============================================================

st.subheader(
    "Late-Delivery Risk Probability"
)

fig_probability = px.histogram(
    prediction_df,
    x="ML Risk Probability",
    nbins=50,
    title="Distribution of Predicted Late-Delivery Probability"
)

fig_probability.add_vline(
    x=0.40,
    line_dash="dash",
    annotation_text="Operating Threshold 0.40"
)

fig_probability.add_vline(
    x=0.70,
    line_dash="dash",
    annotation_text="HIGH Risk Threshold 0.70"
)

st.plotly_chart(
    fig_probability,
    width="stretch"
)


# ============================================================
# RISK THRESHOLDS
# ============================================================

st.subheader(
    "Risk Thresholds"
)

threshold_table = pd.DataFrame(
    {
        "Risk Level": [
            "LOW",
            "MEDIUM",
            "HIGH"
        ],
        "Probability Range": [
            "Below 0.40",
            "0.40 to below 0.70",
            "0.70 or higher"
        ],
        "Interpretation": [
            "Lower predicted late-delivery risk",
            "Requires monitoring",
            "Requires priority attention"
        ]
    }
)

st.dataframe(
    threshold_table,
    width="stretch"
)


# ============================================================
# HIGH-RISK PREDICTIONS
# ============================================================

st.subheader(
    "🚨 Highest-Risk Orders"
)


high_risk_orders = (
    prediction_df[
        prediction_df[
            "ML Risk Level"
        ] == "HIGH"
    ]
    .sort_values(
        "ML Risk Probability",
        ascending=False
    )
    .copy()
)


high_risk_columns = [
    "Order Id",
    "Product Name",
    "Shipping Mode",
    "Days for shipment (scheduled)",
    "ML Risk Probability",
    "ML Risk Level",
    "Order Item Quantity",
    "Order Profit Per Order"
]


high_risk_display = (
    high_risk_orders[
        high_risk_columns
    ]
    .head(25)
    .copy()
)


high_risk_display[
    "ML Risk Probability"
] = (
    high_risk_display[
        "ML Risk Probability"
    ].round(3)
)


high_risk_display[
    "Order Profit Per Order"
] = (
    high_risk_display[
        "Order Profit Per Order"
    ].round(2)
)


st.dataframe(
    high_risk_display,
    width="stretch"
)


# ============================================================
# ORDER-LEVEL PREDICTION
# ============================================================

st.subheader(
    "🔍 Individual Order Prediction"
)

order_ids = (
    prediction_df[
        "Order Id"
    ]
    .tolist()
)


selected_order = st.selectbox(
    "Select an Order ID:",
    order_ids
)


selected_order_data = prediction_df[
    prediction_df["Order Id"] == selected_order
].iloc[0]


selected_probability = (
    selected_order_data[
        "ML Risk Probability"
    ]
)


selected_risk = (
    selected_order_data[
        "ML Risk Level"
    ]
)


if selected_risk == "HIGH":

    st.error(
        f"HIGH RISK — predicted probability: "
        f"{selected_probability:.2f}"
    )

elif selected_risk == "MEDIUM":

    st.warning(
        f"MEDIUM RISK — predicted probability: "
        f"{selected_probability:.2f}"
    )

else:

    st.success(
        f"LOW RISK — predicted probability: "
        f"{selected_probability:.2f}"
    )


# ============================================================
# ORDER DETAILS
# ============================================================

detail_col1, detail_col2 = st.columns(2)


with detail_col1:

    st.markdown("### Order Information")

    st.write(
        f"**Order ID:** {selected_order}"
    )

    st.write(
        f"**Product:** "
        f"{selected_order_data['Product Name']}"
    )

    st.write(
        f"**Shipping Mode:** "
        f"{selected_order_data['Shipping Mode']}"
    )

    st.write(
        f"**Scheduled Shipping Days:** "
        f"{selected_order_data['Days for shipment (scheduled)']}"
    )


with detail_col2:

    st.markdown("### Risk Information")

    st.write(
        f"**Late-Delivery Probability:** "
        f"{selected_probability:.3f}"
    )

    st.write(
        f"**Risk Level:** "
        f"{selected_risk}"
    )

    st.write(
        f"**Order Quantity:** "
        f"{selected_order_data['Order Item Quantity']}"
    )

    st.write(
        f"**Order Profit:** "
        f"{selected_order_data['Order Profit Per Order']:.2f}"
    )


# ============================================================
# MODEL INTERPRETATION
# ============================================================

st.subheader(
    "Model Interpretation"
)

st.markdown(
    """
    The Random Forest model estimates the probability that an order
    belongs to the late-delivery-risk class.

    The probability is then passed to the Risk Engine thresholds:

    - **Below 0.40:** LOW
    - **0.40 to below 0.70:** MEDIUM
    - **0.70 or higher:** HIGH

    The prediction represents a statistical estimate based on
    patterns learned from historical data. It should therefore
    be interpreted as decision-support information rather than
    a guaranteed prediction of future delivery performance.
    """
)


# ============================================================
# MODEL ROLE
# ============================================================

st.info(
    """
    **Prediction pipeline:**

    Supply-Chain Data
    → Feature Engineering
    → Random Forest Model
    → Late-Delivery Probability
    → Risk Engine Threshold
    → LOW / MEDIUM / HIGH Risk
    """
)
