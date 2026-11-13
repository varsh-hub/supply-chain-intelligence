
import streamlit as st
import pandas as pd

from src.feature_engineering import load_data


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Supplier Analytics",
    page_icon="🏭",
    layout="wide"
)


# ============================================================
# DATA
# ============================================================

DATA_PATH = "data/processed/processed_data.csv"


@st.cache_data
def load_supply_chain_data():
    """Load processed supply-chain data."""
    return load_data()


df = load_supply_chain_data()


# ============================================================
# PAGE HEADER
# ============================================================

st.title("🏭 Supplier Analytics")

st.markdown(
    """
    This page is intended to analyse supplier-level performance
    and supplier-related supply-chain risk.
    """
)

st.divider()


# ============================================================
# DATA AVAILABILITY CHECK
# ============================================================

supplier_columns = [
    column
    for column in df.columns
    if "supplier" in column.lower()
]


if len(supplier_columns) == 0:

    st.warning(
        "Supplier-level information is not available in the "
        "current DataCo dataset."
    )

    st.markdown(
        """
        ### Why supplier analysis cannot currently be performed

        The dataset does not contain direct supplier-level fields
        such as:

        - Supplier ID
        - Supplier Name
        - Supplier delivery performance
        - Supplier lead time
        - Supplier reliability
        - Supplier defect rate
        - Supplier capacity

        Therefore, the system does not fabricate supplier records
        or supplier performance values.

        This is an important data limitation because meaningful
        supplier-risk analysis requires supplier-level observations.
        """
    )

else:

    st.success(
        "Supplier-related fields were detected in the dataset."
    )

    st.write(
        "Supplier columns:",
        supplier_columns
    )


# ============================================================
# AVAILABLE ORDER-LEVEL INFORMATION
# ============================================================

st.subheader(
    "Available Order-Level Supply Chain Information"
)

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Total Orders",
        f"{len(df):,}"
    )

with col2:
    st.metric(
        "Unique Products",
        f"{df['Product Card Id'].nunique():,}"
    )

with col3:
    st.metric(
        "Markets",
        f"{df['Market'].nunique():,}"
    )


# ============================================================
# SHIPPING MODE INFORMATION
# ============================================================

st.subheader(
    "Shipping Modes"
)

shipping_mode_counts = (
    df["Shipping Mode"]
    .value_counts()
    .reset_index()
)

shipping_mode_counts.columns = [
    "Shipping Mode",
    "Orders"
]

st.dataframe(
    shipping_mode_counts,
    width="stretch"
)


# ============================================================
# DELIVERY RISK BY SHIPPING MODE
# ============================================================

st.subheader(
    "Delivery Risk by Shipping Mode"
)

shipping_risk = (
    df.groupby("Shipping Mode")[
        "Late_delivery_risk"
    ]
    .mean()
    .reset_index()
)

shipping_risk["Late Risk Percentage"] = (
    shipping_risk["Late_delivery_risk"] * 100
)

shipping_risk = shipping_risk.drop(
    columns=["Late_delivery_risk"]
)

shipping_risk = shipping_risk.sort_values(
    "Late Risk Percentage",
    ascending=False
)

st.dataframe(
    shipping_risk,
    width="stretch"
)


# ============================================================
# DATA REQUIREMENTS FOR FUTURE SUPPLIER ANALYSIS
# ============================================================

st.subheader(
    "Future Supplier Risk Analysis"
)

st.info(
    """
    Supplier-level risk analysis can be added when supplier data
    becomes available.

    Recommended fields include:

    • Supplier_ID
    • Supplier_Name
    • Supplier_Lead_Time
    • Supplier_On_Time_Rate
    • Supplier_Defect_Rate
    • Supplier_Order_Count
    • Supplier_Capacity
    • Supplier_Country
    """
)


# ============================================================
# CONCLUSION
# ============================================================

st.markdown(
    """
    ### Current System Decision

    The current system preserves the integrity of the available
    dataset by reporting the supplier-data limitation instead of
    generating artificial supplier analytics.

    Supplier analytics can therefore be extended as a future
    module when supplier-level data becomes available.
    """
)
