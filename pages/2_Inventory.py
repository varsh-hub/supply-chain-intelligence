
import streamlit as st
import pandas as pd
import plotly.express as px

from src.feature_engineering import load_data


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Inventory Analytics",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# LOAD DATA
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

st.title("📦 Inventory Analytics")

st.markdown(
    """
    This page analyses product demand, order quantities, sales,
    and product-level activity using the information available
    in the supply-chain dataset.
    """
)

st.divider()


# ============================================================
# DATA AVAILABILITY
# ============================================================

st.subheader("Inventory Data Availability")

inventory_columns = [
    column
    for column in df.columns
    if any(
        keyword in column.lower()
        for keyword in [
            "inventory",
            "stock",
            "warehouse",
            "reorder"
        ]
    )
]


if len(inventory_columns) == 0:

    st.warning(
        "Direct inventory-management fields are not available "
        "in the current DataCo dataset."
    )

    st.markdown(
        """
        The current dataset does not contain direct measurements
        such as:

        - Current stock quantity
        - Available inventory
        - Reorder point
        - Safety stock
        - Stockout events
        - Warehouse inventory
        - Inventory holding cost

        Therefore, this page uses order and product activity as
        inventory-related indicators rather than claiming to
        measure actual inventory levels.
        """
    )

else:

    st.success(
        "Inventory-related fields were detected."
    )

    st.write(
        "Available inventory fields:",
        inventory_columns
    )


# ============================================================
# INVENTORY-RELATED KPIs
# ============================================================

st.subheader("Product and Order Activity")

total_orders = len(df)

unique_products = (
    df["Product Card Id"].nunique()
)

total_quantity = (
    df["Order Item Quantity"].sum()
)

total_sales = (
    df["Sales"].sum()
)


col1, col2, col3, col4 = st.columns(4)

with col1:

    st.metric(
        "Total Orders",
        f"{total_orders:,}"
    )

with col2:

    st.metric(
        "Unique Products",
        f"{unique_products:,}"
    )

with col3:

    st.metric(
        "Total Units Ordered",
        f"{total_quantity:,}"
    )

with col4:

    st.metric(
        "Total Sales",
        f"${total_sales:,.2f}"
    )


# ============================================================
# ORDER QUANTITY DISTRIBUTION
# ============================================================

st.subheader(
    "Order Quantity Distribution"
)

quantity_distribution = (
    df["Order Item Quantity"]
    .value_counts()
    .sort_index()
    .reset_index()
)

quantity_distribution.columns = [
    "Order Quantity",
    "Orders"
]


fig_quantity = px.bar(
    quantity_distribution,
    x="Order Quantity",
    y="Orders",
    title="Orders by Quantity"
)

st.plotly_chart(
    fig_quantity,
    width="stretch"
)


# ============================================================
# TOP PRODUCTS BY ORDER VOLUME
# ============================================================

st.subheader(
    "Top Products by Order Volume"
)

top_products = (
    df.groupby("Product Name")
    .agg(
        Orders=("Order Id", "count"),
        Units_Ordered=("Order Item Quantity", "sum"),
        Sales=("Sales", "sum")
    )
    .reset_index()
    .sort_values(
        "Orders",
        ascending=False
    )
    .head(20)
)


top_products["Sales"] = (
    top_products["Sales"].round(2)
)


st.dataframe(
    top_products,
    width="stretch"
)


# ============================================================
# PRODUCT SALES ANALYSIS
# ============================================================

st.subheader(
    "Top Products by Sales"
)

top_sales_products = (
    df.groupby("Product Name")[
        "Sales"
    ]
    .sum()
    .reset_index()
    .sort_values(
        "Sales",
        ascending=False
    )
    .head(15)
)


fig_sales = px.bar(
    top_sales_products,
    x="Sales",
    y="Product Name",
    orientation="h",
    title="Top 15 Products by Sales"
)

st.plotly_chart(
    fig_sales,
    width="stretch"
)


# ============================================================
# PRODUCT STATUS
# ============================================================

st.subheader(
    "Product Status"
)

product_status = (
    df["Product Status"]
    .value_counts()
    .reset_index()
)

product_status.columns = [
    "Product Status",
    "Orders"
]


fig_status = px.pie(
    product_status,
    names="Product Status",
    values="Orders",
    title="Product Status Distribution"
)

st.plotly_chart(
    fig_status,
    width="stretch"
)


# ============================================================
# PRODUCT CATEGORY ANALYSIS
# ============================================================

st.subheader(
    "Product Category Activity"
)

category_summary = (
    df.groupby("Category Name")
    .agg(
        Orders=("Order Id", "count"),
        Units_Ordered=(
            "Order Item Quantity",
            "sum"
        ),
        Sales=("Sales", "sum")
    )
    .reset_index()
    .sort_values(
        "Orders",
        ascending=False
    )
    .head(15)
)

category_summary["Sales"] = (
    category_summary["Sales"].round(2)
)


st.dataframe(
    category_summary,
    width="stretch"
)


# ============================================================
# CATEGORY ORDER VOLUME CHART
# ============================================================

fig_category = px.bar(
    category_summary,
    x="Category Name",
    y="Orders",
    title="Top Product Categories by Order Volume"
)

fig_category.update_layout(
    xaxis_tickangle=-45
)

st.plotly_chart(
    fig_category,
    width="stretch"
)


# ============================================================
# INVENTORY RISK INTERPRETATION
# ============================================================

st.subheader(
    "Inventory Risk Interpretation"
)

st.markdown(
    """
    ### Available analytical indicators

    The current dataset can support indirect inventory-related
    analysis using:

    - Order quantity
    - Product order frequency
    - Product sales
    - Product category activity
    - Product status
    - Historical product demand patterns

    These indicators can help identify products with high order
    activity that may deserve additional inventory monitoring.

    However, they should not be interpreted as actual stock
    levels because the dataset does not contain current inventory
    quantities.
    """
)


# ============================================================
# FUTURE INVENTORY ANALYTICS
# ============================================================

st.subheader(
    "Future Inventory Analytics"
)

st.info(
    """
    A complete inventory-risk module could be added when
    inventory-management data becomes available.

    Recommended fields include:

    • Product_ID

    • Warehouse_ID

    • Current_Stock

    • Daily_Demand

    • Reorder_Point

    • Safety_Stock

    • Stockout_Flag

    • Lead_Time

    • Inventory_Holding_Cost

    • Inventory_Turnover
    """
)


# ============================================================
# CONCLUSION
# ============================================================

st.markdown(
    """
    ### Current System Decision

    The Inventory Analytics page uses measurable product and
    order activity from the available dataset while clearly
    distinguishing these indicators from actual inventory levels.

    This prevents unsupported inventory values from being
    introduced into the system and provides a clear foundation
    for future integration with warehouse or inventory-management
    data.
    """
)
