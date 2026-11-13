import pandas as pd
import plotly.express as px

DATA_PATH = "data/processed/processed_data.csv"


def load_data():
    """Load the processed dataset."""
    return pd.read_csv(DATA_PATH)


def basic_analysis(df):
    """Display basic statistics about the processed dataset."""

    print("\n========== DATASET OVERVIEW ==========")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\n========== LATE DELIVERY RISK ==========")
    print(df["Late_delivery_risk"].value_counts())

    print("\n========== LATE DELIVERY RISK (%) ==========")
    print(df["Late_delivery_risk"].value_counts(normalize=True) * 100)

    print("\n========== DELIVERY DELAY ==========")
    print(df["delivery_delay_days"].describe())

    print("\n========== SHIPPING MODE ==========")
    print(df["Shipping Mode"].value_counts())

    print("\n========== DELIVERY STATUS ==========")
    print(df["Delivery Status"].value_counts())


def shipping_mode_analysis(df):
    """Analyze late-delivery risk by shipping mode."""

    result = (
        df.groupby("Shipping Mode")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    print("\n========== LATE DELIVERY RISK BY SHIPPING MODE (%) ==========")
    print(result)


def market_analysis(df):
    """Analyze late-delivery risk by market."""

    result = (
        df.groupby("Market")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
        * 100
    )

    print("\n========== LATE DELIVERY RISK BY MARKET (%) ==========")
    print(result)


def category_analysis(df):
    """Analyze late-delivery risk by product category."""

    result = (
        df.groupby("Category Name")["Late_delivery_risk"]
        .mean()
        .sort_values(ascending=False)
    )

    print("\n========== TOP 10 CATEGORIES BY LATE DELIVERY RISK ==========")
    print((result.head(10) * 100))


def create_charts(df):
    """Create basic EDA charts."""

    # Chart 1: Late delivery risk distribution
    risk_counts = (
        df["Late_delivery_risk"]
        .value_counts()
        .reset_index()
    )

    risk_counts.columns = ["Late_delivery_risk", "Count"]

    fig = px.bar(
        risk_counts,
        x="Late_delivery_risk",
        y="Count",
        title="Late Delivery Risk Distribution",
        labels={
            "Late_delivery_risk": "Late Delivery Risk",
            "Count": "Number of Orders"
        }
    )

    fig.show()

    # Chart 2: Delivery delay distribution
    fig = px.histogram(
        df,
        x="delivery_delay_days",
        nbins=30,
        title="Distribution of Delivery Delay",
        labels={
            "delivery_delay_days": "Delivery Delay (Days)"
        }
    )

    fig.show()

    # Chart 3: Late delivery risk by shipping mode
    mode_risk = (
        df.groupby("Shipping Mode")["Late_delivery_risk"]
        .mean()
        .reset_index()
    )

    mode_risk["Risk Percentage"] = (
        mode_risk["Late_delivery_risk"] * 100
    )

    fig = px.bar(
        mode_risk,
        x="Shipping Mode",
        y="Risk Percentage",
        title="Late Delivery Risk by Shipping Mode",
        labels={
            "Risk Percentage": "Late Delivery Risk (%)"
        }
    )

    fig.show()


if __name__ == "__main__":

    print("Loading processed dataset...")

    df = load_data()

    basic_analysis(df)
    shipping_mode_analysis(df)
    market_analysis(df)
    category_analysis(df)

    create_charts(df)