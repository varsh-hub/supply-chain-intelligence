import pandas as pd

DATA_PATH = "data/processed/processed_data.csv"


def load_data():
    """Load the processed dataset."""
    return pd.read_csv(DATA_PATH)


def prepare_features(df):
    """Prepare leakage-safe features and target for ML."""

    target = "Late_delivery_risk"

    # Convert order date to datetime
    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"],
        errors="coerce"
    )

    # Create useful calendar features
    df["order_year"] = df["order date (DateOrders)"].dt.year
    df["order_month"] = df["order date (DateOrders)"].dt.month
    df["order_day_of_week"] = (
        df["order date (DateOrders)"].dt.dayofweek
    )
    df["order_hour"] = df["order date (DateOrders)"].dt.hour

    # Columns that describe the delivery outcome
    # and must not be used as predictive features.
    leakage_columns = [
        "Late_delivery_risk",
        "Days for shipping (real)",
        "delivery_delay_days",
        "Delivery Status",
        "shipping date (DateOrders)",
        "order_to_shipping_days"
    ]

    # Identifier and raw date are not directly useful as model features.
    columns_to_remove = [
        "Order Id",
        "order date (DateOrders)"
    ]

    columns_to_remove.extend(leakage_columns)

    X = df.drop(columns=columns_to_remove)
    y = df[target]

    return X, y


def show_feature_information(X, y):
    """Display information about ML features and target."""

    print("\n========== ML FEATURE INFORMATION ==========")
    print(f"Number of records: {X.shape[0]}")
    print(f"Number of features: {X.shape[1]}")

    print("\n========== FEATURES ==========")
    for column in X.columns:
        print(f"- {column}")

    print("\n========== TARGET ==========")
    print(y.value_counts())

    print("\n========== TARGET PERCENTAGE ==========")
    print(y.value_counts(normalize=True) * 100)

    print("\n========== MISSING VALUES ==========")
    print(X.isnull().sum().sum())


if __name__ == "__main__":

    print("Loading processed dataset...")

    df = load_data()

    X, y = prepare_features(df)

    show_feature_information(X, y)