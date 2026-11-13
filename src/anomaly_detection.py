import pandas as pd
import numpy as np

from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from feature_engineering import load_data


DATA_PATH = "data/processed/processed_data.csv"


# Expected proportion of unusual records.
# 0.05 means approximately 5% of records are treated as anomalies.
CONTAMINATION = 0.05


def prepare_anomaly_features(df):
    """
    Prepare numerical features for anomaly detection.
    """

    selected_features = [
        "Days for shipment (scheduled)",
        "Benefit per order",
        "Sales per customer",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
        "Order Profit Per Order",
        "Order Item Product Price",
        "Order Item Profit Ratio",
        "Product Price",
        "order date (DateOrders)",
    ]

    data = df[selected_features].copy()

    # Convert order date into useful numerical features
    data["order date (DateOrders)"] = pd.to_datetime(
        data["order date (DateOrders)"],
        errors="coerce"
    )

    data["order_hour"] = (
        data["order date (DateOrders)"].dt.hour
    )

    data["order_day_of_week"] = (
        data["order date (DateOrders)"].dt.day_of_week
    )

    data["order_month"] = (
        data["order date (DateOrders)"].dt.month
    )

    # Remove original datetime column
    data = data.drop(
        columns=["order date (DateOrders)"]
    )

    # Replace infinite values
    data = data.replace(
        [np.inf, -np.inf],
        np.nan
    )

    # Remove rows with missing anomaly features
    data = data.dropna()

    return data


def train_anomaly_model(X):
    """
    Train Isolation Forest anomaly detector.
    """

    scaler = StandardScaler()

    X_scaled = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=100,
        contamination=CONTAMINATION,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_scaled)

    return model, scaler


def calculate_anomaly_results(
    model,
    scaler,
    X
):
    """
    Calculate anomaly labels and scores.
    """

    X_scaled = scaler.transform(X)

    predictions = model.predict(X_scaled)

    raw_scores = model.decision_function(
        X_scaled
    )

    # Convert Isolation Forest output:
    # -1 = anomaly
    #  1 = normal
    anomaly_labels = np.where(
        predictions == -1,
        1,
        0
    )

    # Higher score means more normal.
    # Reverse it so higher anomaly score means
    # more anomalous.
    anomaly_scores = -raw_scores

    return anomaly_labels, anomaly_scores


if __name__ == "__main__":

    print("Loading processed dataset...")

    df = load_data()

    print(
        f"Records loaded: {len(df)}"
    )

    print("\nPreparing anomaly features...")

    X = prepare_anomaly_features(df)

    print(
        f"Anomaly records: {len(X)}"
    )

    print(
        f"Anomaly features: {X.shape[1]}"
    )

    print("\nTraining Isolation Forest...")

    model, scaler = train_anomaly_model(X)

    print(
        "Isolation Forest training completed."
    )

    print("\nCalculating anomaly scores...")

    anomaly_labels, anomaly_scores = (
        calculate_anomaly_results(
            model,
            scaler,
            X
        )
    )

    results = X.copy()

    results["anomaly_label"] = (
        anomaly_labels
    )

    results["anomaly_score"] = (
        anomaly_scores
    )

    print("\n========== ANOMALY DISTRIBUTION ==========")

    print(
        results["anomaly_label"]
        .value_counts()
        .sort_index()
    )

    anomaly_percentage = (
        results["anomaly_label"].mean()
        * 100
    )

    print(
        f"\nAnomaly percentage: "
        f"{anomaly_percentage:.2f}%"
    )

    print(
        "\n========== ANOMALY SCORE SUMMARY =========="
    )

    print(
        results["anomaly_score"]
        .describe()
    )

    print(
        "\n========== TOP ANOMALOUS RECORDS =========="
    )

    print(
        results
        .sort_values(
            "anomaly_score",
            ascending=False
        )
        .head(10)
        .to_string(index=False)
    )
    ANOMALY_RESULTS_PATH = (
    "data/processed/anomaly_results.csv"
)
    results["anomaly_score"] = (
    anomaly_scores
)
    results.to_csv(
    ANOMALY_RESULTS_PATH,
    index=False
)

print(
    f"\nAnomaly results saved to: "
    f"{ANOMALY_RESULTS_PATH}"
)
