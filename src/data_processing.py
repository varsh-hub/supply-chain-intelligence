import pandas as pd

RAW_DATA_PATH = "data/raw/supply_chain_data.csv"
PROCESSED_DATA_PATH = "data/processed/processed_data.csv"


def load_data(file_path):
    """Load the raw supply-chain dataset."""
    return pd.read_csv(file_path, encoding="latin1")


def preprocess_data(df):
    """Clean and prepare data for analysis and machine learning."""

    # Select relevant columns
    selected_columns = [
        "Type",
        "Days for shipping (real)",
        "Days for shipment (scheduled)",
        "Benefit per order",
        "Sales per customer",
        "Delivery Status",
        "Late_delivery_risk",
        "Category Id",
        "Category Name",
        "Customer Segment",
        "Department Id",
        "Department Name",
        "Market",
        "Order Country",
        "Order Region",
        "Order State",
        "Order City",
        "order date (DateOrders)",
        "Order Id",
        "Order Item Quantity",
        "Sales",
        "Order Item Total",
        "Order Profit Per Order",
        "Order Item Product Price",
        "Order Item Profit Ratio",
        "Shipping Mode",
        "shipping date (DateOrders)",
        "Product Card Id",
        "Product Category Id",
        "Product Name",
        "Product Price",
        "Product Status"
    ]

    df = df[selected_columns].copy()

    # Convert date columns to datetime
    df["order date (DateOrders)"] = pd.to_datetime(
        df["order date (DateOrders)"],
        errors="coerce"
    )

    df["shipping date (DateOrders)"] = pd.to_datetime(
        df["shipping date (DateOrders)"],
        errors="coerce"
    )

    # Create delivery delay feature
    df["delivery_delay_days"] = (
        df["Days for shipping (real)"]
        - df["Days for shipment (scheduled)"]
    )

    # Create order processing time
    df["order_to_shipping_days"] = (
        df["shipping date (DateOrders)"]
        - df["order date (DateOrders)"]
    ).dt.total_seconds() / (24 * 60 * 60)

    # Remove rows with invalid dates
    df = df.dropna(
        subset=[
            "order date (DateOrders)",
            "shipping date (DateOrders)"
        ]
    )

    # Reset index after preprocessing
    df = df.reset_index(drop=True)

    return df


def save_processed_data(df, file_path):
    """Save the processed dataset."""
    df.to_csv(file_path, index=False)


if __name__ == "__main__":

    print("Loading raw dataset...")

    df = load_data(RAW_DATA_PATH)

    print("\n========== RAW DATA ==========")
    print(f"Rows: {df.shape[0]}")
    print(f"Columns: {df.shape[1]}")

    print("\nPreprocessing data...")

    processed_df = preprocess_data(df)

    print("\n========== PROCESSED DATA ==========")
    print(f"Rows: {processed_df.shape[0]}")
    print(f"Columns: {processed_df.shape[1]}")

    print("\n========== PROCESSED COLUMNS ==========")
    print(processed_df.columns.tolist())

    print("\n========== MISSING VALUES ==========")
    print(processed_df.isnull().sum())

    save_processed_data(
        processed_df,
        PROCESSED_DATA_PATH
    )

    print("\nProcessed dataset saved successfully.")
    print(f"Location: {PROCESSED_DATA_PATH}")