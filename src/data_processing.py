import pandas as pd


DATA_PATH = "data/raw/supply_chain_data.csv"


def load_data(file_path):
    """Load the raw supply-chain dataset."""
    return pd.read_csv(file_path, encoding="cp1252")


def inspect_data(df):
    """Display basic information about the dataset."""

    print("\n========== DATASET SHAPE ==========")
    print(df.shape)

    print("\n========== COLUMNS ==========")
    print(df.columns.tolist())

    print("\n========== FIRST 5 ROWS ==========")
    print(df.head())

    print("\n========== DATA TYPES ==========")
    print(df.dtypes)

    print("\n========== MISSING VALUES ==========")
    print(df.isnull().sum())

    print("\n========== DUPLICATE ROWS ==========")
    print(df.duplicated().sum())


if __name__ == "__main__":

    df = load_data(DATA_PATH)

    inspect_data(df)