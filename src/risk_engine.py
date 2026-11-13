import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score
)

from feature_engineering import load_data, prepare_features


DATA_PATH = "data/processed/processed_data.csv"


def identify_feature_types(X):
    """Identify numerical and categorical features."""

    categorical_features = X.select_dtypes(
        include=["object", "string"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object", "string"]
    ).columns.tolist()

    return numerical_features, categorical_features


def build_model(numerical_features, categorical_features):
    """Build the Random Forest prediction pipeline."""

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(
                    handle_unknown="ignore"
                ),
                categorical_features
            ),
            (
                "numerical",
                "passthrough",
                numerical_features
            )
        ]
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )

    return model


def evaluate_thresholds(
    model,
    X_test,
    y_test
):
    """Evaluate different probability thresholds."""

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    thresholds = [
        0.30,
        0.35,
        0.40,
        0.45,
        0.50,
        0.55,
        0.60,
        0.65,
        0.70
    ]

    results = []

    for threshold in thresholds:

        predictions = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_test,
            predictions
        )

        recall = recall_score(
            y_test,
            predictions
        )

        f1 = f1_score(
            y_test,
            predictions
        )

        results.append({
            "Threshold": threshold,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1
        })

    results_df = pd.DataFrame(results)

    print("\n========== THRESHOLD ANALYSIS ==========")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Threshold": "{:.2f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format
            }
        )
    )

    return results_df


def classify_risk(probability):
    """
    Convert probability into a risk category.

    Temporary thresholds:
    < 0.40  -> LOW
    0.40-0.69 -> MEDIUM
    >= 0.70 -> HIGH
    """

    if probability < 0.40:
        return "LOW"

    elif probability < 0.70:
        return "MEDIUM"

    else:
        return "HIGH"


def create_risk_output(
    X_test,
    y_test,
    probabilities
):
    """Create a risk output table."""

    risk_levels = [
        classify_risk(probability)
        for probability in probabilities
    ]

    risk_output = X_test.copy()

    risk_output["Actual Risk"] = y_test.values

    risk_output["Risk Probability"] = probabilities

    risk_output["Risk Level"] = risk_levels

    return risk_output


if __name__ == "__main__":

    print("Loading processed dataset...")

    df = load_data()

    print("Preparing ML features...")

    X, y = prepare_features(df)

    numerical_features, categorical_features = (
        identify_feature_types(X)
    )

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nBuilding Random Forest model...")

    model = build_model(
        numerical_features,
        categorical_features
    )

    print("\nTraining model...")

    model.fit(
        X_train,
        y_train
    )

    print("Model training completed.")

    # Evaluate probability thresholds
    results = evaluate_thresholds(
        model,
        X_test,
        y_test
    )

    # Use current temporary thresholds
    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    risk_output = create_risk_output(
        X_test,
        y_test,
        probabilities
    )

    print("\n========== RISK DISTRIBUTION ==========")

    print(
        risk_output["Risk Level"]
        .value_counts()
    )

    print("\n========== SAMPLE RISK PREDICTIONS ==========")

    print(
        risk_output[
            [
                "Actual Risk",
                "Risk Probability",
                "Risk Level"
            ]
        ]
        .head(10)
        .to_string(index=False)
    )