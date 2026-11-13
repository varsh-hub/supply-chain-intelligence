import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from feature_engineering import load_data, prepare_features


def identify_feature_types(X):
    """Identify numerical and categorical features."""

    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns.tolist()

    numerical_features = X.select_dtypes(
        exclude=["object"]
    ).columns.tolist()

    return numerical_features, categorical_features


def build_model(numerical_features, categorical_features):
    """Build preprocessing and Random Forest model pipeline."""

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

    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    return pipeline


def evaluate_model(model, X_test, y_test):
    """Evaluate the trained model."""

    predictions = model.predict(X_test)

    print("\n========== MODEL EVALUATION ==========")

    accuracy = accuracy_score(
        y_test,
        predictions
    )

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

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")

    print("\n========== CLASSIFICATION REPORT ==========")
    print(
        classification_report(
            y_test,
            predictions
        )
    )


if __name__ == "__main__":

    print("Loading processed dataset...")

    df = load_data()

    print("\nPreparing ML features...")

    X, y = prepare_features(df)

    print(f"Records: {X.shape[0]}")
    print(f"Features: {X.shape[1]}")

    numerical_features, categorical_features = (
        identify_feature_types(X)
    )

    print("\n========== FEATURE TYPES ==========")

    print(f"Numerical features: {len(numerical_features)}")
    print(f"Categorical features: {len(categorical_features)}")

    print("\n========== TRAIN / TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(f"Training records: {X_train.shape[0]}")
    print(f"Testing records:  {X_test.shape[0]}")

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

    evaluate_model(
        model,
        X_test,
        y_test
    )