import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    ConfusionMatrixDisplay,
    classification_report
)

from feature_engineering import load_data, prepare_features


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
    """Build preprocessing and Random Forest pipeline."""

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


def show_confusion_matrix(model, X_test, y_test):
    """Display confusion matrix."""

    predictions = model.predict(X_test)

    cm = confusion_matrix(
        y_test,
        predictions
    )

    print("\n========== CONFUSION MATRIX ==========")
    print(cm)

    print("\nMatrix format:")
    print("[[True Negative, False Positive]")
    print(" [False Negative, True Positive]]")

    display = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Not Late", "Late Risk"]
    )

    display.plot()
    plt.title("Late Delivery Risk - Confusion Matrix")
    plt.tight_layout()
    plt.show()


def show_feature_importance(model):
    """Display top model features."""

    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["model"]

    feature_names = (
        preprocessor
        .get_feature_names_out()
    )

    importances = classifier.feature_importances_

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importances
    })

    importance_df = (
        importance_df
        .sort_values(
            "Importance",
            ascending=False
        )
        .head(20)
    )

    print("\n========== TOP 20 FEATURES ==========")
    print(importance_df.to_string(index=False))

    plt.figure(figsize=(10, 8))

    plt.barh(
        importance_df["Feature"],
        importance_df["Importance"]
    )

    plt.gca().invert_yaxis()

    plt.title(
        "Top 20 Features Influencing Late Delivery Risk"
    )

    plt.xlabel("Feature Importance")
    plt.ylabel("Feature")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    print("Loading processed dataset...")

    df = load_data()

    print("Preparing ML features...")

    X, y = prepare_features(df)

    numerical_features, categorical_features = (
        identify_feature_types(X)
    )

    print("\n========== FEATURE TYPES ==========")
    print(
        f"Numerical features: {len(numerical_features)}"
    )

    print(
        f"Categorical features: {len(categorical_features)}"
    )

    print("\nSplitting dataset...")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nBuilding model...")

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

    predictions = model.predict(X_test)

    print("\n========== CLASSIFICATION REPORT ==========")

    print(
        classification_report(
            y_test,
            predictions
        )
    )

    show_confusion_matrix(
        model,
        X_test,
        y_test
    )

    show_feature_importance(
        model
    )