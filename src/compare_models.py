import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
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


def build_preprocessor(
    numerical_features,
    categorical_features,
    scale_numeric=False
):
    """Build feature preprocessing pipeline."""

    numerical_transformer = "passthrough"

    if scale_numeric:
        numerical_transformer = StandardScaler()

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
                numerical_transformer,
                numerical_features
            )
        ]
    )

    return preprocessor


def build_random_forest(
    numerical_features,
    categorical_features
):
    """Build Random Forest pipeline."""

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features
    )

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        n_jobs=-1
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )


def build_logistic_regression(
    numerical_features,
    categorical_features
):
    """Build Logistic Regression pipeline."""

    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
        scale_numeric=True
    )

    classifier = LogisticRegression(
        max_iter=1000,
        random_state=42
    )

    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", classifier)
        ]
    )


def evaluate_model(model, X_test, y_test):
    """Evaluate model performance."""

    predictions = model.predict(X_test)

    probabilities = model.predict_proba(
        X_test
    )[:, 1]

    return {
        "Accuracy": accuracy_score(
            y_test,
            predictions
        ),
        "Precision": precision_score(
            y_test,
            predictions
        ),
        "Recall": recall_score(
            y_test,
            predictions
        ),
        "F1 Score": f1_score(
            y_test,
            predictions
        ),
        "ROC AUC": roc_auc_score(
            y_test,
            probabilities
        )
    }


if __name__ == "__main__":

    print("Loading processed dataset...")

    df = load_data()

    print("Preparing ML features...")

    X, y = prepare_features(df)

    numerical_features, categorical_features = (
        identify_feature_types(X)
    )

    print("\n========== FEATURE INFORMATION ==========")
    print(
        f"Numerical features: {len(numerical_features)}"
    )
    print(
        f"Categorical features: {len(categorical_features)}"
    )

    print("\n========== TRAIN / TEST SPLIT ==========")

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    models = {
        "Random Forest": build_random_forest(
            numerical_features,
            categorical_features
        ),
        "Logistic Regression": build_logistic_regression(
            numerical_features,
            categorical_features
        )
    }

    results = []

    for name, model in models.items():

        print(f"\nTraining {name}...")

        model.fit(
            X_train,
            y_train
        )

        metrics = evaluate_model(
            model,
            X_test,
            y_test
        )

        metrics["Model"] = name

        results.append(metrics)

    results_df = pd.DataFrame(results)

    results_df = results_df[
        [
            "Model",
            "Accuracy",
            "Precision",
            "Recall",
            "F1 Score",
            "ROC AUC"
        ]
    ]

    print("\n========== MODEL COMPARISON ==========")

    print(
        results_df.to_string(
            index=False,
            formatters={
                "Accuracy": "{:.4f}".format,
                "Precision": "{:.4f}".format,
                "Recall": "{:.4f}".format,
                "F1 Score": "{:.4f}".format,
                "ROC AUC": "{:.4f}".format
            }
        )
    )