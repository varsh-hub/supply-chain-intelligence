
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report
)

from feature_engineering import prepare_features


DATA_PATH = "data/processed/processed_data.csv"
MODEL_PATH = "models/risk_model.pkl"


print("Loading processed dataset...")

df = pd.read_csv(DATA_PATH)

print(f"Records: {df.shape[0]}")
print(f"Columns: {df.shape[1]}")


print("\nPreparing predictive features...")

X, y = prepare_features(df)

print(f"Features: {X.shape[1]}")
print(f"Target records: {y.shape[0]}")


# Identify categorical and numerical columns
categorical_features = X.select_dtypes(
    include=["object", "string"]
).columns.tolist()

numerical_features = X.select_dtypes(
    exclude=["object", "string"]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical features:")
print(numerical_features)


# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features
        ),
        (
            "numerical",
            "passthrough",
            numerical_features
        )
    ]
)


# Random Forest model
model = RandomForestClassifier(
    n_estimators=50,
    random_state=42,
    n_jobs=-1
)


# Complete ML pipeline
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)


print("\n========== TRAINING ==========")

print(f"Training records: {X_train.shape[0]}")
print(f"Testing records: {X_test.shape[0]}")


# Train model
pipeline.fit(X_train, y_train)


# Predictions
y_pred = pipeline.predict(X_test)


# Evaluation
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)


print("\n========== MODEL PERFORMANCE ==========")

print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")


print("\n========== CLASSIFICATION REPORT ==========")

print(
    classification_report(
        y_test,
        y_pred
    )
)


# Save complete pipeline
joblib.dump(
    pipeline,
    MODEL_PATH
)


print("\n========== MODEL SAVING ==========")

print(f"Model saved successfully.")
print(f"Location: {MODEL_PATH}")
