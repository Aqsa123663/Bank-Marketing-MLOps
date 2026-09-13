# ============================================
# BANK MARKETING MLOPS PROJECT
# Model Training + MLflow Tracking
# ============================================

import os
import pandas as pd
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

import joblib


# ============================================
# 1. CREATE REQUIRED FOLDERS
# ============================================

os.makedirs("models", exist_ok=True)


# ============================================
# 2. LOAD DATASET
# ============================================

DATA_PATH = "data/bank.csv"

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, sep=";")


# ============================================
# 3. DISPLAY DATASET INFORMATION
# ============================================

print("\nDataset Shape:")
print(df.shape)

print("\nFirst 5 Rows:")
print(df.head())

print("\nDataset Information:")
print(df.info())


# ============================================
# 4. REMOVE DUPLICATES
# ============================================

df = df.drop_duplicates()

print("\nDataset Shape After Removing Duplicates:")
print(df.shape)


# ============================================
# 5. DEFINE FEATURES AND TARGET
# ============================================

TARGET = "y"

X = df.drop(TARGET, axis=1)
y = df[TARGET]


# ============================================
# 6. TRAIN TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

print("\nTraining Data Shape:", X_train.shape)
print("Testing Data Shape:", X_test.shape)


# ============================================
# 7. IDENTIFY NUMERICAL AND CATEGORICAL COLUMNS
# ============================================

numerical_columns = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()


print("\nNumerical Columns:")
print(numerical_columns)

print("\nCategorical Columns:")
print(categorical_columns)


# ============================================
# 8. PREPROCESSING
# ============================================

numerical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)


categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)


preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            numerical_transformer,
            numerical_columns
        ),
        (
            "cat",
            categorical_transformer,
            categorical_columns
        )
    ]
)


# ============================================
# 9. MLFLOW SETUP
# ============================================

mlflow.set_experiment(
    "Bank_Marketing_Classification"
)


# ============================================
# 10. FUNCTION FOR MODEL TRAINING
# ============================================

def train_and_evaluate(
    model,
    model_name,
    parameters
):

    print("\n" + "=" * 50)
    print(f"Training {model_name}")
    print("=" * 50)

    # Create ML Pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    # Start MLflow Run
    with mlflow.start_run(
        run_name=model_name
    ):

        # Train Model
        pipeline.fit(
            X_train,
            y_train
        )

        # Predictions
        predictions = pipeline.predict(
            X_test
        )

        # Metrics
        accuracy = accuracy_score(
            y_test,
            predictions
        )

        precision = precision_score(
            y_test,
            predictions,
            pos_label="yes",
            zero_division=0
        )

        recall = recall_score(
            y_test,
            predictions,
            pos_label="yes",
            zero_division=0
        )

        f1 = f1_score(
            y_test,
            predictions,
            pos_label="yes",
            zero_division=0
        )

        # Print Metrics
        print(f"\nModel: {model_name}")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1:.4f}")

        # Log Parameters
        mlflow.log_params(
            parameters
        )

        # Log Metrics
        mlflow.log_metric(
            "accuracy",
            accuracy
        )

        mlflow.log_metric(
            "precision",
            precision
        )

        mlflow.log_metric(
            "recall",
            recall
        )

        mlflow.log_metric(
            "f1_score",
            f1
        )

        # Log Model
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model"
        )

        return pipeline, accuracy


# ============================================
# 11. LOGISTIC REGRESSION
# ============================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_parameters = {
    "model_type": "Logistic Regression",
    "max_iter": 1000,
    "random_state": 42
}


logistic_pipeline, logistic_accuracy = train_and_evaluate(
    logistic_model,
    "Logistic_Regression",
    logistic_parameters
)


# ============================================
# 12. RANDOM FOREST
# ============================================

random_forest_model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)

random_forest_parameters = {
    "model_type": "Random Forest",
    "n_estimators": 100,
    "random_state": 42
}


rf_pipeline, rf_accuracy = train_and_evaluate(
    random_forest_model,
    "Random_Forest",
    random_forest_parameters
)


# ============================================
# 13. SELECT BEST MODEL
# ============================================

if logistic_accuracy >= rf_accuracy:

    best_model = logistic_pipeline
    best_model_name = "Logistic Regression"
    best_accuracy = logistic_accuracy

else:

    best_model = rf_pipeline
    best_model_name = "Random Forest"
    best_accuracy = rf_accuracy


# ============================================
# 14. SAVE BEST MODEL
# ============================================

MODEL_PATH = "models/best_model.pkl"

joblib.dump(
    best_model,
    MODEL_PATH
)


# ============================================
# 15. FINAL RESULTS
# ============================================

print("\n" + "=" * 50)
print("TRAINING COMPLETED")
print("=" * 50)

print(f"\nBest Model: {best_model_name}")

print(
    f"Best Accuracy: {best_accuracy:.4f}"
)

print(
    f"\nModel saved successfully at: {MODEL_PATH}"
)