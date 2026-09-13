import os
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

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
    classification_report,
)


# ==========================================
# 1. File Paths
# ==========================================

DATA_PATH = "data/bank.csv"
MODEL_PATH = "models/bank_marketing_model.pkl"


# ==========================================
# 2. Load Dataset
# ==========================================

print("Loading dataset...")

df = pd.read_csv(DATA_PATH, sep=";")

print("Dataset loaded successfully!")
print("Dataset shape:", df.shape)

print("\nFirst 5 rows:")
print(df.head())


# ==========================================
# 3. Data Cleaning
# ==========================================

print("\nCleaning data...")

# Remove duplicate rows
df = df.drop_duplicates()

# Convert target column
df["y"] = df["y"].map({
    "yes": 1,
    "no": 0
})

# Remove rows where target is missing
df = df.dropna(subset=["y"])

print("Dataset after cleaning:", df.shape)


# ==========================================
# 4. Separate Features and Target
# ==========================================

X = df.drop("y", axis=1)
y = df["y"]


# ==========================================
# 5. Identify Categorical Columns
# ==========================================

categorical_columns = X.select_dtypes(
    include=["object"]
).columns.tolist()

numeric_columns = X.select_dtypes(
    exclude=["object"]
).columns.tolist()

print("\nCategorical columns:")
print(categorical_columns)

print("\nNumeric columns:")
print(numeric_columns)


# ==========================================
# 6. Preprocessing
# ==========================================

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_columns
        )
    ],
    remainder="passthrough"
)


# ==========================================
# 7. Machine Learning Model
# ==========================================

model = RandomForestClassifier(
    n_estimators=100,
    random_state=42,
    class_weight="balanced"
)


# ==========================================
# 8. Create ML Pipeline
# ==========================================

pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", model)
    ]
)


# ==========================================
# 9. Train/Test Split
# ==========================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTraining samples:", len(X_train))
print("Testing samples:", len(X_test))


# ==========================================
# 10. MLflow Experiment
# ==========================================

mlflow.set_experiment("Bank Marketing Classification")


with mlflow.start_run():

    print("\nTraining model...")

    pipeline.fit(X_train, y_train)

    print("Training completed!")


    # ======================================
    # 11. Predictions
    # ======================================

    y_pred = pipeline.predict(X_test)


    # ======================================
    # 12. Evaluation Metrics
    # ======================================

    accuracy = accuracy_score(y_test, y_pred)

    precision = precision_score(
        y_test,
        y_pred,
        zero_division=0
    )

    recall = recall_score(
        y_test,
        y_pred,
        zero_division=0
    )

    f1 = f1_score(
        y_test,
        y_pred,
        zero_division=0
    )


    # ======================================
    # 13. Display Results
    # ======================================

    print("\n==============================")
    print("MODEL RESULTS")
    print("==============================")

    print("Accuracy :", round(accuracy, 4))
    print("Precision:", round(precision, 4))
    print("Recall   :", round(recall, 4))
    print("F1 Score :", round(f1, 4))

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            y_pred,
            zero_division=0
        )
    )


    # ======================================
    # 14. Log Parameters in MLflow
    # ======================================

    mlflow.log_param(
        "model",
        "RandomForestClassifier"
    )

    mlflow.log_param(
        "n_estimators",
        100
    )

    mlflow.log_param(
        "test_size",
        0.20
    )


    # ======================================
    # 15. Log Metrics in MLflow
    # ======================================

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


    # ======================================
    # 16. Save Model in MLflow
    # ======================================

    mlflow.sklearn.log_model(
        pipeline,
        "bank_marketing_model"
    )


# ==========================================
# 17. Save Model Locally
# ==========================================

os.makedirs(
    "models",
    exist_ok=True
)

joblib.dump(
    pipeline,
    MODEL_PATH
)

print("\n==============================")
print("SUCCESS")
print("==============================")

print("Model saved at:")
print(MODEL_PATH)