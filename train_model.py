"""Train and persist the Telco churn pipeline used by app.py."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_PATH = PROJECT_DIR / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = PROJECT_DIR / "churn_model.pkl"
TARGET = "Churn"
ID_COLUMN = "customerID"
BUSINESS_RISK_THRESHOLD = 0.65


def load_training_data(path: Path = DATA_PATH) -> tuple[pd.DataFrame, pd.Series]:
    """Load raw Telco data and normalize TotalCharges' blank strings."""
    frame = pd.read_csv(path)
    if TARGET not in frame or ID_COLUMN not in frame:
        raise ValueError(f"Expected columns {TARGET!r} and {ID_COLUMN!r} in {path}")

    # The source file stores blank TotalCharges for some zero-tenure customers.
    # Converting to numeric makes those blanks missing values, which the pipeline
    # imputes rather than crashing during model fitting.
    frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    target = frame.pop(TARGET).map({"No": 0, "Yes": 1})
    features = frame.drop(columns=[ID_COLUMN])
    if target.isna().any():
        raise ValueError("Unexpected target value in Churn; expected only Yes/No.")
    return features, target.astype("int8")


def build_pipeline(features: pd.DataFrame) -> Pipeline:
    numeric_columns = features.select_dtypes(include=["number"]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_columns),
            ("categorical", categorical_pipeline, categorical_columns),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessing", preprocessing),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1500,
                    random_state=42,
                ),
            ),
        ]
    )


def main() -> None:
    features, target = load_training_data()
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        target,
        test_size=0.2,
        stratify=target,
        random_state=42,
    )

    model = build_pipeline(features)
    model.fit(x_train, y_train)
    churn_probability = model.predict_proba(x_valid)[:, 1]
    print(f"Validation ROC AUC: {roc_auc_score(y_valid, churn_probability):.3f}")
    print(f"Validation PR AUC:  {average_precision_score(y_valid, churn_probability):.3f}")
    print(f"Validation Brier score: {brier_score_loss(y_valid, churn_probability):.3f}")

    # Preserve the holdout metrics above, then use every labeled customer for
    # the final artifact that will serve predictions in the app.
    model.fit(features, target)

    # Keep the fitted preprocessing inside the trusted local artifact so the app
    # receives the same columns and transformations used during training.
    artifact = {
        "model": model,
        "feature_columns": features.columns.tolist(),
        "threshold": BUSINESS_RISK_THRESHOLD,
        "validation_rows": len(y_valid),
        "validation_churn_rate": float(np.mean(y_valid)),
    }
    with MODEL_PATH.open("wb") as model_file:
        pickle.dump(artifact, model_file, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()
