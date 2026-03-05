from __future__ import annotations

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

from src.config import MODEL_DIR, MODEL_PATH, TEST_SIZE, RANDOM_STATE
from src.data_loader import load_data
from src.features import transform_features
from src.logger import setup_logger

logger = setup_logger()


def train_and_save() -> dict:
    """
    Train the Real Estate price prediction model and save artifacts for deployment.

    Pipeline steps:
    1) Load data
    2) Feature engineering (boolean conversion, houseAge, drop made)
    3) Split train/test
    4) Fit scaler on X_train only (no leakage)
    5) Train Linear Regression
    6) Evaluate on test set
    7) Save model bundle (scaler + model + feature list)

    Returns:
        dict: evaluation metrics
    """
    try:
        logger.info("=== Training started ===")

        # 1) Load + 2) feature engineering
        df = load_data()
        df = transform_features(df)

        if "price" not in df.columns:
            raise ValueError("Target column 'price' is missing after transformations.")

        # Separate target and features
        X = df.drop(columns=["price"])
        y = df["price"]

        # Basic check: no non-numeric columns (after Yes/No mapping)
        non_numeric = X.select_dtypes(exclude=["number"]).columns.tolist()
        if non_numeric:
            raise ValueError(
                f"Non-numeric feature columns found: {non_numeric}. "
                "Ensure all categorical columns are converted."
            )

        # 3) Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
        )
        logger.info(f"Train/Test split done. Train: {X_train.shape}, Test: {X_test.shape}")

        # 4) Scale (fit on train only)
        scaler = MinMaxScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        # 5) Train
        model = LinearRegression()
        model.fit(X_train_scaled, y_train)
        logger.info("Model training completed (LinearRegression).")

        # 6) Evaluate
        y_pred = model.predict(X_test_scaled)
        rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
        mae = float(mean_absolute_error(y_test, y_pred))
        r2 = float(r2_score(y_test, y_pred))

        metrics = {"rmse": rmse, "mae": mae, "r2": r2}
        logger.info(f"Evaluation metrics: {metrics}")

        # 7) Save artifacts for Streamlit (model + scaler + feature order)
        MODEL_DIR.mkdir(exist_ok=True)

        bundle = {
            "model": model,
            "scaler": scaler,
            "feature_names": list(X.columns),
            "metrics": metrics,
        }

        joblib.dump(bundle, MODEL_PATH)
        logger.info(f"Saved model bundle to: {MODEL_PATH}")

        logger.info("=== Training finished successfully ===")
        return metrics

    except Exception as e:
        logger.exception(f"Training failed: {e}")
        raise


if __name__ == "__main__":
    metrics = train_and_save()
    print("Training complete. Metrics:")
    print(metrics)
    print(f"Saved model to: {MODEL_PATH}")