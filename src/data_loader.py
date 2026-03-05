from __future__ import annotations

import pandas as pd
from src.config import DATA_PATH
from src.logger import setup_logger

logger = setup_logger()


def load_data(path=DATA_PATH) -> pd.DataFrame:
    """
    Load the housing dataset from CSV and perform basic validation.

    Returns:
        pd.DataFrame: Loaded dataset.

    Raises:
        FileNotFoundError: If the CSV file does not exist.
        ValueError: If the file is empty or required columns are missing.
    """
    try:
        logger.info(f"Loading dataset from: {path}")

        if not path.exists():
            logger.error(f"Dataset not found at: {path}")
            raise FileNotFoundError(f"Dataset not found at: {path}")

        df = pd.read_csv(path)

        if df.empty:
            logger.error("Dataset loaded but is empty.")
            raise ValueError("Dataset is empty.")

        # Minimum required columns for this project
        required_cols = {"price"}
        missing = required_cols - set(df.columns)

        if missing:
            logger.error(f"Missing required columns: {missing}")
            raise ValueError(f"Missing required columns: {missing}")

        logger.info(f"Dataset loaded successfully. Shape: {df.shape}")
        return df

    except Exception as e:
        logger.exception(f"Failed to load dataset: {e}")
        raise