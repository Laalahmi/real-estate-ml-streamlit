from datetime import datetime
import pandas as pd

from src.logger import setup_logger

logger = setup_logger()


def transform_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply feature engineering and cleaning steps to the dataset.

    Steps:
    - Convert Yes/No columns to 1/0
    - Create houseAge feature
    - Drop 'made' column

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset

    Returns
    -------
    pd.DataFrame
        Transformed dataset
    """

    try:
        logger.info("Starting feature transformation")

        df = df.copy()

        # Convert Yes/No columns to binary
        binary_cols = [
            "hasYard",
            "hasPool",
            "isNewBuilt",
            "hasStormProtector",
        ]

        for col in binary_cols:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.lower()
                    .map({"yes": 1, "no": 0})
                )

        # Create houseAge feature
        current_year = datetime.now().year

        if "made" in df.columns:
            df["houseAge"] = current_year - df["made"]

            # Drop original column
            df.drop(columns=["made"], inplace=True)

        logger.info("Feature transformation completed")

        return df

    except Exception as e:
        logger.exception(f"Feature transformation failed: {e}")
        raise