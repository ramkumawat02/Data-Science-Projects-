"""Feature selection and engineering module for ICU mortality prediction.

Handles feature extraction, selection, and validation with comprehensive
documentation and error handling.
"""

import logging

import pandas as pd

from src import config
from src.exceptions import DataValidationError
from src.validators import validate_dataframe, check_missing_columns

logger = logging.getLogger(__name__)

__all__ = ["select_features", "get_feature_info"]


def get_feature_info(features: list[str]) -> dict:
    """
    Get information about selected features.

    Args:
        features: List of feature names.

    Returns:
        Dictionary with feature statistics and metadata.
    """
    return {
        "count": len(features),
        "names": features,
        "feature_set": "medical_vitals",
        "version": "1.0",
    }


def select_features(
    df: pd.DataFrame,
    features: list[str] | None = None,
    target: str | None = None,
    validate_types: bool = True,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Select features and target from DataFrame with comprehensive validation.

    Extracts specified features and target column from the input DataFrame,
    with optional type checking and validation.

    Args:
        df: Input DataFrame containing all data.
        features: List of feature column names. Uses config.DEFAULT_FEATURES if None.
        target: Target column name. Uses config.DEFAULT_TARGET if None.
        validate_types: If True, validate data types are numeric (default: True).

    Returns:
        Tuple of (X_features, y_target) where:
        - X_features: DataFrame with selected features
        - y_target: Series with target values

    Raises:
        DataValidationError: If DataFrame is invalid or columns are missing.
        ValueError: If feature selection fails.

   
    """
    # Validate input DataFrame
    validate_dataframe(df, "Input DataFrame")

    # Set defaults from config
    features = features or config.DEFAULT_FEATURES
    target = target or config.DEFAULT_TARGET

    try:
        # Check that all required features exist
        check_missing_columns(df, features)

        # Check that target column exists
        if target not in df.columns:
            raise DataValidationError(
                f"Target column '{target}' not found in DataFrame. "
                f"Available columns: {list(df.columns)}"
            )

        # Extract features and target
        X = df[features].copy()
        y = df[target].copy()

        # Optional type validation
        if validate_types:
            non_numeric_features = X.select_dtypes(exclude=["number"]).columns.tolist()
            if non_numeric_features:
                logger.warning(
                    f"Non-numeric features detected: {non_numeric_features}. "
                    f"Consider encoding categorical variables."
                )

        # Log feature selection
        feature_info = get_feature_info(features)
        logger.info(
            f"Selected {feature_info['count']} features and target '{target}' "
            f"from DataFrame with shape {df.shape}"
        )

        return X, y

    except DataValidationError:
        raise
    except Exception as e:
        logger.error(f"Error selecting features: {e}")
        raise DataValidationError(f"Failed to select features: {e}") from e