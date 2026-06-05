"""Dataset splitting module for ICU mortality prediction.

Provides robust data splitting with stratification and comprehensive
validation for reproducible model training.
"""

import logging

from sklearn.model_selection import train_test_split
import pandas as pd

from src import config
from src.exceptions import DataValidationError
from src.validators import validate_dataframe, validate_numeric_range

logger = logging.getLogger(__name__)

__all__ = ["split_data"]


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = config.DEFAULT_TEST_SIZE,
    random_state: int = config.DEFAULT_RANDOM_STATE,
    stratify: bool = config.DEFAULT_STRATIFY,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """
    Split data into train and test sets with optional stratification.

    Args:
        X: Feature DataFrame with shape (n_samples, n_features).
        y: Target Series with shape (n_samples,).
        test_size: Proportion of data for testing, 0-1 (default: 0.2).
        random_state: Random seed for reproducibility (default: 42).
        stratify: Whether to stratify split by target (default: True).

    Returns:
        Tuple of (X_train, X_test, y_train, y_test) with:
        - X_train: Training features
        - X_test: Testing features
        - y_train: Training target
        - y_test: Testing target

    Raises:
        DataValidationError: If inputs are invalid or incompatible.
    """
    # Validate inputs
    validate_dataframe(X, "Features DataFrame X")
    validate_dataframe(y.to_frame() if isinstance(y, pd.Series) else y, "Target y")

    if len(X) != len(y):
        raise DataValidationError(
            f"Data mismatch: X has {len(X):,} samples, "
            f"y has {len(y):,} samples"
        )

    # Validate test_size range
    validate_numeric_range(test_size, 0.0, 1.0, "test_size")

    try:
        stratify_arg = y if stratify else None

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify_arg,
        )

        # Log split statistics
        train_ratio = len(X_train) / len(X)
        test_ratio = len(X_test) / len(X)

        logger.info(
            f"Data split complete:\n"
            f"  Train: {len(X_train):,} samples ({train_ratio:.1%})\n"
            f"  Test: {len(X_test):,} samples ({test_ratio:.1%})\n"
            f"  Features: {X_train.shape[1]}"
        )

        if stratify:
            logger.debug(
                f"Stratified split maintained class distribution: "
                f"Train minority: {(y_train == 1).sum() / len(y_train):.1%}, "
                f"Test minority: {(y_test == 1).sum() / len(y_test):.1%}"
            )

        return X_train, X_test, y_train, y_test

    except DataValidationError:
        raise
    except Exception as e:
        logger.error(f"Error during data split: {e}")
        raise DataValidationError(f"Failed to split data: {e}") from e