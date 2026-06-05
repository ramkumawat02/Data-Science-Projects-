"""Data validation utilities for ICU mortality prediction."""

import logging
from typing import Callable

import pandas as pd
import numpy as np

from src.exceptions import DataValidationError
from src import config

logger = logging.getLogger(__name__)

__all__ = [
    "validate_dataframe",
    "validate_file_path",
    "validate_numeric_range",
    "check_missing_columns",
    "check_data_quality",
]


def validate_dataframe(df: pd.DataFrame, name: str = "DataFrame") -> None:
    """
    Validate that input is a proper DataFrame.

    Args:
        df: DataFrame to validate.
        name: Name of the DataFrame for error messages.

    Raises:
        DataValidationError: If validation fails.
    """
    if df is None:
        raise DataValidationError(f"{name} cannot be None")

    if not isinstance(df, pd.DataFrame):
        raise DataValidationError(f"Expected pd.DataFrame, got {type(df).__name__}")

    if df.empty:
        raise DataValidationError(f"{name} is empty")


def validate_file_path(path: str | object) -> None:
    """
    Validate that file path exists and is readable.

    Args:
        path: File path to validate.

    Raises:
        DataValidationError: If file doesn't exist or isn't readable.
    """
    from pathlib import Path

    try:
        p = Path(path)
        if not p.exists():
            raise DataValidationError(f"File not found: {p}")

        if not p.is_file():
            raise DataValidationError(f"Path is not a file: {p}")

    except DataValidationError:
        raise
    except Exception as e:
        raise DataValidationError(f"Error validating file path: {e}") from e


def validate_numeric_range(value: float | int, min_val: float, max_val: float, name: str) -> None:
    """
    Validate that numeric value is within expected range.

    Args:
        value: Value to validate.
        min_val: Minimum allowed value (inclusive).
        max_val: Maximum allowed value (inclusive).
        name: Name of the value for error messages.

    Raises:
        DataValidationError: If value is out of range.
    """
    if not min_val <= value <= max_val:
        raise DataValidationError(
            f"{name} must be between {min_val} and {max_val}, got {value}"
        )


def check_missing_columns(df: pd.DataFrame, required_columns: list[str]) -> list[str]:
    """
    Check for missing required columns.

    Args:
        df: DataFrame to check.
        required_columns: List of required column names.

    Returns:
        List of missing column names (empty if all present).

    Raises:
        DataValidationError: If critical columns are missing.
    """
    missing = [col for col in required_columns if col not in df.columns]

    if missing:
        logger.error(f"Missing required columns: {missing}")
        raise DataValidationError(f"Missing required columns: {missing}")

    return missing


def check_data_quality(df: pd.DataFrame, max_missing_ratio: float = 0.5) -> dict:
    """
    Check overall data quality and return quality metrics.

    Args:
        df: DataFrame to check.
        max_missing_ratio: Maximum allowed ratio of missing values (0-1).

    Returns:
        Dictionary with quality metrics.

    Raises:
        DataValidationError: If data quality is critically poor.
    """
    validate_dataframe(df)

    metrics = {
        "total_samples": len(df),
        "total_columns": len(df.columns),
        "total_missing": df.isnull().sum().sum(),
        "missing_ratio": df.isnull().sum().sum() / (len(df) * len(df.columns)),
        "duplicate_rows": df.duplicated().sum(),
        "columns_with_missing": df.columns[df.isnull().any()].tolist(),
    }

    if metrics["missing_ratio"] > max_missing_ratio:
        raise DataValidationError(
            f"Data quality too poor: {metrics['missing_ratio']:.1%} missing values"
        )

    logger.info(f"Data quality check: {metrics['missing_ratio']:.1%} missing values")
    return metrics
