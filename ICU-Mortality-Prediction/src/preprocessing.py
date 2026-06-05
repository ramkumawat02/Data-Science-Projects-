"""Data preprocessing module for ICU mortality prediction.

Provides robust data preprocessing including imputation, scaling, and validation.
Handles both numeric and categorical data appropriately.
"""

import logging

import pandas as pd
import numpy as np

from src.exceptions import DataValidationError
from src.validators import validate_dataframe, check_data_quality

logger = logging.getLogger(__name__)

__all__ = ["preprocess_data", "impute_missing_values"]


def impute_missing_values(
    df: pd.DataFrame, strategy: str = "mean", numeric_only: bool = True
) -> pd.DataFrame:
    """
    Impute missing values in DataFrame.

    Args:
        df: Input DataFrame with potential missing values.
        strategy: Imputation strategy - 'mean', 'median', 'forward_fill' (default: 'mean').
        numeric_only: If True, only impute numeric columns (default: True).

    Returns:
        pd.DataFrame: DataFrame with imputed values.

    Raises:
        DataValidationError: If strategy is invalid.
    """
    if strategy not in ("mean", "median", "forward_fill"):
        raise DataValidationError(f"Invalid strategy: {strategy}. Must be 'mean', 'median', or 'forward_fill'")

    df_imputed = df.copy()

    # Get numeric columns if numeric_only is True
    cols_to_impute = (
        df_imputed.select_dtypes(include=["number"]).columns if numeric_only
        else df_imputed.columns
    )

    for col in cols_to_impute:
        if df_imputed[col].isnull().any():
            if strategy == "mean":
                fill_value = df_imputed[col].mean()
            elif strategy == "median":
                fill_value = df_imputed[col].median()
            elif strategy == "forward_fill":
                df_imputed[col] = df_imputed[col].fillna(method="ffill")
                logger.debug(f"Forward-filled missing values in {col}")
                continue

            df_imputed[col].fillna(fill_value, inplace=True)
            logger.debug(f"Imputed {col} using {strategy}: {fill_value:.4f}")

    return df_imputed


def preprocess_data(
    df: pd.DataFrame, imputation_strategy: str = "mean"
) -> pd.DataFrame:
    """
    Preprocess data by filling missing values and validating quality.

    This function performs comprehensive data preprocessing:
    - Validates DataFrame structure
    - Imputes missing values (numeric columns only)
    - Checks overall data quality
    - Removes duplicate rows (optional)

    Args:
        df: Input DataFrame.
        imputation_strategy: Strategy for handling missing values ('mean', 'median').

    Returns:
        pd.DataFrame: Preprocessed DataFrame ready for modeling.

    Raises:
        DataValidationError: If input is invalid or data quality is poor.

    """
    # Validate input
    validate_dataframe(df, "Input DataFrame")

    # Check data quality before preprocessing
    quality_metrics = check_data_quality(df)
    logger.info(
        f"Data quality before preprocessing: {quality_metrics['missing_ratio']:.1%} missing"
    )

    df_processed = df.copy()

    # Impute missing values
    df_processed = impute_missing_values(df_processed, strategy=imputation_strategy)

    # Remove duplicates
    initial_rows = len(df_processed)
    df_processed = df_processed.drop_duplicates()
    duplicates_removed = initial_rows - len(df_processed)

    if duplicates_removed > 0:
        logger.info(f"Removed {duplicates_removed} duplicate rows")

    logger.info(
        f"Preprocessing complete: {df_processed.shape[0]:,} rows × "
        f"{df_processed.shape[1]} columns"
    )

    return df_processed