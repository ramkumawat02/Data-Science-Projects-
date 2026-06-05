"""Data loading module for ICU mortality prediction.

This module provides utilities for loading and validating data from CSV files.
It includes comprehensive error handling and logging for production use.
"""

import logging
from pathlib import Path

import pandas as pd

from src.exceptions import FileIOError, DataValidationError
from src.validators import validate_file_path

logger = logging.getLogger(__name__)

__all__ = ["load_data"]


def load_data(path: str | Path) -> pd.DataFrame:
    """
    Load data from a CSV file with validation.
    
    Args:
        path: Path to the CSV file .

    Returns:
        pd.DataFrame: Loaded data with shape (n_samples, n_features).

    Raises:
        FileIOError: If the file doesn't exist or can't be read.
        DataValidationError: If the CSV is empty or malformed.
        pd.errors.ParserError: If the CSV format is invalid.
    """
    try:
        path = Path(path)

        # Validate file path exists and is readable
        validate_file_path(path)

        # Load CSV file
        df = pd.read_csv(path)

        # Validate loaded data
        if df.empty:
            raise DataValidationError(f"CSV file is empty: {path}")

        logger.info(
            f"Successfully loaded data from {path}: {df.shape[0]:,} rows × "
            f"{df.shape[1]} columns"
        )
        return df

    except (FileIOError, DataValidationError):
        raise
    except pd.errors.ParserError as e:
        logger.error(f"CSV parsing error in {path}: {e}")
        raise FileIOError(f"Failed to parse CSV file {path}: {e}") from e
    except Exception as e:
        logger.error(f"Unexpected error loading data from {path}: {e}")
        raise FileIOError(f"Unexpected error loading {path}: {e}") from e