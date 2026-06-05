"""Utility functions for ICU mortality prediction.

Provides model persistence, directory management, and common helper functions.
"""

import logging
from pathlib import Path
from contextlib import contextmanager

import joblib

from src.exceptions import FileIOError

logger = logging.getLogger(__name__)

__all__ = ["create_directory", "save_model", "load_model", "file_operations_context"]


def create_directory(path: str | Path) -> Path:
    """
    Create a directory .

    Uses pathlib for cross-platform compatibility and creates parent
    directories as needed.

    Args:
        path: Directory path to create (str or pathlib.Path).

    Returns:
        Path: The created directory path.

    Raises:
        FileIOError: If directory creation fails.

    
    """
    try:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        logger.debug(f"Directory ready: {path}")
        return path

    except Exception as e:
        logger.error(f"Error creating directory {path}: {e}")
        raise FileIOError(f"Failed to create directory {path}: {e}") from e


def save_model(model, path: str | Path, overwrite: bool = True) -> None:
    """
    Save a model to disk using joblib.

    Serializes the model object to a file with optional overwrite protection.

    Args:
        model: Model object to save.
        path: File path where model will be saved.
        overwrite: If False, raise error if file exists (default: True).

    Raises:
        FileIOError: If save operation fails.
        ValueError: If model is None.

    
    """
    if model is None:
        raise ValueError("Model cannot be None")

    try:
        path = Path(path)

        # Check if file exists and overwrite is False
        if path.exists() and not overwrite:
            raise FileIOError(
                f"File already exists and overwrite=False: {path}"
            )

        # Create parent directories if needed
        create_directory(path.parent)

        # Save model
        joblib.dump(model, path)
        logger.info(f"Model saved successfully: {path} ({path.stat().st_size:,} bytes)")

    except FileIOError:
        raise
    except Exception as e:
        logger.error(f"Error saving model to {path}: {e}")
        raise FileIOError(f"Failed to save model to {path}: {e}") from e


def load_model(path: str | Path):
    """
    Load a model from disk using joblib.

    Deserializes a previously saved model object from file.

    Args:
        path: File path to the saved model.

    Returns:
        Loaded model object.

    Raises:
        FileIOError: If load operation fails or file doesn't exist.


    """
    try:
        path = Path(path)

        if not path.exists():
            raise FileIOError(f"Model file not found: {path}")

        if not path.is_file():
            raise FileIOError(f"Path is not a file: {path}")

        model = joblib.load(path)
        file_size = path.stat().st_size
        logger.info(f"Model loaded successfully: {path} ({file_size:,} bytes)")

        return model

    except FileIOError:
        raise
    except Exception as e:
        logger.error(f"Error loading model from {path}: {e}")
        raise FileIOError(f"Failed to load model from {path}: {e}") from e


@contextmanager
def file_operations_context(operation_name: str):
    """
    Context manager for file operations with logging.

    Provides automatic logging of file operation start and completion.

    Args:
        operation_name: Description of the operation for logging.

    Yields:
        None

    
    """
    logger.info(f"Starting: {operation_name}")
    try:
        yield
        logger.info(f"Completed: {operation_name}")
    except Exception as e:
        logger.error(f"Failed: {operation_name} - {e}")
        raise