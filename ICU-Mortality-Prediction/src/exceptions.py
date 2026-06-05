"""Custom exceptions for ICU mortality prediction."""

__all__ = [
    "ICUPredictionError",
    "DataValidationError",
    "ModelTrainingError",
    "ModelEvaluationError",
    "FileIOError",
    "ConfigurationError",
]


class ICUPredictionError(Exception):
    """Base exception for ICU prediction project."""

    pass


class DataValidationError(ICUPredictionError):
    """Raised when data validation fails."""

    pass


class ModelTrainingError(ICUPredictionError):
    """Raised when model training fails."""

    pass


class ModelEvaluationError(ICUPredictionError):
    """Raised when model evaluation fails."""

    pass


class FileIOError(ICUPredictionError):
    """Raised when file operations fail."""

    pass


class ConfigurationError(ICUPredictionError):
    """Raised when configuration is invalid."""

    pass
