"""ICU Mortality Prediction package.

A comprehensive machine learning pipeline for predicting ICU patient mortality
using gradient boosting and neural network models.

This package provides:
- Data loading and preprocessing
- Feature engineering
- Model training (Gradient Boosting, LSTM)
- Model evaluation and explainability
- Utility functions for model persistence

Modules:
    - data_loader: Load data from CSV files
    - preprocessing: Data cleaning and imputation
    - feature_engineering: Feature selection
    - dataset: Data splitting utilities
    - model_gb: Gradient Boosting models
    - model_lstm: LSTM neural networks
    - evaluate: Model evaluation metrics
    - explainability: SHAP-based model interpretation
    - utils: Utility functions
    - config: Project configuration and constants
    - exceptions: Custom exception classes
    - validators: Input validation functions
    - logging_config: Logging setup
"""

__version__ = "1.0.0"
__author__ = "ICU Prediction Team"

# Core functionality
from .data_loader import load_data
from .preprocessing import preprocess_data, impute_missing_values
from .feature_engineering import select_features, get_feature_info
from .dataset import split_data

# Models
from .model_gb import train_gb, evaluate_gb, predict_gb

# Try to import LSTM (optional if torch not installed)
try:
    from .model_lstm import LSTMModel
except ImportError:
    LSTMModel = None  # type: ignore

# Evaluation and explainability
from .evaluate import evaluate, compute_all_metrics
from .explainability import run_shap, get_feature_importance

# Utilities
from .utils import create_directory, save_model, load_model, file_operations_context

# Configuration and exceptions
from .config import (
    DEFAULT_FEATURES,
    DEFAULT_TARGET,
    GB_N_ESTIMATORS,
    LSTM_HIDDEN_SIZE,
)
from .exceptions import (
    ICUPredictionError,
    DataValidationError,
    ModelTrainingError,
    ModelEvaluationError,
    FileIOError,
)
from .validators import (
    validate_dataframe,
    validate_file_path,
    check_missing_columns,
    check_data_quality,
)

# Training pipeline
from .train import train, main

__all__ = [
    # Core functions
    "load_data",
    "preprocess_data",
    "impute_missing_values",
    "select_features",
    "get_feature_info",
    "split_data",
    # Models
    "train_gb",
    "evaluate_gb",
    "predict_gb",
    "LSTMModel",
    # Evaluation
    "evaluate",
    "compute_all_metrics",
    "run_shap",
    "get_feature_importance",
    # Utilities
    "create_directory",
    "save_model",
    "load_model",
    "file_operations_context",
    # Configuration
    "DEFAULT_FEATURES",
    "DEFAULT_TARGET",
    "GB_N_ESTIMATORS",
    "LSTM_HIDDEN_SIZE",
    # Exceptions
    "ICUPredictionError",
    "DataValidationError",
    "ModelTrainingError",
    "ModelEvaluationError",
    "FileIOError",
    # Validators
    "validate_dataframe",
    "validate_file_path",
    "check_missing_columns",
    "check_data_quality",
    # Training
    "train",
    "main",
]

