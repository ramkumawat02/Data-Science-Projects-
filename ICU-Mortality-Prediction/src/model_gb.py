"""Gradient Boosting model training and evaluation module.

Provides comprehensive model training, evaluation, and prediction capabilities
with detailed logging and error handling.
"""

import logging

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import roc_auc_score, precision_score, recall_score, f1_score
import pandas as pd
import numpy as np

from src import config
from src.exceptions import ModelTrainingError, ModelEvaluationError
from src.validators import validate_dataframe

logger = logging.getLogger(__name__)

__all__ = ["train_gb", "evaluate_gb", "predict_gb"]


def train_gb(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = config.GB_N_ESTIMATORS,
    learning_rate: float = config.GB_LEARNING_RATE,
    max_depth: int = config.GB_MAX_DEPTH,
    random_state: int = config.DEFAULT_RANDOM_STATE,
) -> GradientBoostingClassifier:
    """
    Train a Gradient Boosting classifier model.

    Trains a gradient boosting ensemble with specified hyperparameters
    on the provided training data.

    Args:
        X_train: Training features with shape (n_samples, n_features).
        y_train: Training target labels with shape (n_samples,).
        n_estimators: Number of boosting stages (default: 100).
        learning_rate: Learning rate / shrinkage parameter (default: 0.1).
        max_depth: Maximum tree depth (default: 3).
        random_state: Random seed for reproducibility (default: 42).

    Returns:
        GradientBoostingClassifier: Trained model ready for prediction.

    Raises:
        ModelTrainingError: If training fails or data is invalid.


    """
    # Validate inputs
    validate_dataframe(X_train, "X_train")
    validate_dataframe(y_train.to_frame(), "y_train")

    if len(X_train) != len(y_train):
        raise ModelTrainingError(
            f"Data mismatch: X_train has {len(X_train)} rows, "
            f"y_train has {len(y_train)} rows"
        )

    try:
        logger.info(
            f"Initializing Gradient Boosting model with parameters: "
            f"n_estimators={n_estimators}, learning_rate={learning_rate}, "
            f"max_depth={max_depth}"
        )

        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            random_state=random_state,
        )

        logger.info(
            f"Training on {len(X_train):,} samples with {X_train.shape[1]} features..."
        )
        model.fit(X_train, y_train)

        logger.info(
            f"Training complete! Features: {X_train.shape[1]}, "
            f"Samples: {len(X_train):,}"
        )

        return model

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise ModelTrainingError(f"Failed to train Gradient Boosting model: {e}") from e


def evaluate_gb(
    model: GradientBoostingClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    threshold: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Evaluate a trained Gradient Boosting model.

    Computes comprehensive evaluation metrics including ROC-AUC, precision,
    recall, and F1 score.

    Args:
        model: Trained GradientBoostingClassifier model.
        X_test: Test features.
        y_test: Test target labels.
        threshold: Classification threshold for binary predictions (default: 0.5).

    Returns:
        Tuple of (predicted_probabilities, metrics_dict) where metrics_dict contains:
        - roc_auc: Area under ROC curve
        - precision: Precision score
        - recall: Recall score
        - f1: F1 score

    Raises:
        ModelEvaluationError: If evaluation fails.

    Examples:
        >>> preds, metrics = evaluate_gb(model, X_test, y_test)
        >>> print(f"ROC-AUC: {metrics['roc_auc']:.4f}")
    """
    if model is None:
        raise ModelEvaluationError("Model cannot be None")

    validate_dataframe(X_test, "X_test")

    try:
        # Generate predictions
        preds_proba = model.predict_proba(X_test)[:, 1]
        preds_binary = (preds_proba >= threshold).astype(int)

        # Calculate metrics
        metrics = {
            "roc_auc": roc_auc_score(y_test, preds_proba),
            "precision": precision_score(y_test, preds_binary, zero_division=0),
            "recall": recall_score(y_test, preds_binary, zero_division=0),
            "f1": f1_score(y_test, preds_binary, zero_division=0),
        }

        logger.info(
            f"Model evaluation complete!\n"
            f"  ROC-AUC: {metrics['roc_auc']:.4f}\n"
            f"  Precision: {metrics['precision']:.4f}\n"
            f"  Recall: {metrics['recall']:.4f}\n"
            f"  F1: {metrics['f1']:.4f}"
        )

        return preds_proba, metrics

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise ModelEvaluationError(f"Failed to evaluate model: {e}") from e


def predict_gb(
    model: GradientBoostingClassifier,
    X: pd.DataFrame,
    return_proba: bool = True,
) -> np.ndarray:
    """
    Generate predictions using a trained Gradient Boosting model.

    Args:
        model: Trained model.
        X: Features to predict on.
        return_proba: If True, return probabilities; else binary predictions (default: True).

    Returns:
        np.ndarray: Predictions with shape (n_samples,).

    Raises:
        ModelEvaluationError: If prediction fails.
    """
    if model is None:
        raise ModelEvaluationError("Model cannot be None")

    try:
        if return_proba:
            return model.predict_proba(X)[:, 1]
        else:
            return model.predict(X)

    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise ModelEvaluationError(f"Failed to generate predictions: {e}") from e