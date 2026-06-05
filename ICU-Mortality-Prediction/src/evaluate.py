"""Model evaluation module for ICU mortality prediction.

Provides comprehensive metrics for evaluating binary classification models
including ROC-AUC, precision, recall, F1, and calibration metrics.
"""

import logging

from sklearn.metrics import (
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)
import numpy as np

from src.exceptions import ModelEvaluationError

logger = logging.getLogger(__name__)

__all__ = ["evaluate", "compute_all_metrics"]


def evaluate(y_true: np.ndarray | list, preds: np.ndarray | list) -> float:
    """
    Evaluate predictions using ROC-AUC score (legacy function).

    Args:
        y_true: True binary labels.
        preds: Predicted probabilities.

    Returns:
        float: ROC-AUC score.

    Raises:
        ModelEvaluationError: If inputs are invalid.
    """
    try:
        auc = roc_auc_score(y_true, preds)
        logger.info(f"ROC-AUC Score: {auc:.4f}")
        return auc

    except Exception as e:
        logger.error(f"Error calculating AUC: {e}")
        raise ModelEvaluationError(f"Failed to calculate ROC-AUC: {e}") from e


def compute_all_metrics(
    y_true: np.ndarray | list,
    y_pred_proba: np.ndarray | list,
    threshold: float = 0.5,
) -> dict:
    """
    Compute comprehensive evaluation metrics.

    Calculates multiple metrics for thorough model evaluation including
    threshold-dependent and threshold-independent scores.

    Args:
        y_true: True binary labels with shape (n_samples,).
        y_pred_proba: Predicted probabilities with shape (n_samples,).
        threshold: Classification threshold (default: 0.5).

    Returns:
        Dictionary with metrics:
        - roc_auc: Area under ROC curve (threshold-independent)
        - precision: Precision at threshold
        - recall: Recall at threshold
        - f1: F1 score at threshold
        - tn: True negatives
        - fp: False positives
        - fn: False negatives
        - tp: True positives
        - specificity: True negative rate
        - accuracy: Overall accuracy

    Raises:
        ModelEvaluationError: If metrics cannot be computed.

    """
    if y_true is None or len(y_true) == 0:
        raise ModelEvaluationError("y_true cannot be None or empty")

    if y_pred_proba is None or len(y_pred_proba) == 0:
        raise ModelEvaluationError("y_pred_proba cannot be None or empty")

    if len(y_true) != len(y_pred_proba):
        raise ModelEvaluationError(
            f"Length mismatch: y_true={len(y_true)}, y_pred_proba={len(y_pred_proba)}"
        )

    try:
        y_true = np.asarray(y_true)
        y_pred_proba = np.asarray(y_pred_proba)

        # Binary predictions at threshold
        y_pred_binary = (y_pred_proba >= threshold).astype(int)

        # Confusion matrix
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()

        # Calculate metrics
        metrics = {
            "roc_auc": roc_auc_score(y_true, y_pred_proba),
            "precision": precision_score(y_true, y_pred_binary, zero_division=0),
            "recall": recall_score(y_true, y_pred_binary, zero_division=0),
            "f1": f1_score(y_true, y_pred_binary, zero_division=0),
            "specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
            "accuracy": (tp + tn) / (tp + tn + fp + fn),
            "tn": int(tn),
            "fp": int(fp),
            "fn": int(fn),
            "tp": int(tp),
        }

        logger.info(
            f"Evaluation metrics computed:\n"
            f"  ROC-AUC: {metrics['roc_auc']:.4f}\n"
            f"  Precision: {metrics['precision']:.4f}\n"
            f"  Recall: {metrics['recall']:.4f}\n"
            f"  F1: {metrics['f1']:.4f}"
        )

        return metrics

    except Exception as e:
        logger.error(f"Error computing metrics: {e}")
        raise ModelEvaluationError(f"Failed to compute evaluation metrics: {e}") from e