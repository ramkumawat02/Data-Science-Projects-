"""Model explainability module for ICU mortality prediction.

Provides SHAP-based model interpretation and feature importance analysis
for understanding model predictions.
"""

import logging
from typing import Optional

import numpy as np
import shap
import matplotlib.pyplot as plt

from src.exceptions import ModelEvaluationError

logger = logging.getLogger(__name__)

__all__ = ["run_shap", "get_feature_importance"]


def run_shap(model, X_test, show_plot: bool = True) -> Optional[np.ndarray]:
    """
    Generate SHAP explanations for model predictions.

    Computes SHAP values to explain individual predictions and global
    feature importance using TreeExplainer for tree-based models.

    Args:
        model: Trained tree-based model (GradientBoosting, RandomForest, etc.).
        X_test: Test data for explanation with shape (n_samples, n_features).
        show_plot: If True, display SHAP summary plot (default: True).

    Returns:
        np.ndarray: SHAP values with optional second dimension for multi-class.

    Raises:
        ModelEvaluationError: If model doesn't support TreeExplainer.
    """
    if model is None:
        raise ModelEvaluationError("Model cannot be None")

    if X_test is None or X_test.empty:
        raise ModelEvaluationError("X_test cannot be None or empty")

    try:
        logger.info("Initializing SHAP TreeExplainer...")
        explainer = shap.TreeExplainer(model)

        logger.info(f"Computing SHAP values for {len(X_test):,} samples...")
        shap_values = explainer.shap_values(X_test)

        if show_plot:
            logger.info("Generating SHAP summary plot...")
            plt.figure(figsize=(12, 8))
            shap.summary_plot(shap_values, X_test, show=False)
            plt.tight_layout()
            logger.info("SHAP summary plot generated")

        logger.info("SHAP analysis complete")
        return shap_values

    except AttributeError as e:
        logger.error("Model type not supported by TreeExplainer")
        raise ModelEvaluationError(
            "Model must be a tree-based model (GradientBoosting, RandomForest, etc.)"
        ) from e
    except Exception as e:
        logger.error(f"SHAP analysis failed: {e}")
        raise ModelEvaluationError(f"Failed to compute SHAP values: {e}") from e


def get_feature_importance(
    model, X_test, feature_names: list[str], top_k: int = 10
) -> dict:
    """
    Get feature importance from SHAP values.

    Args:
        model: Trained model.
        X_test: Test data.
        feature_names: List of feature column names.
        top_k: Number of top features to return (default: 10).

    Returns:
        Dictionary with feature importance rankings.

    Examples:
        >>> importance = get_feature_importance(model, X_test, X_test.columns)
        >>> print(importance['top_features'])
    """
    try:
        shap_values = run_shap(model, X_test, show_plot=False)

        # Handle binary classification (list of 2 arrays)
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        # Calculate mean absolute SHAP values
        importance_scores = np.abs(shap_values).mean(axis=0)

        # Rank features
        feature_importance_dict = dict(zip(feature_names, importance_scores))
        sorted_importance = sorted(
            feature_importance_dict.items(), key=lambda x: x[1], reverse=True
        )

        top_features = sorted_importance[:top_k]

        logger.info(f"Top {top_k} features by SHAP importance: {[f[0] for f in top_features]}")

        return {
            "feature_importance": feature_importance_dict,
            "top_features": [f[0] for f in top_features],
            "top_scores": [f[1] for f in top_features],
        }

    except Exception as e:
        logger.error(f"Failed to compute feature importance: {e}")
        raise ModelEvaluationError(f"Failed to get feature importance: {e}") from e