"""End-to-end model training pipeline for ICU mortality prediction.

Orchestrates the complete ML workflow from data loading through model
evaluation and persistence, with comprehensive logging and error handling.
"""

import sys
import logging
from pathlib import Path

from src import config
from src.logging_config import setup_logging
from src.data_loader import load_data
from src.preprocessing import preprocess_data
from src.feature_engineering import select_features
from src.dataset import split_data
from src.model_gb import train_gb, evaluate_gb
from src.utils import save_model, create_directory
from src.exceptions import ICUPredictionError

logger = logging.getLogger(__name__)

__all__ = ["train", "main"]


def train() -> dict:
    """
    Execute the complete model training pipeline.

    Performs end-to-end training:
    1. Load and validate data
    2. Preprocess and clean data
    3. Extract and select features
    4. Split data into train/test sets
    5. Train Gradient Boosting model
    6. Evaluate model performance
    7. Save trained model

    Returns:
        dict: Training results with:
        - status: 'success' or 'failed'
        - model: Trained model object (if successful)
        - metrics: Evaluation metrics
        - data_info: Information about processed data

    Raises:
        ICUPredictionError: If any pipeline step fails.


    """
    try:
        # Initialize logging
        setup_logging()
        logger.info("=" * 80)
        logger.info("Starting ICU Mortality Prediction training pipeline")
        logger.info("=" * 80)

        # Define paths
        data_path = Path(config.DATA_PATH)
        model_dir = Path(config.MODELS_DIR)
        model_path = model_dir / "gradient_boosting.pkl"

        # Step 1: Verify input data exists
        logger.info(f"Step 1: Verifying data file exists at {data_path}...")
        if not data_path.exists():
            raise FileNotFoundError(f"Data file not found: {data_path}")

        # Step 2: Load data
        logger.info("Step 2: Loading data from CSV...")
        df = load_data(data_path)
        initial_shape = df.shape
        logger.info(f"  Loaded: {initial_shape[0]:,} rows × {initial_shape[1]} columns")

        # Step 3: Preprocess data
        logger.info("Step 3: Preprocessing data...")
        df = preprocess_data(df)
        logger.info(f"  After preprocessing: {df.shape[0]:,} rows × {df.shape[1]} columns")

        # Step 4: Select features
        logger.info("Step 4: Selecting features and target...")
        X, y = select_features(df)
        logger.info(f"  Features: {X.shape[1]}, Target distribution: "
                   f"{(y == 1).sum() / len(y):.1%} positive")

        # Step 5: Split data
        logger.info("Step 5: Splitting data into train/test sets...")
        X_train, X_test, y_train, y_test = split_data(X, y)

        # Step 6: Train model
        logger.info("Step 6: Training Gradient Boosting model...")
        model = train_gb(X_train, y_train)

        # Step 7: Evaluate model
        logger.info("Step 7: Evaluating model on test set...")
        preds, metrics = evaluate_gb(model, X_test, y_test)

        # Step 8: Save model
        logger.info("Step 8: Saving trained model...")
        create_directory(model_dir)
        save_model(model, model_path)

        logger.info("=" * 80)
        logger.info(f"✓ Training pipeline completed successfully!")
        logger.info(f"  Model saved to: {model_path}")
        logger.info(f"  Final ROC-AUC: {metrics['roc_auc']:.4f}")
        logger.info("=" * 80)

        return {
            "status": "success",
            "model": model,
            "metrics": metrics,
            "data_info": {
                "initial_shape": initial_shape,
                "final_shape": df.shape,
                "train_size": len(X_train),
                "test_size": len(X_test),
            },
            "model_path": str(model_path),
        }

    except ICUPredictionError as e:
        logger.error(f"Pipeline error: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}
    except Exception as e:
        logger.error(f"Unexpected pipeline error: {e}", exc_info=True)
        return {"status": "failed", "error": str(e)}


def main() -> int:
    """
    Main entry point for training pipeline.

    Returns:
        int: Exit code (0 for success, 1 for failure).
    """
    try:
        results = train()

        if results["status"] == "success":
            logger.info("Training completed successfully")
            return 0
        else:
            logger.error(f"Training failed: {results.get('error', 'Unknown error')}")
            return 1

    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)