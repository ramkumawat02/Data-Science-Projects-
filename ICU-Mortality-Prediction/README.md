# ICU Mortality Prediction

Machine learning models for in-hospital mortality prediction using ICU patient data derived from MIMIC-IV.

## Overview

This project implements gradient boosting and LSTM neural networks for in-hospital mortality prediction using ICU patient features, with feature engineering, model evaluation, and SHAP-based explainability analysis.

## Installation

Recommended environment: Python 3.11+.

Create an environment and install the project dependencies:

```bash
pip install -r requirements.txt
```

The saved scikit-learn artifacts in `models/` were created with scikit-learn `1.8.0`, so using that version avoids model-loading compatibility warnings.

Raw MIMIC-IV files are not required to explore the dashboard if the processed data and saved model artifacts are present. Full data regeneration requires access to the source dataset.

## Features

- **Data Processing**: EDA, preprocessing, and feature engineering on MIMIC-IV ICU data
- **Models**: 
  - Gradient Boosting Classifier (scikit-learn)
  - LSTM Neural Network (PyTorch)
- **Evaluation**: ROC-AUC, Precision, Recall, F1, Specificity, Confusion Matrix
- **Explainability**: SHAP feature importance and model interpretation
- **Production Ready**: Type hints, error handling, logging, validation

## Development Note

**This project was developed with GitHub Copilot assistance for:**
- Code quality improvements (type hints, docstrings, error handling)
- Documentation and examples
- Refactoring and architectural improvements

The core ML architecture, medical domain logic, data processing strategy, and project design are original work.

## Project Structure

```text
src/                    Core training, evaluation, explainability, and utility code
notebooks/              EDA, feature engineering, modeling, and SHAP notebooks
models/                 Saved gradient boosting and LSTM artifacts
data/processed/         Processed feature dataset used by the training scripts
results/                Metrics and figures
```

## Data Availability

- `data/mimic-iv-3.1/` contains raw source files and is excluded from Git tracking.
- `data/processed/` contains derived features used by training and dashboard workflows.
- The current prediction target uses `hospital_expire_flag`, which corresponds to in-hospital mortality rather than ICU-only mortality.

## Quick Start

Run the interactive dashboard:

```bash
streamlit run app.py
```

Inspect saved model artifacts:

```bash
python src/inspect_models.py
```

Score a sample row with a saved gradient boosting artifact:

```bash
python src/predict_gb.py --artifact models/gb_model.pkl --row 0
```

## Training

Train the gradient boosting model with:

```bash
python -m src.train
```

This reads `data/processed/icu_features.csv` and writes the trained model to `models/gradient_boosting.pkl`.

## Inspect Saved Models

Use the inspector to view artifact metadata instead of opening `.pkl` or `.pt` files directly:

```bash
python src/inspect_models.py
```

Examples:

```bash
python src/inspect_models.py models/gb_model.pkl
python src/inspect_models.py models/gradient_boosting.pkl
python src/inspect_models.py models/gb_shap_inference_bundle.pkl
python src/inspect_models.py models/lstm_model.pt
```

The inspector reports model type, feature names, key hyperparameters, bundle metadata, and checkpoint structure.

## Run Sample Predictions

Use the saved gradient boosting artifacts to score a sample row from the processed dataset:

```bash
python src/predict_gb.py --artifact models/gb_model.pkl --row 0
python src/predict_gb.py --artifact models/gb_shap_inference_bundle.pkl --row 0
```

This prints the selected feature values, predicted probability, and predicted class.

## Dashboard

The Streamlit dashboard uses `models/gb_shap_inference_bundle.pkl` and provides:

- manual patient feature entry
- sample-row scoring from `data/processed/icu_features.csv`
- mortality probability and threshold-based risk classification
- local SHAP feature contributions for the selected patient

This is useful for research demonstration, interactive model review, and quick qualitative inspection of patient-level predictions.

## Notes

- `.pkl` artifacts should be loaded with `joblib` or `pickle`, not `pickletools`.
- `models/lstm_model.pt` is a PyTorch checkpoint and should be loaded with `torch.load(...)`.
- Saved scikit-learn artifacts were created with scikit-learn `1.8.0`; using the same version avoids compatibility warnings when loading them.
