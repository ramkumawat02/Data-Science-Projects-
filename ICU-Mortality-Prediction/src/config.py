"""Project constants and configuration."""

# Model Parameters
DEFAULT_TEST_SIZE = 0.2
DEFAULT_RANDOM_STATE = 42
DEFAULT_STRATIFY = True

# Gradient Boosting Parameters
GB_N_ESTIMATORS = 100
GB_LEARNING_RATE = 0.1
GB_MAX_DEPTH = 3

# LSTM Parameters
LSTM_HIDDEN_SIZE = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3

# Feature List
DEFAULT_FEATURES = [
    "age",
    "gender",
    "los",
    "heart_rate",
    "sbp",
    "dbp",
    "mbp",
    "resp_rate",
    "spo2",
    "temperature",
]
DEFAULT_TARGET = "mortality"

# File Paths
DATA_PATH = "data/processed/icu_features.csv"
MODELS_DIR = "models"
LOGS_DIR = "logs"
CONFIG_DIR = "config"

# Model Paths
GB_MODEL_PATH = f"{MODELS_DIR}/gradient_boosting.pkl"
LSTM_MODEL_PATH = f"{MODELS_DIR}/lstm_model.pt"

# Validation Rules
MIN_SAMPLES_FOR_SPLIT = 10
MAX_MISSING_VALUE_RATIO = 0.5  # 50%

# Logging
LOG_FILE = f"{LOGS_DIR}/icu_prediction.log"
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

# Data Validation
REQUIRED_FEATURE_COLUMNS = DEFAULT_FEATURES
REQUIRED_TARGET_COLUMN = DEFAULT_TARGET

__all__ = [
    "DEFAULT_TEST_SIZE",
    "DEFAULT_RANDOM_STATE",
    "DEFAULT_STRATIFY",
    "GB_N_ESTIMATORS",
    "GB_LEARNING_RATE",
    "GB_MAX_DEPTH",
    "LSTM_HIDDEN_SIZE",
    "LSTM_NUM_LAYERS",
    "LSTM_DROPOUT",
    "DEFAULT_FEATURES",
    "DEFAULT_TARGET",
    "DATA_PATH",
    "MODELS_DIR",
    "LOGS_DIR",
    "LOG_FILE",
    "LOG_LEVEL",
    "LOG_FORMAT",
]
