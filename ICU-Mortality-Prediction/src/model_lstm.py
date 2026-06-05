"""LSTM neural network module for ICU mortality prediction.

Provides PyTorch-based LSTM models for sequence classification of
ICU patient data.
"""

import logging

import torch
import torch.nn as nn

from src import config
from src.exceptions import ModelTrainingError

logger = logging.getLogger(__name__)

__all__ = ["LSTMModel"]


class LSTMModel(nn.Module):
    """
    LSTM-based neural network for sequence classification.

    Architecture:
    - Bidirectional LSTM layers for sequence processing
    - Dropout for regularization
    - Fully connected output layer with sigmoid activation

    Attributes:
        input_size: Number of input features
        hidden_size: Dimension of LSTM hidden states
        num_layers: Number of stacked LSTM layers
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int = config.LSTM_HIDDEN_SIZE,
        num_layers: int = config.LSTM_NUM_LAYERS,
        dropout: float = config.LSTM_DROPOUT,
    ):
        """
        Initialize LSTM model.

        Args:
            input_size: Number of input features (must be > 0).
            hidden_size: LSTM hidden state dimension (default: 64).
            num_layers: Number of LSTM layers (default: 2).
            dropout: Dropout rate 0-1 (default: 0.3).

        Raises:
            ValueError: If parameters are invalid.

     
        """
        super().__init__()

        # Validate parameters
        if not isinstance(input_size, int) or input_size <= 0:
            raise ValueError(f"input_size must be positive integer, got {input_size}")

        if not isinstance(hidden_size, int) or hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive integer, got {hidden_size}")

        if not isinstance(num_layers, int) or num_layers <= 0:
            raise ValueError(f"num_layers must be positive integer, got {num_layers}")

        if not (0 <= dropout < 1):
            raise ValueError(f"dropout must be in [0, 1), got {dropout}")

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # LSTM layers
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)
        self.sigmoid = nn.Sigmoid()

        logger.info(
            f"LSTM model initialized: input_size={input_size}, "
            f"hidden_size={hidden_size}, num_layers={num_layers}, dropout={dropout:.2f}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the LSTM model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size).

        Returns:
            torch.Tensor: Output tensor of shape (batch_size, 1) with sigmoid activation.

        Raises:
            ValueError: If input is None or invalid shape.

        Examples:
            >>> x = torch.randn(32, 3, 10)  # batch_size=32, seq_len=3, features=10
            >>> model = LSTMModel(input_size=10)
            >>> output = model(x)
            >>> print(output.shape)
            torch.Size([32, 1])
        """
        if x is None:
            raise ValueError("Input tensor cannot be None")

        if not isinstance(x, torch.Tensor):
            raise ValueError(f"Expected torch.Tensor, got {type(x)}")

        # LSTM forward: outputs all hidden states
        lstm_out, (h_n, c_n) = self.lstm(x)

        # Take output from last time step
        last_hidden = lstm_out[:, -1, :]

        # Apply dropout for regularization
        last_hidden = self.dropout(last_hidden)

        # Fully connected layer
        logits = self.fc(last_hidden)

        # Sigmoid activation for binary classification
        output = self.sigmoid(logits)

        return output

    def get_config(self) -> dict:
        """
        Get model configuration dictionary.

        Returns:
            Dictionary with model hyperparameters.
        """
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "model_type": "LSTM",
        }