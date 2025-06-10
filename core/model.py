import logging
import os
from pathlib import Path
from typing import Optional, Tuple, Union

import torch
import torch.nn as nn

# Configure logging
logger = logging.getLogger(__name__)


class BitrateLSTM(nn.Module):
    """
    LSTM-based model for bitrate prediction.

    Architecture:
    - LSTM layer for sequence processing
    - Fully connected layer for final prediction
    """

    def __init__(
        self,
        input_size: int = 1,
        hidden_size: int = 16,
        num_layers: int = 1,
        dropout: float = 0.0,
    ):
        """
        Initialize the BitrateLSTM model.

        Args:
            input_size: Number of input features (default: 1 for throughput)
            hidden_size: Hidden state size for LSTM
            num_layers: Number of LSTM layers
            dropout: Dropout probability for regularization
        """
        super(BitrateLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.dropout = dropout

        # LSTM layer with optional dropout
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout
            if num_layers > 1
            else 0.0,  # Dropout only works with >1 layers
        )

        # Output layer
        self.fc = nn.Linear(hidden_size, 1)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        """Initialize model weights using Xavier/Glorot initialization."""
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm" in name:
                    # LSTM weights
                    nn.init.xavier_uniform_(param)
                elif "fc" in name:
                    # FC layer weights
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                # Initialize biases to zero
                nn.init.constant_(param, 0.0)

    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Forward pass through the model.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional hidden state tuple (h_0, c_0)

        Returns:
            Tuple of (output, hidden_state)
            - output: Predicted values of shape (batch_size, 1)
            - hidden_state: Tuple of (h_n, c_n)
        """
        # Validate input shape
        if len(x.shape) != 3:
            raise ValueError(
                f"Expected 3D input tensor (batch, seq_len, features), got shape {x.shape}"
            )

        if x.shape[2] != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.shape[2]}")

        # LSTM forward pass
        lstm_out, hidden_state = self.lstm(x, hidden)

        # Take the last time step output
        last_output = lstm_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Apply fully connected layer
        output = self.fc(last_output)  # Shape: (batch_size, 1)

        return output, hidden_state

    def get_embedding(
        self,
        x: torch.Tensor,
        hidden: Optional[Tuple[torch.Tensor, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """
        Get LSTM embeddings without the final FC layer.

        Args:
            x: Input tensor of shape (batch_size, seq_len, input_size)
            hidden: Optional hidden state tuple

        Returns:
            LSTM embeddings of shape (batch_size, hidden_size)
        """
        with torch.no_grad():
            lstm_out, _ = self.lstm(x, hidden)
            return lstm_out[:, -1, :]  # Return last timestep

    def decode_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """
        Decode embeddings back to predictions using only the FC layer.

        Args:
            embedding: Embedding tensor of shape (batch_size, hidden_size)

        Returns:
            Predictions of shape (batch_size, 1)
        """
        if embedding.shape[1] != self.hidden_size:
            raise ValueError(
                f"Expected embedding size {self.hidden_size}, got {embedding.shape[1]}"
            )

        with torch.no_grad():
            return self.fc(embedding)

    def save_model(self, filepath: Union[str, Path]):
        """Save model state dict and configuration."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "state_dict": self.state_dict(),
            "config": {
                "input_size": self.input_size,
                "hidden_size": self.hidden_size,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
            },
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: Union[str, Path]) -> "BitrateLSTM":
        """Load model from saved checkpoint."""
        checkpoint = torch.load(filepath, map_location="cpu")

        # Create model with saved configuration
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])

        logger.info(f"Model loaded from {filepath}")
        return model

    def get_model_info(self) -> dict:
        """Get model architecture information."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "dropout": self.dropout,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
        }


# Global model instance (for backward compatibility)
_global_model: Optional[BitrateLSTM] = None


def get_model(model_path: Optional[str] = None) -> BitrateLSTM:
    """
    Get or create the global model instance.

    Args:
        model_path: Optional path to load model from

    Returns:
        BitrateLSTM model instance
    """
    global _global_model

    if _global_model is None:
        if model_path and os.path.exists(model_path):
            logger.info(f"Loading model from {model_path}")
            _global_model = BitrateLSTM.load_model(model_path)
        else:
            logger.info("Creating new model with random weights")
            _global_model = BitrateLSTM()

        _global_model.eval()

        # Log model info
        info = _global_model.get_model_info()
        logger.info(f"Model info: {info}")

    return _global_model


def predict_bitrate(
    throughput_kbps: float,
    model_path: Optional[str] = None,
    normalization_factor: float = 1000.0,
    scale_factor: float = 4000.0,
) -> float:
    """
    Predict bitrate from throughput using the LSTM model.

    Args:
        throughput_kbps: Input throughput in kbps
        model_path: Optional path to model file
        normalization_factor: Factor to normalize input (default: 1000.0)
        scale_factor: Factor to scale output (default: 4000.0 for ~4Mbps max)

    Returns:
        Predicted bitrate in kbps
    """
    try:
        model = get_model(model_path)

        # Validate input
        if throughput_kbps < 0:
            logger.warning(
                f"Negative throughput value: {throughput_kbps}, clamping to 0"
            )
            throughput_kbps = 0.0

        # Normalize and create tensor
        normalized_input = throughput_kbps / normalization_factor
        x = torch.tensor(
            [[[normalized_input]]], dtype=torch.float32
        )  # Shape: (1, 1, 1)

        # Predict
        with torch.no_grad():
            output, _ = model(x)

        # Denormalize and scale
        predicted_bitrate = float(output.item() * scale_factor)

        # Clamp to reasonable bounds
        predicted_bitrate = max(0.0, min(predicted_bitrate, 1000000.0))  # 0 to 1Gbps

        return predicted_bitrate

    except Exception as e:
        logger.error(f"Error in predict_bitrate: {e}")
        return 0.0


def get_embedding(
    throughput_kbps: float,
    model_path: Optional[str] = None,
    normalization_factor: float = 1000.0,
) -> list:
    """
    Get LSTM embedding for throughput value.

    Args:
        throughput_kbps: Input throughput in kbps
        model_path: Optional path to model file
        normalization_factor: Factor to normalize input

    Returns:
        List of embedding values
    """
    try:
        model = get_model(model_path)

        # Validate and normalize input
        if throughput_kbps < 0:
            throughput_kbps = 0.0

        normalized_input = throughput_kbps / normalization_factor
        x = torch.tensor([[[normalized_input]]], dtype=torch.float32)

        # Get embedding
        embedding = model.get_embedding(x)
        return embedding.numpy().flatten().tolist()

    except Exception as e:
        logger.error(f"Error in get_embedding: {e}")
        return []


def decode_embedding(
    embedding_values: list,
    model_path: Optional[str] = None,
    scale_factor: float = 4000.0,
) -> float:
    """
    Decode embedding back to bitrate prediction.

    Args:
        embedding_values: List of embedding values
        model_path: Optional path to model file
        scale_factor: Factor to scale output

    Returns:
        Predicted bitrate in kbps
    """
    try:
        model = get_model(model_path)

        # Convert to tensor
        embedding_tensor = torch.tensor([embedding_values], dtype=torch.float32)

        # Decode
        output = model.decode_embedding(embedding_tensor)
        predicted_bitrate = float(output.item() * scale_factor)

        # Clamp to reasonable bounds
        return max(0.0, min(predicted_bitrate, 1000000.0))

    except Exception as e:
        logger.error(f"Error in decode_embedding: {e}")
        return 0.0


# Initialize default model for backward compatibility
model = get_model()
