"""ITShadow — Bitrate Adaptive Model Core.

Project, 2025
"""

import logging
import os
import threading
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_model_lock = threading.Lock()

# Configure logging
logger = logging.getLogger("ITShadow")


def quantile_loss(preds, target, quantiles=None):
    """Compute quantile regression loss for each quantile in preds.

    preds: [batch, n_quantiles]
    target: [batch] or [batch, 1]
    Returns mean quantile loss across batch and quantiles.
    """
    if quantiles is None:
        quantiles = [0.5, 0.9, 0.95]

    loss = 0.0
    for i, q in enumerate(quantiles):
        errors = target - preds[:, i]
        loss += torch.mean(torch.maximum(q * errors, (q - 1) * errors))
    return loss / len(quantiles)  # Average across quantiles


def compute_smoothness_loss(outputs: torch.Tensor) -> torch.Tensor:
    """Compute smoothness loss by penalizing large differences between consecutive outputs.

    Args:
        outputs (torch.Tensor): Model outputs, expected shape [batch, seq_len, features].

    Returns:
        torch.Tensor: Scalar tensor representing the smoothness loss.

    """
    if outputs.ndim == 3:
        return torch.mean((outputs[:, 1:, :] - outputs[:, :-1, :]) ** 2)
    return torch.tensor(0.0, device=outputs.device)


def clamp_and_clean_inputs(window: list, minval: float = 0.0) -> list:
    """Clamps and cleans input values to ensure non-negative."""
    cleaned = []
    for x in window:
        try:
            val = float(x)
            if val != val or val < minval:  # NaN or negative
                val = minval
        except Exception:
            val = minval
        cleaned.append(max(val, minval))
    return cleaned


def pax_aurora_features(
    throughput_window: list,
    buffer_window: list,
    bitrate_window: list,
    rtt_window: list,
    normalization: dict | None = None,
) -> torch.Tensor:
    """Prepare a sliding window tensor for the LSTM model using PaxAurora signature.

    Args:
        throughput_window: List[float], sliding window of throughput values (length = seq_length).
        buffer_window: List[float], sliding window of buffer occupancy values.
        last_bitrate_window: List[float], sliding window of previously requested bitrates.
        bitrate_window: List[float], sliding window of previous requested bitrates.
        rtt_window: List[float], sliding window of RTT values (ms).
        normalization: Optional; dict mapping feature names to normalization constants.
            Defaults: {'throughput': 1000.0, 'buffer': 30.0, 'bitrate': 8000.0, 'rtt': 200.0}

    Returns:
        torch.Tensor: Shape [batch_size=1, seq_length, input_size=4]

    """
    # Default normalization values if not provided
    if normalization is None:
        logger.warning("No normalization provided, using default values.")

    normalization = normalization or {
        "throughput": 1000.0,  # Default normalization for throughput
        "buffer": 30.0,  # Default normalization for buffer level
        "bitrate": 8000.0,  # Default normalization for bitrate
        "rtt": 200.0,  # Default normalization for RTT
    }
    seq_length = len(throughput_window)
    # Defensive: all windows must be the same length
    # Defensive: all windows must be same length
    assert all(
        len(lst) == seq_length for lst in [buffer_window, bitrate_window, rtt_window]
    ), "All input windows must be of the same length!"

    feature_seq = [
        [
            max(0.0, throughput_window[i]) / normalization["throughput"],
            max(0.0, buffer_window[i]) / normalization["buffer"],
            max(0.0, bitrate_window[i]) / normalization["bitrate"],
            max(0.0, rtt_window[i]) / normalization["rtt"],
        ]
        for i in range(seq_length)
    ]
    return torch.tensor([feature_seq], dtype=torch.float32)  # [1, seq_length, 4]


# Define the model class
class BitrateLSTM(nn.Module):
    """Sequence model for next-bitrate prediction."""

    def __init__(
        self,
        hidden_size: int = 64,
        num_layers: int = 1,
        input_size: int = 4,
        num_outputs: int = 3,  # for quantiles or multi-step, else keep 1
        dropout: float = 0.0,
    ):
        """Initialize the BitrateLSTM model.

        Args:
            input_size: Number of features in the input sequence.
            hidden_size: Number of features in the hidden state.
            num_layers: Number of recurrent layers.
            num_outputs: Number of outputs from the model (e.g., number of quantiles or steps).
            dropout: Dropout probability between LSTM layers (0 disables).

        """
        super(BitrateLSTM, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_outputs = num_outputs
        self.dropout = dropout

        # Define the LSTM layer
        self.lstm_layer = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,  # No dropout for single layer
        )

        # Output layer
        self.output_fc = nn.Linear(hidden_size, num_outputs)  # Assuming a single output

        # Initialize weights
        self.init_weights()

    def init_weights(self):
        """Initialize model weights.

        Applies Xavier/Glorot initialization to all LSTM and fully connected (fc) layer weights,
        and initializes all biases to zero.
        """
        for name, param in self.named_parameters():
            if "weight" in name:
                if "lstm_layer" in name:
                    # LSTM weights
                    nn.init.xavier_uniform_(param)
                elif "output_fc" in name:
                    # Fully connected layer weights
                    nn.init.xavier_uniform_(param)
            elif "bias" in name:
                nn.init.constant_(param, 0.0)  # Initialize biases to zero

    def forward(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """Perform a forward pass of the BitrateLSTM model.

        Args:
            x: Input tensor of shape [batch_size, sequence_length, input_size].
            hidden: Optional tuple of the LSTM's previous (hidden_state, cell_state).

        Returns:
            output: Output predictions after passing through the LSTM and FC layers.
            hidden_state: The updated hidden and cell states from the LSTM.

        """
        # Validate input shape
        if len(x.shape) != 3:
            raise ValueError(f"Input tensor must be 3D, got {x.shape} instead.")
        if x.shape[2] != self.input_size:
            raise ValueError(f"Expected input_size {self.input_size}, got {x.shape[2]}")

        lstm_out, hidden_state = self.lstm_layer(x, hidden)
        last_output = lstm_out[:, -1, :]
        output = self.output_fc(last_output)
        return output, hidden_state

    def extract_embedding(
        self,
        x: torch.Tensor,
        hidden: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        """Get LSTM embeddings without the final FC layer."""
        with torch.no_grad():
            lstm_out, _ = self.lstm_layer(x, hidden)
            return lstm_out[:, -1, :]  # Return last timestep

    def predict_from_embedding(self, embedding: torch.Tensor) -> torch.Tensor:
        """Decode a hidden embedding to an output prediction using the model's output layer.

        Args:
            embedding: Tensor of shape [batch_size, hidden_size].

        Returns:
            torch.Tensor: Output tensor of shape [batch_size, 1].

        """
        if embedding.shape[1] != self.hidden_size:
            raise ValueError(
                f"Expected embedding size {self.hidden_size}, got {embedding.shape[1]}"
            )
        with torch.no_grad():
            return self.output_fc(embedding)

    def black_saber_quantize(self) -> nn.Module:
        """Quantize (compress) the model using dynamic quantization for deployment.

        Returns a quantized model (weights stored as 8-bit integers).
        """
        import torch.nn as nn

        quantized = torch.quantization.quantize_dynamic(
            self, {nn.LSTM, nn.Linear}, dtype=torch.qint8
        )
        logger.info("Model quantized with BlackSaber utility.")
        return quantized

    def save_model(self, filepath: str | Path):
        """Save the model’s state dictionary and configuration to disk.

        Args:
            filepath: Path or string representing where to save the checkpoint.

        """
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
    def load_model(cls, filepath: str | Path) -> "BitrateLSTM":
        """Load a BitrateLSTM model from a saved checkpoint.

        Args:
            filepath: Path or string pointing to the checkpoint file.

        Returns:
            BitrateLSTM: The model instance loaded with state and configuration.

        """
        checkpoint = torch.load(filepath, map_location=torch.device("cpu"))

        # Create model with the saved configuration
        model = cls(**checkpoint["config"])
        model.load_state_dict(checkpoint["state_dict"])

        logger.info(f"Model loaded from {filepath}")
        return model

    def get_model_info(self) -> dict:
        """Return a dictionary with model hyperparameters and parameter counts.

        Returns:
            dict: Contains input_size, hidden_size, num_layers, dropout, total_parameters,
                and trainable_parameters.

        """
        global _global_model
        with _model_lock:
            if _global_model is None:
                raise ValueError(
                    "Model has not been initialized. Call get_model() first."
                )

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
_global_model: BitrateLSTM | None = None


def get_model(model_path: str | None = None) -> BitrateLSTM:
    """Return a global BitrateLSTM model instance.

    Load from a checkpoint if provided and not already loaded. Otherwise, create a new randomly
    initialized model.

    Args:
        model_path: Optional path to a model checkpoint.

    Returns:
        BitrateLSTM: Loaded or new model instance.

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
        # Log model information
        info = _global_model.get_model_info()
        logger.info(f"Model info: {info}")

    return _global_model


def prepare_tensors_for_prediction(data: np.ndarray) -> torch.Tensor:
    """Convert raw input data (e.g., from logs or CSV) into a tensor with the correct shape infer.

    Args:
        data (np.ndarray): 2D array of shape [sequence_length, input_size]

    Returns:
        torch.Tensor: 3D tensor [batch=1, sequence_length, input_size] as float32

    """
    if not isinstance(data, np.ndarray):
        raise ValueError("Input must be a NumPy array")

    if len(data.shape) != 2:
        raise ValueError(
            f"Expected shape (sequence_length, input_size), got {data.shape}"
        )

    return torch.tensor(data[None, :, :], dtype=torch.float32)


def predict_next_bitrate(
    throughput_window: list,
    buffer_window: list,
    bitrate_window: list,
    rtt_window: list,
    model_path: str | None = None,
    normalization: dict | None = None,
    scale_factor: float = 4000.0,
    quantile_idx: int = 0,  # 0 = p50, 1 = p90, 2 = p95
) -> float:
    """Predict the next bitrate from current network and playback metrics using the LSTM model.

    Args:
        throughput_window: List of recent throughput values (e.g., in kbps).
        buffer_window: List of recent buffer occupancy values (e.g., in seconds).
        bitrate_window: List of previously requested bitrate values (e.g., in kbps).
        rtt_window: List of recent round-trip time values (e.g., in ms).
        model_path: Optional; path to a saved model checkpoint.
        normalization: Optional; dict mapping feature names to normalization constants.
            Example: {'throughput': 1000.0, 'buffer': 30.0, 'bitrate': 8000.0, 'rtt': 200.0}
        scale_factor: Scales model output back to the bitrate domain.
        quantile_idx: Index of quantile to return (0=median, 1=90th, 2=95th).

    Returns:
        Predicted next bitrate as a float, or 0.0 if an error occurs.

    """
    try:
        normailization_factor = normalization or {
            "throughput": 1000.0,  # Default normalization for throughput
            "buffer": 30.0,  # Default normalization for buffer level
            "bitrate": 8000.0,  # Default normalization for bitrate
            "rtt": 200.0,  # Default normalization for RTT
        }

        # Validate and clean inputs
        throughput_window = clamp_and_clean_inputs(throughput_window)
        buffer_window = clamp_and_clean_inputs(buffer_window)
        rtt_window = clamp_and_clean_inputs(rtt_window)
        bitrate_window = clamp_and_clean_inputs(bitrate_window)
        # Defensive: ensure all windows are the same length
        if not all(
            len(lst) == len(throughput_window)
            for lst in [buffer_window, rtt_window, bitrate_window]
        ):
            raise ValueError("All input windows must be of the same length!")

        x = pax_aurora_features(
            throughput_window,
            buffer_window,
            bitrate_window,
            rtt_window,
            normalization=normailization_factor,
        )

        # Predict
        model = get_model(model_path)
        with torch.no_grad():
            outputs, _ = model(x)
        quantiles = outputs[0].tolist()
        predicted_bitrate = quantiles[quantile_idx] * scale_factor
        # Clamp output to a reasonable range
        predict_next_bitrate = max(0.0, min(predicted_bitrate, 1_000_000.0))
        return predict_next_bitrate

    except Exception as e:
        logger.error(f"Error predicting next bitrate: {e}")
        return 0.0


def get_embedding(
    throughput_window: list[float],
    buffer_window: list[float],
    bitrate_window: list[float],
    rtt_window: list[float],
    model_path: str | None = None,
) -> list[float]:
    """Generate an embedding vector from input feature windows.

    Args:
        throughput_window (List[float]): Sliding window of throughput values.
        buffer_window (List[float]): Sliding window of buffer occupancy values.
        bitrate_window (List[float]): Sliding window of previously requested bitrates.
        rtt_window (List[float]): Sliding window of RTT values.
        model_path (Optional[str]): Optional model checkpoint path.

    Returns:
        List[float]: Flattened embedding vector

    """
    try:
        model = get_model(model_path)

        # Construct the feature tensor: [1, seq_len, input_size]
        x = pax_aurora_features(
            throughput_window, buffer_window, bitrate_window, rtt_window
        )

        # Get embedding vector from last LSTM hidden state
        with torch.no_grad():
            _, (h_n, _) = model.lstm(x)
            embedding = h_n[-1]  # shape: [batch, hidden_size]

        return embedding.numpy().flatten().tolist()

    except Exception as e:
        logger.error(f"Error in get_embedding: {e}")
        return []


def decode_embedding(
    embedding_tensor: torch.Tensor,
    num_outputs: int = 1,
) -> float:
    """Decode a latent embedding vector into a bitrate prediction.

    Args:
        embedding_tensor (torch.Tensor): Tensor of shape [batch_size, hidden_size]
        num_outputs (int): Number of prediction outputs (default: 1)

    Returns:
        float: The first predicted bitrate value from the decoded output.

    """
    try:
        # Lightweight decoder head for standalone testing (matches model head)
        decoder = torch.nn.Linear(embedding_tensor.shape[1], num_outputs)

        with torch.no_grad():
            prediction = decoder(embedding_tensor)

        return prediction[0, 0].item()

    except Exception as e:
        logger.error(f"Error in decode_embedding: {e}")
        return 0.0


model = get_model()  # Default model instance

criterion = nn.MSELoss()  # Default loss function for training
lambda_stardust = 0.1  # Smoothness loss weight (flair—optional, configurable)


def train_batch(
    model: BitrateLSTM,
    optimizer: torch.optim.Optimizer,
    batch: tuple,
    quantiles: list = [0.5, 0.9, 0.95],
    device: torch.device = torch.device("cpu"),
) -> tuple:
    """Perform a training step with quantile loss and smoothness regularization.

    Args:
        model: BitrateLSTM instance.
        optimizer: Optimizer for updating model weights.
        batch: Tuple (inputs, targets); inputs shape [batch, seq_len, input_size], targets [batch].
        quantiles: List of quantiles predicted by the model.
        device: Device to run training step on.

    Returns:
        Tuple: (total_loss, main_loss, smoothness_loss) as floats.

    """
    model.train()  # Set model to training mode
    inputs, targets = batch
    inputs, targets = inputs.to(device), targets.to(device)

    raw = model(inputs)
    outputs = raw[0] if isinstance(raw, tuple) else raw

    # Broadcast our single‐value targets across all quantile outputs
    targets_q = targets.unsqueeze(1).expand_as(outputs)  # now [batch, num_outputs]
    loss_mse = criterion(outputs, targets_q)  # MSE over each quantile
    # Main quantile loss (over batch)
    main_loss = quantile_loss(outputs, targets, quantiles=quantiles)

    smoothness = compute_smoothness_loss(outputs)  # Compute smoothness loss
    total_loss = main_loss + lambda_stardust * smoothness * loss_mse
    total_loss.backward()  # Backpropagation
    optimizer.step()  # Update weights

    # Logging/return
    logger.info(
        (
            f"Total Loss: {total_loss.item():.4f} | Main: {main_loss.item():.4f} | "
            f"Stardust Smoothness: {smoothness.item():.6f}"
        )
    )

    return total_loss.item(), main_loss.item(), smoothness.item()
