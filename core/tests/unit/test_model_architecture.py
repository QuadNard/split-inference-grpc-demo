# core/tests/unit/test_model_architecture.py

import torch

from core.model import BitrateLSTM


def test_extract_embedding_shape_and_no_errors():
    """#2 Embedding Extraction Validity
    - Given a random input tensor, extract_embedding should:
      * return a 2D tensor of shape (batch_size, hidden_size)
      * not raise any errors
    """
    batch_size, seq_len, input_size = 4, 6, 5
    hidden_size = 8

    # Instantiate model
    model = BitrateLSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.0,
        num_outputs=1,
    )

    model.eval()  # Make sure dropout is disabled

    # Random dummy input
    x = torch.randn(batch_size, seq_len, input_size)

    # Run embedding extraction
    with torch.no_grad():
        emb = model.extract_embedding(x)

    # Assertions
    assert isinstance(emb, torch.Tensor), "Embedding must be a torch.Tensor"
    assert emb.ndim == 2, f"Expected 2D tensor, got {emb.ndim}D"
    assert emb.shape == (
        batch_size,
        hidden_size,
    ), f"Expected shape {(batch_size, hidden_size)}, got {tuple(emb.shape)}"


def test_predict_from_embedding_outputs():
    """#3 Prediction Head Behavior
    - Given a random embedding tensor, predict_from_embedding should:
      * return a tensor of shape (batch_size, num_outputs)
      * not raise any errors
    """
    batch_size = 3
    hidden_size = 8
    num_outputs = 4

    # Instantiate model
    model = BitrateLSTM(
        input_size=5,
        hidden_size=hidden_size,
        num_layers=1,
        dropout=0.0,
        num_outputs=num_outputs,
    )

    model.eval()

    # Create a dummy embedding tensor
    emb = torch.randn(batch_size, hidden_size)

    # Run prediction head
    with torch.no_grad():
        out = model.predict_from_embedding(emb)

    # Assertions
    assert isinstance(out, torch.Tensor), "Output must be a torch.Tensor"
    assert out.ndim == 2, f"Expected 2D tensor, got {out.ndim}D"
    assert out.shape == (
        batch_size,
        num_outputs,
    ), f"Expected shape {(batch_size, num_outputs)}, got {tuple(out.shape)}"
