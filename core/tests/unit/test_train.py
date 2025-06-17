# core/tests/unit/test_train.py

import pytest
import torch

from core.model import BitrateLSTM, quantile_loss


@pytest.fixture(scope="module")
def dummy_model():
    # match your __init__(input_size, hidden_size, num_layers, num_outputs, dropout)
    input_dim = 1
    hidden_dim = 8
    num_layers = 1
    num_outputs = 3  # your model hard-codes self.quantiles = [0.5,0.9,0.95]
    dropout = 0.0

    return BitrateLSTM(input_dim, hidden_dim, num_layers, num_outputs, dropout)


def test_forward_shape_and_dtype(dummy_model):
    model = dummy_model
    model.eval()

    B, T = 2, 5
    x = torch.randn(B, T, 1)  # (batch, seq_len, input_dim)

    # forward returns (preds, hidden_state)
    preds, _ = model(x)

    # preds should be (batch, Q)
    Q = model.output_fc.out_features
    assert preds.shape == (B, Q), f"Expected ({B}, {Q}), got {preds.shape}"
    assert preds.dtype == x.dtype


def test_training_step_no_errors(dummy_model):
    model = dummy_model
    model.train()

    B, T = 3, 7
    x = torch.randn(B, T, 1)
    y = torch.randn(B)  # one target per batch

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    optimizer.zero_grad()

    preds, _ = model(x)
    Q = preds.shape[1]
    assert preds.shape == (B, Q)

    loss = quantile_loss(preds, y)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0

    # ensure backward + step run without throwing
    loss.backward()
    optimizer.step()


def test_quantile_loss_gradients():
    # quantile_loss works on (batch, Q) preds and (batch,) target
    B, Q = 4, 3
    preds = torch.randn(B, Q, requires_grad=True)
    target = torch.randn(B)

    loss = quantile_loss(preds, target)
    assert isinstance(loss, torch.Tensor) and loss.ndim == 0

    loss.backward()
    assert preds.grad is not None
    assert preds.grad.shape == preds.shape
