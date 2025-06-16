import pytest
import torch

from core.model import BitrateLSTM, quantile_loss


@pytest.fixture
def dummy_batch():
    # batch_size=4, seq_len=10, input_size=4
    x = torch.rand(4, 10, 4)
    # target is a single scalar per sample
    y = torch.rand(4)  # shape: [4]
    return x, y


@pytest.fixture
def model_and_optimizer():
    model = BitrateLSTM(
        input_size=4,
        hidden_size=16,
        num_layers=1,
        dropout=0.0,
        num_outputs=3,  # three quantile outputs
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    return model, optimizer


def test_model_backward_pass_no_errors(dummy_batch, model_and_optimizer):
    """#15 Backward Pass Works
    - Model computes a scalar loss and backward() runs without error
    """
    x, y = dummy_batch
    model, optimizer = model_and_optimizer

    model.train()
    preds, _ = model(x)
    # call with default quantiles=[0.5,0.9,0.95]
    loss = quantile_loss(preds, y)
    optimizer.zero_grad()
    loss.backward()  # should not raise


def test_model_weights_update_after_step(dummy_batch, model_and_optimizer):
    """#16 Weight Update Verification
    - Model weights change after optimizer.step()
    """
    x, y = dummy_batch
    model, optimizer = model_and_optimizer

    model.train()
    preds, _ = model(x)
    loss = quantile_loss(preds, y)
    optimizer.zero_grad()

    # snapshot one param before update
    param_before = next(model.parameters()).clone()
    loss.backward()
    optimizer.step()
    param_after = next(model.parameters())

    assert not torch.allclose(param_before, param_after), "Weights did not change"


def test_model_can_overfit_tiny_batch():
    """#17 Overfit Tiny Batch
    - Model should reduce loss on a single repeated example
    """
    torch.manual_seed(0)
    model = BitrateLSTM(
        input_size=4, hidden_size=32, num_layers=1, dropout=0.0, num_outputs=3
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)

    # single example, single scalar target
    x = torch.rand(1, 10, 4)
    y = torch.tensor([0.2])  # shape: [1]

    model.train()
    losses = []
    for _ in range(100):
        preds, _ = model(x)
        loss = quantile_loss(preds, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0], "Loss did not decrease on a tiny batch"
