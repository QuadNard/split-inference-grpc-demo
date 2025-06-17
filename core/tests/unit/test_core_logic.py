import pytest
import torch

from core.model import (
    clamp_and_clean_inputs,
    pax_aurora_features,
    predict_next_bitrate,
    quantile_loss,
)


@pytest.mark.parametrize("quantile", [0, 1, 2])
def test_predict_next_bitrate_range_and_type(quantile):
    """#4 predict_next_bitrate() Output Sanity
    - Returns a float
    - Falls within [0, 1_000_000]
    """
    throughput = [100_000 + i * 50 for i in range(10)]
    buffer = [10 + i for i in range(10)]
    bitrate = [500_000 + i * 10 for i in range(10)]
    rtt = [50 + i for i in range(10)]

    val = predict_next_bitrate(throughput, buffer, bitrate, rtt)
    assert isinstance(val, float), "Expected a float"
    assert 0.0 <= val <= 1_000_000.0, f"Value out of range: {val}"


def test_predict_next_bitrate_handles_nan_and_none():
    """#5b Window Input Validation
    - Should handle None, NaN, and negative values without crashing
    - Still returns a float
    """
    throughput = [100, None, float("nan"), -10, 200]
    buffer = [10, 12, None, float("nan"), 15]
    bitrate = [300_000, None, -50, 400_000, float("nan")]
    rtt = [30, None, float("nan"), 40, 50]

    val = predict_next_bitrate(throughput, buffer, bitrate, rtt)
    assert isinstance(val, float), "Should return a float even with invalid inputs"


def test_clamp_and_clean_inputs_clamps_below_zero_and_preserves_others():
    """#6 clamp_and_clean_inputs() Clamping Behavior
    - Negative values → 0.0
    - Other values unchanged
    """
    raw = [-50, 0, 25, 50, 100, 150]
    cleaned = clamp_and_clean_inputs(raw)
    expected = [0.0, 0.0, 25.0, 50.0, 100.0, 150.0]
    assert cleaned == expected, f"Expected {expected}, got {cleaned}"


def test_quantile_loss_non_negative_and_zero_at_perfect_prediction():
    """#7 quantile_loss() Consistency
    - Loss is non-negative
    - Zero when prediction == target
    """
    # preds should be [batch, n_quantiles]
    preds = torch.tensor([[1.0], [2.0], [3.0]])
    targets = torch.tensor([1.0, 2.0, 3.0])
    # pass quantiles as a list
    loss_perfect = quantile_loss(preds, targets, quantiles=[0.5])
    assert isinstance(loss_perfect, torch.Tensor)
    assert loss_perfect.item() == pytest.approx(0.0, abs=1e-6)

    preds2 = preds + 1.0
    loss_error = quantile_loss(preds2, targets, quantiles=[0.5])
    assert loss_error.item() > 0.0


def test_clamp_and_clean_inputs_cleans_invalid_inputs():
    """#8 clamp_and_clean_inputs() Robustness
    - Preserves length and removes NaNs
    """
    raw = [None, float("nan"), -5, 50, 200]
    cleaned = clamp_and_clean_inputs(raw)
    assert len(cleaned) == len(raw), "Length should be preserved"
    assert not any(torch.isnan(torch.tensor(cleaned)).tolist()), "No NaNs should remain"


def test_pax_aurora_features_shape_and_range():
    """#9 pax_aurora_features() Shape and Normalization
    - Returns a tensor of shape [1, seq_len, input_size]
    - All values ∈ [0.0, 1.0]
    """
    window = list(range(8))
    features = pax_aurora_features(
        throughput_window=window,
        buffer_window=window,
        bitrate_window=window,
        rtt_window=window,
    )
    assert isinstance(features, torch.Tensor)
    assert features.ndim == 3, f"Expected 3D tensor, got {features.ndim}D"
    batch, seq_len, input_size = features.shape
    assert batch == 1, f"Batch dim should be 1, got {batch}"
    assert seq_len == len(window), f"Seq len mismatch: {seq_len} vs {len(window)}"
    assert torch.all(
        (features >= 0.0) & (features <= 1.0)
    ), "Features out of [0,1] range"
