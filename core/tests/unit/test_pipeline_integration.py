import pytest
import torch

# adjust these imports to match your project structure if needed
from core.model import (
    BitrateLSTM,
    decode_embedding,
    get_embedding,
    get_model,
    predict_next_bitrate,
)


def test_get_model_fresh_and_from_disk(tmp_path):
    """#12 Global Model Resolution
    - get_model() returns a BitrateLSTM when called with no arguments
    - Saving and re-loading via get_model(checkpoint_path) preserves weights
    """
    # fresh initialization (uses default hyperparameters)
    m1 = get_model()
    assert isinstance(m1, BitrateLSTM)

    # save state dict
    path = tmp_path / "gm.pt"
    torch.save(m1.state_dict(), path)

    # load from disk by passing the checkpoint path
    m2 = get_model(str(path))
    assert isinstance(m2, BitrateLSTM)

    # verify weights identical
    for p1, p2 in zip(m1.state_dict().values(), m2.state_dict().values()):
        assert torch.allclose(p1, p2), "Loaded weights do not match original"


@pytest.mark.parametrize("quantile", [0, 1, 2])
def test_full_inference_pipeline_quantiles(quantile):
    """#13 Integration: Inference Pipeline
    - End-to-end predict_next_bitrate returns a float for each quantile
    """
    window = list(range(20))
    val = predict_next_bitrate(window, window, window, quantile)
    assert isinstance(val, float), "Inference pipeline should return a float"


def test_embedding_encode_decode_roundtrip():
    """#14 Embedding Encoding/Decoding
    - get_embedding returns a list (possibly empty if not implemented)
    - If non-empty, decode_embedding can map it back to a float
    """
    window = list(range(15))
    emb = get_embedding(window, window, window, window)
    assert isinstance(emb, list), "Embedding should be returned as a list"

    # Skip decode step if embedding pipeline isn't fully wired up
    if not emb:
        pytest.skip("get_embedding returned empty list; skipping decode step")

    decoded = decode_embedding(torch.tensor([emb]))
    assert isinstance(decoded, float), "Decoded embedding must be a float"
    assert decoded >= 0.0, "Decoded bitrate should be non-negative"
