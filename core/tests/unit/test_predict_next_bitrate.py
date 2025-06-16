from core.model import predict_next_bitrate


def test_predict_next_bitrate_valid_output():
    """Tests the `predict_next_bitrate` function for end-to-end inference.

    Asserts:
        - The function returns a float.
        - The value is within the expected bitrate range.
    """
    # Arrange: sliding window inputs (length = 5)
    throughput = [1500.0, 1600.0, 1400.0, 1700.0, 1550.0]
    buffer = [20.0, 21.0, 19.0, 22.0, 20.5]
    bitrate = [3000.0, 3200.0, 3100.0, 3300.0, 3150.0]
    rtt = [80.0, 75.0, 85.0, 70.0, 78.0]

    # Act: call prediction
    predicted_bitrate = predict_next_bitrate(
        throughput_window=throughput,
        buffer_window=buffer,
        bitrate_window=bitrate,
        rtt_window=rtt,
        model_path=None,  # use default global model
        quantile_idx=0,  # median
    )

    # Assert
    assert isinstance(predicted_bitrate, float), "Output should be a float."
    assert (
        0.0 <= predicted_bitrate <= 1_000_000.0
    ), f"Predicted bitrate out of range: {predicted_bitrate}"
