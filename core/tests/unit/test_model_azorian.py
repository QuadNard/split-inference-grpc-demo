"""Unit tests for the BitrateLSTM model architecture and data flow.

Focus: Validating the model's structural integrity before training.
"""

import torch


def test_bitrate_lstm_forward_pass(bitrate_model_and_params):
    """Tests the forward pass of the BitrateLSTM model.

    This test now receives both the model and its configuration parameters
    from the fixture, eliminating hardcoded "magic numbers" and making
    the test more robust and maintainable.

    Args:
        bitrate_model_and_params (tuple): A tuple containing the model
            instance and a dictionary of its parameters.

    Asserts:
        - The model processes a correctly shaped input tensor without errors.
        - The output tensor from the model has the expected shape.

    """
    # 1. Arrange
    model, params = bitrate_model_and_params
    batch_size = 5
    sequence_length = 15
    input_size = params["input_size"]
    num_outputs = params["num_outputs"]

    dummy_input = torch.randn(batch_size, sequence_length, input_size)

    # 2. Act
    with torch.no_grad():
        output, _ = model(dummy_input)

    # 3. Assert
    expected_shape = (batch_size, num_outputs)
    assert output.shape == expected_shape, (
        f"Model output shape is incorrect. "
        f"Expected {expected_shape}, but got {output.shape}."
    )
    assert isinstance(output, torch.Tensor), "Model output is not a torch.Tensor."
