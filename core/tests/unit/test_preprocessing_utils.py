import numpy as np
import torch

from core.model import prepare_tensors_for_prediction


def test_prepare_tensors_for_prediction():
    """Tests the `prepare_tensors_for_prediction` helper function.

    Asserts:
        - The function correctly converts a NumPy array into a torch.Tensor.
        - The resulting tensor has the correct shape (includes a batch dimension).
        - The resulting tensor has the correct data type (torch.float32).
    """
    # 1. Arrange: Create a sample input sequence as a NumPy array.
    # This simulates raw data before it's processed.
    sequence_length = 20
    num_features = 4  # Matches the default input_size of your model

    # Shape: (sequence_length, num_features)
    raw_data = np.random.rand(sequence_length, num_features)

    # 2. Act: Call the function to process the data.
    processed_tensor = prepare_tensors_for_prediction(raw_data)

    # 3. Assert: Verify the output tensor's properties.
    assert isinstance(
        processed_tensor, torch.Tensor
    ), "Output should be a torch.Tensor."

    # The function should add a batch dimension of 1.
    expected_shape = (1, sequence_length, num_features)
    assert processed_tensor.shape == expected_shape, (
        f"Tensor shape is incorrect. "
        f"Expected {expected_shape}, but got {processed_tensor.shape}."
    )

    # The function should convert the data to float32.
    assert processed_tensor.dtype == torch.float32, (
        f"Tensor dtype is incorrect. "
        f"Expected torch.float32, but got {processed_tensor.dtype}."
    )
