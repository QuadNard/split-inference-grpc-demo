"""This file contains shared fixtures for the pytest suite.

Fixtures are reusable objects for tests, such as model instances,
database connections, or data loaders. Pytest automatically discovers
and injects them into tests that request them.

Reference: https://docs.pytest.org/en/latest/how-to/fixtures.html
"""

import pytest

from core.model import BitrateLSTM


@pytest.fixture(scope="session")
def bitrate_model_and_params():
    """Pytest fixture to provide a non-training instance of the BitrateLSTM model.

    This fixture has a 'session' scope, meaning it is created only once for
    the entire test session and shared among all tests that request it. This
    is efficient as we avoid re-initializing the model for every test.

    Yields:
        torch.nn.Module: An initialized BitrateLSTM model in evaluation mode.

    """
    params = {
        "input_size": 10,
        "hidden_size": 64,
        "num_layers": 2,
        "num_outputs": 3,
        "dropout": 0.0,
    }

    # Instantiate the model using the correct parameter names from your __init__
    model = BitrateLSTM(**params)
    # Set the model to evaluation mode. This disables layers like Dropout.
    model.eval()
    return model, params
