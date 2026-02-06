"""Pytest fixtures for MANTRA tests."""

import pyro
import pytest
import torch

from mantra.utils.seeds import set_all_seeds


@pytest.fixture(autouse=True)
def reset_pyro_state():
    """Reset Pyro's global state before each test."""
    pyro.clear_param_store()
    yield
    pyro.clear_param_store()


@pytest.fixture
def seed():
    """Default random seed for tests."""
    return 42


@pytest.fixture
def set_seeds(seed):
    """Set all random seeds for reproducibility."""
    set_all_seeds(seed)
    return seed


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    return torch.device("cpu")


@pytest.fixture
def small_synthetic_tensor(set_seeds, device):
    """Create a small synthetic 3D tensor for testing.

    Returns a tensor of shape (10, 5, 8) - 10 samples, 5 slices, 8 features.
    """
    from mantra.data import DataGenerator

    generator = DataGenerator(
        n_samples=10,
        n_drugs=5,
        n_features=8,
        R=3,
        a=1.0,
        b=0.5,
        use_gpu=False,
        device=device,
    )
    generator.generate(seed=42)
    return generator


@pytest.fixture
def medium_synthetic_tensor(set_seeds, device):
    """Create a medium synthetic 3D tensor for testing.

    Returns a tensor of shape (50, 10, 20) - 50 samples, 10 slices, 20 features.
    """
    from mantra.data import DataGenerator

    generator = DataGenerator(
        n_samples=50,
        n_drugs=10,
        n_features=20,
        R=5,
        a=1.0,
        b=0.5,
        use_gpu=False,
        device=device,
    )
    generator.generate(seed=42)
    return generator


@pytest.fixture
def synthetic_data(small_synthetic_tensor):
    """Get the generated synthetic data dictionary."""
    return small_synthetic_tensor.get_sim_data()


@pytest.fixture
def tensor_data(synthetic_data):
    """Get just the tensor Y from synthetic data."""
    return synthetic_data["Y_sim"]


@pytest.fixture
def factor_matrices(synthetic_data):
    """Get the true factor matrices from synthetic data."""
    return {
        "A1": synthetic_data["A1_sim"],
        "A2": synthetic_data["A2_sim"],
        "A3": synthetic_data["A3_sim"],
    }


@pytest.fixture
def n_features():
    """Default number of features for single-view tests."""
    return [8]


@pytest.fixture
def mantra_model(tensor_data, n_features, device):
    """Create an untrained MANTRA model."""
    from mantra import MANTRA

    model = MANTRA(
        observations=tensor_data,
        n_features=n_features,
        R=3,
        use_gpu=False,
        device=device,
    )
    return model


@pytest.fixture
def trained_mantra_model(mantra_model, set_seeds):
    """Create a trained MANTRA model with minimal epochs for testing."""
    mantra_model.fit(
        n_epochs=10,
        n_particles=1,
        learning_rate=0.01,
        optimizer="adam",
        verbose=False,
        seed=42,
    )
    return mantra_model
