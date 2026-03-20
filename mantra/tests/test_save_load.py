"""Tests for MANTRA save/load functionality."""

import tempfile

import numpy as np
import torch

from mantra import MANTRA


def test_save_load_roundtrip(trained_mantra_model):
    """Test that save/load preserves embeddings exactly."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trained_mantra_model.save(tmpdir)

        loaded = MANTRA.load(tmpdir)

        # Embeddings should be exactly preserved
        orig = trained_mantra_model.get_embeddings()
        restored = loaded.get_embeddings()

        for key in ("A1", "A2", "A3"):
            assert torch.allclose(orig[key], restored[key], atol=1e-6), (
                f"{key} embeddings differ after save/load"
            )


def test_save_load_metadata(trained_mantra_model):
    """Test that metadata is preserved after save/load."""
    with tempfile.TemporaryDirectory() as tmpdir:
        trained_mantra_model.save(tmpdir)
        loaded = MANTRA.load(tmpdir)

        assert loaded.R == trained_mantra_model.R
        assert loaded.n_features == trained_mantra_model.n_features
        assert loaded._trained is True


def test_save_creates_files(trained_mantra_model):
    """Test that save creates the expected files."""
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmpdir:
        trained_mantra_model.save(tmpdir)
        path = Path(tmpdir)

        assert (path / "metadata.json").exists()
        assert (path / "data.npz").exists()
        assert (path / "params.npz").exists()


class TestInputFlexibility:
    """Tests for flexible input types."""

    def test_numpy_array_input(self, set_seeds, device):
        """Test that numpy arrays are accepted as input."""
        arr = np.random.randn(10, 5, 8).astype(np.float32)
        model = MANTRA(
            observations=arr,
            n_features=[8],
            R=3,
            use_gpu=False,
            device=device,
        )
        assert isinstance(model.observations, torch.Tensor)

    def test_n_features_auto_inferred_single_tensor(self, tensor_data, device):
        """Test that n_features is inferred from a single tensor."""
        model = MANTRA(
            observations=tensor_data,
            R=3,
            use_gpu=False,
            device=device,
        )
        assert model.n_features == [8]  # tensor_data has shape (10, 5, 8)

    def test_n_features_auto_inferred_list(self, tensor_data, device):
        """Test that n_features is inferred from a list of tensors."""
        t1 = tensor_data[:, :, :4]
        t2 = tensor_data[:, :, 4:]
        model = MANTRA(
            observations=[t1, t2],
            R=3,
            use_gpu=False,
            device=device,
        )
        assert model.n_features == [4, 4]

    def test_explicit_n_features_still_works(self, tensor_data, device):
        """Test that explicitly passing n_features still works."""
        model = MANTRA(
            observations=tensor_data,
            n_features=[8],
            R=3,
            use_gpu=False,
            device=device,
        )
        assert model.n_features == [8]
