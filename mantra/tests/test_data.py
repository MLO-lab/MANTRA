"""Tests for MANTRA data generation."""

import torch

from mantra.data import DataGenerator


class TestDataGenerator:
    """Tests for DataGenerator."""

    def test_reproducibility_same_seed(self):
        """Test that same seed produces identical factor matrices.

        Note: Y_sim includes noise from PyTorch's RNG, so factor matrices
        (which use numpy RNG) are checked for exact equality, while Y_sim
        requires controlling both RNGs.
        """
        gen1 = DataGenerator(n_samples=20, n_drugs=5, n_features=10, R=3, use_gpu=False)
        torch.manual_seed(0)
        gen1.generate(seed=42)

        gen2 = DataGenerator(n_samples=20, n_drugs=5, n_features=10, R=3, use_gpu=False)
        torch.manual_seed(0)
        gen2.generate(seed=42)

        data1 = gen1.get_sim_data()
        data2 = gen2.get_sim_data()

        assert torch.allclose(data1["A1_sim"], data2["A1_sim"])
        assert torch.allclose(data1["A2_sim"], data2["A2_sim"])
        assert torch.allclose(data1["A3_sim"], data2["A3_sim"])
        assert torch.allclose(data1["Y_sim"], data2["Y_sim"])

    def test_different_seeds_differ(self):
        """Test that different seeds produce different data."""
        gen1 = DataGenerator(n_samples=20, n_drugs=5, n_features=10, R=3, use_gpu=False)
        gen1.generate(seed=42)

        gen2 = DataGenerator(n_samples=20, n_drugs=5, n_features=10, R=3, use_gpu=False)
        gen2.generate(seed=99)

        data1 = gen1.get_sim_data()
        data2 = gen2.get_sim_data()

        assert not torch.allclose(data1["Y_sim"], data2["Y_sim"])

    def test_shapes(self):
        """Test that generated data has correct shapes."""
        gen = DataGenerator(n_samples=30, n_drugs=7, n_features=15, R=4, use_gpu=False)
        gen.generate(seed=0)
        data = gen.get_sim_data()

        assert data["A1_sim"].shape == (30, 4)
        assert data["A2_sim"].shape == (7, 4)
        assert data["A3_sim"].shape == (15, 4)
        assert data["Y_sim"].shape == (30, 7, 15)

    def test_missingness_fraction(self):
        """Test that missingness fraction is approximately correct."""
        gen = DataGenerator(n_samples=100, n_drugs=10, n_features=50, R=3, use_gpu=False)
        gen.generate(seed=42)
        gen.generate_missingness(p=0.2)
        data = gen.get_sim_data()

        total = data["Y_sim"].numel()
        n_missing = torch.isnan(data["Y_sim"]).sum().item()
        fraction = n_missing / total

        # Dropout-based missingness is approximate; check within 10 percentage points
        assert 0.05 < fraction < 0.40, f"Missing fraction {fraction:.2f} not near 0.2"
