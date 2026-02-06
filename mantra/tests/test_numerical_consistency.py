"""Numerical consistency tests for MANTRA.

These tests verify that code refactoring does not change the numerical
behavior of the model. They compare current outputs against golden
reference values generated before refactoring.

IMPORTANT: Run scripts/generate_golden_references.py ONCE before
making any code changes, and commit the golden_references.pkl file.

Acceptance criteria (from CLAUDE.md):
- ELBO within 1% of golden reference
- Embeddings with >0.98 absolute correlation (accounting for sign invariance)
- Reconstruction RMSE within 0.5% of golden reference
"""

import pickle
from pathlib import Path

import pyro
import pytest
import torch

from mantra import MANTRA
from mantra.data import DataGenerator
from mantra.utils.seeds import set_all_seeds


def compute_absolute_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute absolute correlation between two tensors.

    Factor models have sign ambiguity - Factor k in one run might equal
    -Factor k in another. This function accounts for that by taking the
    absolute value of the correlation.
    """
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()

    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()

    numerator = (a_centered * b_centered).sum()
    denominator = torch.sqrt((a_centered**2).sum() * (b_centered**2).sum())

    if denominator < 1e-8:
        return 0.0

    correlation = (numerator / denominator).item()
    return abs(correlation)


@pytest.fixture(scope="module")
def golden_references():
    """Load golden reference values."""
    golden_path = Path(__file__).parent.parent.parent / "tests" / "golden_references.pkl"

    if not golden_path.exists():
        pytest.skip(
            f"Golden references not found at {golden_path}. "
            "Run 'python scripts/generate_golden_references.py' first."
        )

    with open(golden_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture(scope="module")
def current_run(golden_references):
    """Run the model with the same configuration as golden references."""
    config = golden_references["config"]

    set_all_seeds(config["seed"])

    generator = DataGenerator(
        n_samples=config["n_samples"],
        n_drugs=config["n_drugs"],
        n_features=config["n_features"],
        R=config["R"],
        a=1.0,
        b=0.5,
        use_gpu=False,
        device=torch.device("cpu"),
    )
    generator.generate(seed=config["seed"])
    sim_data = generator.get_sim_data()

    pyro.clear_param_store()

    model = MANTRA(
        observations=sim_data["Y_sim"],
        n_features=[config["n_features"]],
        R=config["R"],
        a=1.0,
        b=0.5,
        c=1.0,
        d=0.5,
        use_gpu=False,
        device=torch.device("cpu"),
    )

    loss_history, _ = model.fit(
        n_epochs=config["n_epochs"],
        n_particles=config["n_particles"],
        learning_rate=config["learning_rate"],
        optimizer="adam",
        verbose=False,
        seed=config["seed"],
    )

    samples = model._guide.median()
    A1 = samples["A1"].detach().cpu()
    A2 = samples["A2"].detach().cpu()
    A3 = samples["A3"].detach().cpu()
    reconstruction = torch.einsum("ir,jr,kr->ijk", A1, A2, A3)
    rmse = torch.sqrt(torch.mean((reconstruction - sim_data["Y_sim"].cpu()) ** 2)).item()

    return {
        "A1_embedding": A1,
        "A2_embedding": A2,
        "A3_embedding": A3,
        "reconstruction": reconstruction,
        "reconstruction_rmse": rmse,
        "elbo_history": loss_history,
        "final_elbo": loss_history[-1],
    }


class TestNumericalConsistency:
    """Test suite for numerical consistency."""

    def test_elbo_within_tolerance(self, golden_references, current_run):
        """Test that final ELBO is within 1% of golden reference."""
        golden_elbo = golden_references["final_elbo"]
        current_elbo = current_run["final_elbo"]

        relative_diff = abs(current_elbo - golden_elbo) / abs(golden_elbo)

        assert relative_diff < 0.01, (
            f"ELBO differs by {relative_diff * 100:.2f}% "
            f"(golden: {golden_elbo}, current: {current_elbo})"
        )

    def test_rmse_within_tolerance(self, golden_references, current_run):
        """Test that reconstruction RMSE is within 0.5% of golden reference."""
        golden_rmse = golden_references["reconstruction_rmse"]
        current_rmse = current_run["reconstruction_rmse"]

        relative_diff = abs(current_rmse - golden_rmse) / golden_rmse

        assert relative_diff < 0.005, (
            f"RMSE differs by {relative_diff * 100:.2f}% "
            f"(golden: {golden_rmse}, current: {current_rmse})"
        )

    def test_a1_embedding_correlation(self, golden_references, current_run):
        """Test that A1 embeddings have >0.98 absolute correlation."""
        golden_A1 = golden_references["A1_embedding"]
        current_A1 = current_run["A1_embedding"]

        correlation = compute_absolute_correlation(golden_A1, current_A1)

        assert correlation > 0.98, f"A1 embedding correlation is {correlation:.4f}, expected >0.98"

    def test_a2_embedding_correlation(self, golden_references, current_run):
        """Test that A2 embeddings have >0.98 absolute correlation."""
        golden_A2 = golden_references["A2_embedding"]
        current_A2 = current_run["A2_embedding"]

        correlation = compute_absolute_correlation(golden_A2, current_A2)

        assert correlation > 0.98, f"A2 embedding correlation is {correlation:.4f}, expected >0.98"

    def test_a3_embedding_correlation(self, golden_references, current_run):
        """Test that A3 embeddings have >0.98 absolute correlation."""
        golden_A3 = golden_references["A3_embedding"]
        current_A3 = current_run["A3_embedding"]

        correlation = compute_absolute_correlation(golden_A3, current_A3)

        assert correlation > 0.98, f"A3 embedding correlation is {correlation:.4f}, expected >0.98"

    def test_embedding_shapes_match(self, golden_references, current_run):
        """Test that embedding shapes match golden references."""
        for key in ["A1_embedding", "A2_embedding", "A3_embedding"]:
            golden_shape = golden_references[key].shape
            current_shape = current_run[key].shape

            assert (
                golden_shape == current_shape
            ), f"{key} shape mismatch: golden {golden_shape}, current {current_shape}"
