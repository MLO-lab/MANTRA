#!/usr/bin/env python
"""Validate numerical consistency after code changes.

This script provides a quick way to verify that code changes have not
affected numerical outputs. It runs a subset of consistency checks
and provides clear pass/fail output.

Usage:
    python scripts/validate_consistency.py

Run this script AFTER EVERY code change to ensure numerical consistency.
"""
import logging
import pickle
import sys
from pathlib import Path

import pyro
import torch

# Add the package to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from mantra import MANTRA
from mantra.data import DataGenerator
from mantra.utils.seeds import set_all_seeds

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def compute_absolute_correlation(a: torch.Tensor, b: torch.Tensor) -> float:
    """Compute absolute correlation between two tensors."""
    a_flat = a.flatten().float()
    b_flat = b.flatten().float()
    a_centered = a_flat - a_flat.mean()
    b_centered = b_flat - b_flat.mean()
    numerator = (a_centered * b_centered).sum()
    denominator = torch.sqrt((a_centered**2).sum() * (b_centered**2).sum())
    if denominator < 1e-8:
        return 0.0
    return abs((numerator / denominator).item())


def main():
    """Run consistency validation."""
    # Load golden references
    golden_path = Path(__file__).parent.parent / "tests" / "golden_references.pkl"

    if not golden_path.exists():
        logger.error("=" * 60)
        logger.error("ERROR: Golden references not found!")
        logger.error(f"Expected at: {golden_path}")
        logger.error("")
        logger.error("Run the following command first:")
        logger.error("  python scripts/generate_golden_references.py")
        logger.error("=" * 60)
        sys.exit(1)

    logger.info("=" * 60)
    logger.info("MANTRA Numerical Consistency Validation")
    logger.info("=" * 60)

    logger.info("\nLoading golden references...")
    with open(golden_path, "rb") as f:
        golden = pickle.load(f)

    config = golden["config"]

    logger.info("Generating current outputs...")
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

    # Run checks
    logger.info("\n" + "-" * 60)
    logger.info("Running consistency checks...")
    logger.info("-" * 60)

    all_passed = True

    # Check 1: ELBO within 1%
    golden_elbo = golden["final_elbo"]
    current_elbo = loss_history[-1]
    elbo_diff = abs(current_elbo - golden_elbo) / abs(golden_elbo) * 100

    if elbo_diff < 1.0:
        logger.info(f"[PASS] ELBO: {elbo_diff:.2f}% difference (< 1% threshold)")
    else:
        logger.error(f"[FAIL] ELBO: {elbo_diff:.2f}% difference (>= 1% threshold)")
        logger.error(f"       Golden: {golden_elbo:.4f}, Current: {current_elbo:.4f}")
        all_passed = False

    # Check 2: RMSE within 0.5%
    golden_rmse = golden["reconstruction_rmse"]
    rmse_diff = abs(rmse - golden_rmse) / golden_rmse * 100

    if rmse_diff < 0.5:
        logger.info(f"[PASS] RMSE: {rmse_diff:.2f}% difference (< 0.5% threshold)")
    else:
        logger.error(f"[FAIL] RMSE: {rmse_diff:.2f}% difference (>= 0.5% threshold)")
        logger.error(f"       Golden: {golden_rmse:.4f}, Current: {rmse:.4f}")
        all_passed = False

    # Check 3-5: Embedding correlations
    for name, golden_key, current_val in [
        ("A1", "A1_embedding", A1),
        ("A2", "A2_embedding", A2),
        ("A3", "A3_embedding", A3),
    ]:
        corr = compute_absolute_correlation(golden[golden_key], current_val)
        if corr > 0.98:
            logger.info(f"[PASS] {name} correlation: {corr:.4f} (> 0.98 threshold)")
        else:
            logger.error(f"[FAIL] {name} correlation: {corr:.4f} (<= 0.98 threshold)")
            all_passed = False

    # Summary
    logger.info("\n" + "=" * 60)
    if all_passed:
        logger.info("ALL CHECKS PASSED - Numerical consistency maintained!")
        logger.info("=" * 60)
        sys.exit(0)
    else:
        logger.error("SOME CHECKS FAILED - Numerical consistency broken!")
        logger.error("")
        logger.error("This means your code changes have altered the model's")
        logger.error("numerical behavior. Either:")
        logger.error("  1. Revert the changes if this was unintentional")
        logger.error("  2. If intentional, regenerate golden references:")
        logger.error("     python scripts/generate_golden_references.py")
        logger.error("=" * 60)
        sys.exit(1)


if __name__ == "__main__":
    main()
