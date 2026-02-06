#!/usr/bin/env python
"""Generate golden reference outputs for numerical consistency testing.

This script generates reference outputs that are used to verify that code
refactoring does not change the numerical behavior of the model.

Run this script ONCE before making any code changes, and commit the
generated golden_references.pkl to the repository.

Usage:
    python scripts/generate_golden_references.py
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configuration for golden reference generation
GOLDEN_REF_CONFIG = {
    "seed": 42,
    "n_samples": 50,
    "n_drugs": 10,
    "n_features": 20,
    "R": 5,
    "n_epochs": 100,
    "learning_rate": 0.01,
    "n_particles": 1,
}


def generate_golden_references() -> dict:
    """Generate golden reference outputs.

    Returns
    -------
    dict
        Dictionary containing all golden reference values
    """
    config = GOLDEN_REF_CONFIG

    logger.info("Setting random seeds...")
    set_all_seeds(config["seed"])

    logger.info("Generating synthetic data...")
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

    logger.info("Creating MANTRA model...")
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

    logger.info("Training model...")
    loss_history, _ = model.fit(
        n_epochs=config["n_epochs"],
        n_particles=config["n_particles"],
        learning_rate=config["learning_rate"],
        optimizer="adam",
        verbose=True,
        seed=config["seed"],
    )

    logger.info("Extracting posterior samples...")
    samples = model._guide.median()

    # Compute reconstruction
    A1 = samples["A1"].detach().cpu()
    A2 = samples["A2"].detach().cpu()
    A3 = samples["A3"].detach().cpu()
    reconstruction = torch.einsum("ir,jr,kr->ijk", A1, A2, A3)

    # Compute RMSE
    rmse = torch.sqrt(torch.mean((reconstruction - sim_data["Y_sim"].cpu()) ** 2)).item()

    logger.info(f"Final ELBO: {loss_history[-1]}")
    logger.info(f"Reconstruction RMSE: {rmse}")

    # Build golden reference dictionary
    golden = {
        # Configuration
        "config": config,
        # Input data
        "input_tensor": sim_data["Y_sim"].cpu(),
        "true_A1": sim_data["A1_sim"].cpu(),
        "true_A2": sim_data["A2_sim"].cpu(),
        "true_A3": sim_data["A3_sim"].cpu(),
        # Learned embeddings
        "A1_embedding": A1,
        "A2_embedding": A2,
        "A3_embedding": A3,
        # Reconstruction
        "reconstruction": reconstruction,
        "reconstruction_rmse": rmse,
        # Training trajectory
        "elbo_history": loss_history,
        "final_elbo": loss_history[-1],
        # Checkpoints at specific epochs
        "elbo_epoch_10": loss_history[9] if len(loss_history) > 9 else None,
        "elbo_epoch_50": loss_history[49] if len(loss_history) > 49 else None,
    }

    return golden


def main():
    """Main function to generate and save golden references."""
    output_path = Path(__file__).parent.parent / "tests" / "golden_references.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating golden references...")
    golden = generate_golden_references()

    logger.info(f"Saving to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(golden, f)

    logger.info("Golden references generated successfully!")
    logger.info(f"  - Final ELBO: {golden['final_elbo']}")
    logger.info(f"  - Reconstruction RMSE: {golden['reconstruction_rmse']}")
    logger.info(f"  - A1 shape: {golden['A1_embedding'].shape}")
    logger.info(f"  - A2 shape: {golden['A2_embedding'].shape}")
    logger.info(f"  - A3 shape: {golden['A3_embedding'].shape}")


if __name__ == "__main__":
    main()
