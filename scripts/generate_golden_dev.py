#!/usr/bin/env python
"""Generate golden reference outputs from dev branch for comparison."""
import logging
import pickle
import sys
from pathlib import Path

import numpy as np
import pyro
import torch

# Add paths for the dev branch structure
sys.path.insert(0, str(Path(__file__).parent.parent / "multi_parafac"))
sys.path.insert(0, str(Path(__file__).parent.parent / "multi_parafac" / "parafac"))

from models import PARAFAC
from synthetic import DataGenerator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Same configuration as the main golden references
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
    """Generate golden reference outputs."""
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

    logger.info("Creating PARAFAC model...")
    pyro.clear_param_store()

    model = PARAFAC(
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
    rmse = torch.sqrt(
        torch.mean((reconstruction - sim_data["Y_sim"].cpu()) ** 2)
    ).item()

    logger.info(f"Final ELBO: {loss_history[-1]}")
    logger.info(f"Reconstruction RMSE: {rmse}")

    # Build golden reference dictionary
    golden = {
        "config": config,
        "input_tensor": sim_data["Y_sim"].cpu(),
        "true_A1": sim_data["A1_sim"].cpu(),
        "true_A2": sim_data["A2_sim"].cpu(),
        "true_A3": sim_data["A3_sim"].cpu(),
        "A1_embedding": A1,
        "A2_embedding": A2,
        "A3_embedding": A3,
        "reconstruction": reconstruction,
        "reconstruction_rmse": rmse,
        "elbo_history": loss_history,
        "final_elbo": loss_history[-1],
    }

    return golden


def main():
    """Main function to generate and save golden references."""
    output_path = Path(__file__).parent.parent / "tests" / "golden_dev_references.pkl"
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logger.info("Generating golden references from dev branch...")
    golden = generate_golden_references()

    logger.info(f"Saving to {output_path}...")
    with open(output_path, "wb") as f:
        pickle.dump(golden, f)

    logger.info("Golden dev references generated successfully!")
    logger.info(f"  - Final ELBO: {golden['final_elbo']}")
    logger.info(f"  - Reconstruction RMSE: {golden['reconstruction_rmse']}")


if __name__ == "__main__":
    main()
