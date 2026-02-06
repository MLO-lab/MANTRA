"""Seed management utilities for reproducibility."""

import logging

import numpy as np
import pyro
import torch

logger = logging.getLogger(__name__)


def set_all_seeds(seed: int = 42) -> None:
    """Set all random seeds for reproducibility.

    MANTRA uses multiple RNGs that ALL must be controlled for
    reproducible results: PyTorch, NumPy, and Pyro.

    Parameters
    ----------
    seed : int, optional
        Random seed value, by default 42

    Example
    -------
    >>> from mantra.utils import set_all_seeds
    >>> set_all_seeds(42)
    """
    logger.debug("Setting all random seeds to %d", seed)

    torch.manual_seed(seed)
    np.random.seed(seed)
    pyro.set_rng_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Ensure deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def clear_pyro_state() -> None:
    """Clear Pyro's global parameter store.

    Should be called before training to ensure a clean start.
    """
    logger.debug("Clearing Pyro parameter store")
    pyro.clear_param_store()
