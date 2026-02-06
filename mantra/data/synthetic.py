"""Synthetic data generation for MANTRA."""

import logging

import numpy as np
import torch
import torch.nn.functional as F

from mantra.utils.gpu import get_free_gpu_idx

logger = logging.getLogger(__name__)


class DataGenerator:
    """Generate synthetic tensor data for testing and benchmarking.

    Creates synthetic 3rd-order tensors with known factor structure
    for validating MANTRA's decomposition capabilities.

    Parameters
    ----------
    n_samples : int, optional
        Number of samples (first dimension), by default 100
    n_features : int, optional
        Number of features (third dimension), by default 12
    n_drugs : int, optional
        Number of drugs/slices (second dimension), by default 5
    R : int, optional
        Tensor rank (number of factors), by default 10
    a : float, optional
        Concentration parameter for factor covariance, by default 1.0
    b : float, optional
        Rate parameter for factor covariance, by default 0.5
    use_gpu : bool, optional
        Whether to use GPU if available, by default True
    device : str, optional
        Specific device to use, by default None (auto-select)

    Attributes
    ----------
    A1 : torch.Tensor
        Sample factor matrix (n_samples x R)
    A2 : torch.Tensor
        Slice factor matrix (n_drugs x R)
    A3 : torch.Tensor
        Feature factor matrix (n_features x R)
    Y : torch.Tensor
        Generated tensor (n_samples x n_drugs x n_features)

    Example
    -------
    >>> from mantra.data import DataGenerator
    >>> gen = DataGenerator(n_samples=100, n_drugs=10, n_features=50, R=5)
    >>> gen.generate(seed=42)
    >>> data = gen.get_sim_data()
    >>> print(data["Y_sim"].shape)
    torch.Size([100, 10, 50])
    """

    def __init__(
        self,
        n_samples: int = 100,
        n_features: int = 12,
        n_drugs: int = 5,
        R: int = 10,
        a: float = 1.0,
        b: float = 0.5,
        use_gpu: bool = True,
        device: str | None = None,
    ) -> None:
        self.n_samples = n_samples
        self.n_features = n_features
        self.n_drugs = n_drugs
        self.R = R
        self.a = a
        self.b = b

        # Set device
        self.device = device
        if device is None:
            self.device = torch.device("cpu")
            if use_gpu and torch.cuda.is_available():
                logger.info("GPU available, running computations on GPU.")
                self.device = f"cuda:{get_free_gpu_idx()}"

        # Initialize factor matrices
        self.A1: torch.Tensor | None = None
        self.A2: torch.Tensor | None = None
        self.A3: torch.Tensor | None = None
        self.Y: torch.Tensor | None = None

    def generate(
        self,
        seed: int | None = 0,
        cluster_std: int = 1,
        center_box: bool = False,
    ) -> np.random.Generator:
        """Generate synthetic tensor data.

        Creates factor matrices A1, A2, A3 from multivariate normal
        distributions and generates a noisy tensor Y = sum_r A1_r * A2_r * A3_r + noise.

        Parameters
        ----------
        seed : int, optional
            Random seed for reproducibility, by default 0
        cluster_std : int, optional
            Standard deviation for clusters (unused), by default 1
        center_box : bool, optional
            Whether to center (unused), by default False

        Returns
        -------
        np.random.Generator
            The random number generator used
        """
        rng = np.random.default_rng()

        if seed is not None:
            rng = np.random.default_rng(seed)

        logger.debug(
            "Generating synthetic data: %d samples, %d drugs, %d features, R=%d",
            self.n_samples,
            self.n_drugs,
            self.n_features,
            self.R,
        )

        # Create covariance matrix for factor generation
        cov = torch.diag(torch.tensor([self.a] * self.R) / torch.tensor([self.b] * self.R))
        zeros = torch.zeros([self.R])

        # Generate factor matrices
        A1 = torch.Tensor(rng.multivariate_normal(zeros, cov, size=self.n_samples))
        A2 = torch.Tensor(rng.multivariate_normal(zeros, cov, size=self.n_drugs))
        A3 = torch.Tensor(rng.multivariate_normal(zeros, cov, size=self.n_features))

        # Generate tensor and add noise
        new_tensor = torch.einsum("ir,jr,kr->ijk", A1, A2, A3)
        X_df = torch.distributions.Normal(new_tensor, 1 / torch.sqrt(torch.Tensor([1]))).sample()

        # Move to device
        self.A1 = A1.to(self.device)
        self.A2 = A2.to(self.device)
        self.A3 = A3.to(self.device)
        self.Y = X_df.to(self.device)

        logger.debug("Generated tensor with shape %s", self.Y.shape)

        return rng

    def get_sim_data(self) -> dict[str, torch.Tensor]:
        """Get the generated simulation data.

        Returns
        -------
        dict
            Dictionary containing:
            - 'A1_sim': Sample factor matrix
            - 'A2_sim': Slice factor matrix
            - 'A3_sim': Feature factor matrix
            - 'Y_sim': Generated tensor
        """
        return {
            "A1_sim": self.A1,
            "A2_sim": self.A2,
            "A3_sim": self.A3,
            "Y_sim": self.Y,
        }

    def generate_missingness(
        self,
        p: float = 0.1,
        seed: int | None = None,
    ) -> None:
        """Introduce missing values into the generated tensor.

        Parameters
        ----------
        p : float, optional
            Proportion of values to set as missing, by default 0.1
        seed : int, optional
            Random seed (unused, uses PyTorch dropout), by default None
        """
        logger.debug("Generating %.1f%% missing values", p * 100)

        Y = F.dropout(self.Y, p=p)
        Y[Y == 0] = float("nan")
        self.Y = Y
