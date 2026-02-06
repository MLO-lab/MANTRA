"""Core MANTRA model implementation.

This module contains the main MANTRA class for Bayesian tensor decomposition
and the underlying probabilistic model (MANTRAModel).
"""

import logging
from collections.abc import Callable
from typing import Any

import numpy as np
import pandas as pd
import pyro
import pyro.distributions as dist
import torch
from pyro.infer import config_enumerate
from pyro.infer.autoguide.guides import AutoNormal
from pyro.nn import PyroModule
from pyro.optim import Adam
from pyro.optim import ClippedAdam
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm

from mantra.utils.gpu import get_free_gpu_idx

logger = logging.getLogger(__name__)


class TensorDataset(Dataset):
    """PyTorch Dataset wrapper for tensor data."""

    def __init__(self, *tensors: torch.Tensor) -> None:
        self.tensors = tensors

    def __len__(self) -> int:
        return self.tensors[0].size(0)

    def __getitem__(self, index: int) -> tuple[int, tuple[torch.Tensor, ...]]:
        return index, tuple(tensor[index] for tensor in self.tensors)


class MANTRA(PyroModule):
    """MANTRA: Multi-view ANalysis with Tensor and matRix Alignment.

    A Bayesian probabilistic framework for integrating collections of tensors
    of different orders using variational inference.

    Parameters
    ----------
    observations : Union[torch.Tensor, List[torch.Tensor]]
        Input tensor(s) to decompose. Can be a single tensor or list of tensors.
    n_features : List[int]
        Number of features for each view.
    outcome_obs : torch.Tensor, optional
        Outcome observations for supervised learning, by default None
    p : int, optional
        Number of outcome classes, by default 1
    metadata : pd.DataFrame, optional
        Sample metadata, by default None
    index : str, optional
        Column name for sample indexing, by default None
    n_samples : int, optional
        Number of samples, by default 100
    R : int, optional
        Tensor rank (number of factors), by default 10
    a : float, optional
        Concentration parameter for tau prior, by default 1
    b : float, optional
        Rate parameter for tau prior, by default 0.5
    c : float, optional
        Concentration parameter for lambda prior, by default 1
    d : float, optional
        Rate parameter for lambda prior, by default 0.5
    covariates : torch.Tensor, optional
        Additional covariates, by default None
    view_names : List[str], optional
        Names for each view, by default None
    device : str, optional
        Device to use ('cpu' or 'cuda:X'), by default None (auto-select)
    use_gpu : bool, optional
        Whether to use GPU if available, by default True
    A2 : torch.Tensor, optional
        Fixed slice factor matrix, by default None
    A3 : torch.Tensor, optional
        Fixed feature factor matrix, by default None
    scale_factor : float, optional
        Scale factor for outcome loss, by default 1.1

    Attributes
    ----------
    tensor_data : torch.Tensor
        The processed input tensor
    device : torch.device
        The device being used for computation

    Example
    -------
    >>> from mantra import MANTRA, DataGenerator
    >>> gen = DataGenerator(n_samples=100, n_drugs=10, n_features=50, R=5)
    >>> gen.generate(seed=42)
    >>> data = gen.get_sim_data()
    >>> model = MANTRA(data["Y_sim"], n_features=[50], R=5)
    >>> history, _ = model.fit(n_epochs=1000, learning_rate=0.01)
    """

    def __init__(
        self,
        observations: torch.Tensor | list[torch.Tensor],
        n_features: list[int],
        outcome_obs: torch.Tensor | None = None,
        p: int = 1,
        metadata: pd.DataFrame | None = None,
        index: str | None = None,
        n_samples: int = 100,
        R: int = 10,
        a: float = 1,
        b: float = 0.5,
        c: float = 1,
        d: float = 0.5,
        covariates: torch.Tensor | None = None,
        view_names: list[str] | None = None,
        device: str | None = None,
        use_gpu: bool = True,
        A2: torch.Tensor | None = None,
        A3: torch.Tensor | None = None,
        scale_factor: float = 1.1,
    ) -> None:
        super().__init__(name="MANTRA")

        self.device = device
        torch.set_default_device(self.device)

        self.covariates = covariates
        self.view_names = view_names
        self.observations = observations
        self.metadata = metadata
        self.index = index
        self.n_samples = n_samples

        if device is None:
            self.device = torch.device("cpu")
            if use_gpu and torch.cuda.is_available():
                logger.info("GPU available, running computations on GPU.")
                self.device = f"cuda:{get_free_gpu_idx()}"

        self._model: MANTRAModel | None = None
        self._guide: AutoNormal | None = None
        self._built: bool = False
        self._trained: bool = False
        self._cache = None
        self._informed = None

        self.tensor_data: torch.Tensor | None = None

        self.outcome_obs = outcome_obs
        self.p = p

        self.n_features = n_features
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.R = R
        self.A2 = A2
        self.A3 = A3
        self.scale_factor = scale_factor

    def _setup_tensor_data(self) -> torch.Tensor:
        """Setup and preprocess tensor data from observations.

        Returns
        -------
        torch.Tensor
            The processed tensor data
        """
        if all(isinstance(x, torch.Tensor) for x in self.observations) and all(
            len(x.shape) == 3 for x in self.observations
        ):
            self.tensor_data = torch.cat(self.observations, 2)

        if all(isinstance(x, pd.DataFrame) for x in self.observations):
            if self.metadata is None:
                logger.warning("Metadata required for DataFrame input")
            else:
                list_tensor = []
                index_len = len(set(self.metadata[self.index]))
                drop_col = len(self.metadata.columns)
                for matrix in self.observations:
                    matrix = matrix.join(self.metadata)
                    X = np.array(
                        [
                            matrix[matrix[self.index] == i].iloc[:, :-drop_col].to_numpy()
                            for i in range(index_len)
                        ]
                    )
                    list_tensor.append(X)
                self.tensor_data = torch.cat(list_tensor, 2)

        self.tensor_data = self.tensor_data.to(self.device)
        logger.debug("Tensor data shape: %s", self.tensor_data.shape)

        return self.tensor_data

    def _setup_model_guide(
        self,
        obs: torch.Tensor | None = None,
        obs_outcome: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        A2: torch.Tensor | None = None,
        A3: torch.Tensor | None = None,
        scale_factor: float = 1.1,
    ) -> bool:
        """Setup the probabilistic model and variational guide.

        Parameters
        ----------
        obs : torch.Tensor, optional
            Observations
        obs_outcome : torch.Tensor, optional
            Outcome observations
        mask : torch.Tensor, optional
            Mask for missing values
        A2 : torch.Tensor, optional
            Fixed slice factors
        A3 : torch.Tensor, optional
            Fixed feature factors
        scale_factor : float, optional
            Scale factor for outcome

        Returns
        -------
        bool
            Whether the build was successful
        """
        if not self._built:
            self._model = MANTRAModel(
                self.tensor_data,
                self.n_features,
                self.metadata,
                p=self.p,
                R=self.R,
                a=self.a,
                b=self.b,
                c=self.c,
                d=self.d,
                device=self.device,
            )

            self._guide = AutoNormal(self._model, init_scale=0.01)
            logger.debug("Model and guide initialized")

            self._built = True
        return self._built

    def _setup_optimizer(
        self,
        batch_size: int,
        n_epochs: int,
        learning_rate: float,
        optimizer: str,
    ) -> pyro.optim.PyroOptim:
        """Setup the SVI optimizer.

        Parameters
        ----------
        batch_size : int
            Batch size for training
        n_epochs : int
            Number of training epochs
        learning_rate : float
            Learning rate
        optimizer : str
            Optimizer type ('adam' or 'clipped')

        Returns
        -------
        pyro.optim.PyroOptim
            The configured optimizer
        """
        logger.debug("Setting up optimizer: %s, lr=%f", optimizer, learning_rate)

        optim = Adam({"lr": learning_rate, "betas": (0.95, 0.999)})
        if optimizer.lower() == "clipped":
            n_iterations = int(n_epochs * (self.n_samples // batch_size))
            logger.debug("Decaying learning rate over %d iterations", n_iterations)
            gamma = 0.1
            lrd = gamma ** (1 / n_iterations)
            optim = ClippedAdam({"lr": learning_rate, "lrd": lrd})

        self._optimizer = optim
        return self._optimizer

    def _setup_svi(
        self,
        optimizer: pyro.optim.PyroOptim,
        n_particles: int,
        scale: bool = True,
    ) -> pyro.infer.SVI:
        """Setup stochastic variational inference.

        Parameters
        ----------
        optimizer : pyro.optim.PyroOptim
            The optimizer to use
        n_particles : int
            Number of particles for ELBO estimation
        scale : bool, optional
            Whether to scale ELBO by number of samples, by default True

        Returns
        -------
        pyro.infer.SVI
            Configured SVI object
        """
        # Note: scale parameter reserved for future use with scaled ELBO
        _ = scale  # Silence unused parameter warning

        svi = pyro.infer.SVI(
            model=self._model,
            guide=self._guide,
            optim=optimizer,
            loss=pyro.infer.Trace_ELBO(
                retain_graph=True,
                num_particles=n_particles,
                vectorize_particles=True,
            ),
        )
        self._svi = svi
        return self._svi

    def _setup_training_data(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor]:
        """Setup training data and masks.

        Returns
        -------
        tuple
            (train_obs, outcome_obs, mask_obs)
        """
        logger.debug("Setting up training data")
        train_obs = self.tensor_data
        train_obs = torch.nan_to_num(train_obs)

        logger.debug("Computing missing value mask")
        mask_obs = ~torch.isnan(train_obs)

        if self.outcome_obs is not None:
            outcome_obs = self.outcome_obs.to(self.device)
        else:
            outcome_obs = self.outcome_obs

        return train_obs, outcome_obs, mask_obs

    def fit(
        self,
        batch_size: int = 0,
        n_epochs: int = 1000,
        n_particles: int | None = None,
        learning_rate: float = 0.005,
        optimizer: str = "clipped",
        callbacks: list[Callable] | None = None,
        verbose: bool = True,
        seed: int | None = None,
    ) -> tuple[list[float], bool]:
        """Perform variational inference to fit the model.

        Parameters
        ----------
        batch_size : int, optional
            Batch size, by default 0 (use all samples)
        n_epochs : int, optional
            Number of training epochs, by default 1000
        n_particles : int, optional
            Number of particles for ELBO, by default 1000 // batch_size
        learning_rate : float, optional
            Learning rate, by default 0.005
        optimizer : str, optional
            Optimizer type ('adam' or 'clipped'), by default "clipped"
        callbacks : List[Callable], optional
            Callbacks to run each epoch, by default None
        verbose : bool, optional
            Whether to show progress bar, by default True
        seed : int, optional
            Random seed for training, by default None

        Returns
        -------
        tuple
            (loss_history, stopped_early)
        """
        # Setup tensor data
        if not isinstance(self.observations, torch.Tensor):
            self._setup_tensor_data()
        else:
            self.tensor_data = self.observations

        train_obs, outcome_obs, mask_obs = self._setup_training_data()

        # Handle batch size
        if batch_size is None or not (0 < batch_size <= self.n_samples):
            batch_size = self.n_samples

        if n_particles is None:
            n_particles = max(1, 1000 // batch_size)

        logger.info("Using %d particles", n_particles)
        logger.info("Preparing model and guide...")

        self._setup_model_guide(
            train_obs,
            outcome_obs,
            mask_obs,
            self.A2,
            self.A3,
            self.scale_factor,
        )

        logger.info("Preparing optimizer...")
        optimizer_obj = self._setup_optimizer(batch_size, n_epochs, learning_rate, optimizer)

        logger.info("Preparing SVI...")
        svi = self._setup_svi(optimizer_obj, n_particles, scale=True)

        logger.info("Preparing training data...")

        if batch_size < self.n_samples:
            logger.info("Using batches of size %d", batch_size)
            tensors = (train_obs, mask_obs)
            if self.covariates is not None:
                tensors += (self.covariates,)

            data_loader = DataLoader(
                TensorDataset(*tensors),
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,
                pin_memory=False,
                drop_last=False,
            )

            def _step() -> float:
                iteration_loss = 0.0
                for _, (sample_idx, batch_tensors) in enumerate(data_loader):
                    iteration_loss += svi.step(
                        sample_idx.to(self.device),
                        *[tensor.to(self.device) for tensor in batch_tensors],
                    )
                return iteration_loss

        else:
            logger.info("Using complete dataset")

            def _step() -> float:
                return svi.step(
                    train_obs.to(self.device),
                    outcome_obs,
                    mask_obs.to(self.device),
                    self.A2,
                    self.A3,
                    self.scale_factor,
                )

        self.seed = seed
        if seed is not None:
            logger.info("Setting training seed to %d", seed)
            pyro.set_rng_seed(seed)

        logger.info("Cleaning parameter store")
        pyro.enable_validation(True)
        pyro.clear_param_store()

        logger.info("Starting training...")
        stop_early = False
        history: list[float] = []
        pbar = range(n_epochs)

        if verbose:
            pbar = tqdm(pbar, desc="Training")
            window_size = 5

        for epoch_idx in pbar:
            epoch_loss = _step()
            history.append(epoch_loss)

            if (
                verbose
                and isinstance(pbar, tqdm)
                and (epoch_idx % window_size == 0 or epoch_idx == n_epochs - 1)
            ):
                pbar.set_postfix({"ELBO": f"{epoch_loss:.2f}"})

            if callbacks is not None:
                stop_early = any(callback(history) for callback in callbacks)
                if stop_early:
                    logger.info("Early stopping triggered at epoch %d", epoch_idx)
                    break

        self._trained = True
        logger.info("Training complete. Final ELBO: %.2f", history[-1])

        return history, stop_early

    def get_posterior(self) -> dict[str, torch.Tensor]:
        """Get posterior samples from the trained model.

        Returns
        -------
        dict
            Dictionary of posterior samples
        """
        if not self._trained:
            logger.warning("Model has not been trained yet")

        return self._guide.median()

    # -------------------------------------------------------------------------
    # Properties for accessing model dimensions and names
    # -------------------------------------------------------------------------

    @property
    def n_factors(self) -> int:
        """Number of factors (rank)."""
        return self.R

    @property
    def n_views(self) -> int:
        """Number of views."""
        return len(self.n_features)

    @property
    def factor_names(self) -> pd.Index:
        """Names of the factors."""
        if not hasattr(self, "_factor_names") or self._factor_names is None:
            self._factor_names = pd.Index([f"Factor_{i}" for i in range(self.R)])
        return self._factor_names

    @factor_names.setter
    def factor_names(self, value: list[str] | pd.Index) -> None:
        """Set factor names."""
        if len(value) != self.R:
            raise ValueError(f"Expected {self.R} factor names, got {len(value)}")
        self._factor_names = pd.Index(value)

    @property
    def sample_names(self) -> pd.Index:
        """Names of the samples."""
        if not hasattr(self, "_sample_names") or self._sample_names is None:
            n_samples = self.tensor_data.shape[0] if self.tensor_data is not None else 0
            self._sample_names = pd.Index([f"Sample_{i}" for i in range(n_samples)])
        return self._sample_names

    @sample_names.setter
    def sample_names(self, value: list[str] | pd.Index) -> None:
        """Set sample names."""
        self._sample_names = pd.Index(value)

    @property
    def feature_names(self) -> list[pd.Index]:
        """Names of the features for each view."""
        if not hasattr(self, "_feature_names") or self._feature_names is None:
            self._feature_names = [
                pd.Index([f"Feature_{v}_{i}" for i in range(n)])
                for v, n in enumerate(self.n_features)
            ]
        return self._feature_names

    @feature_names.setter
    def feature_names(self, value: list[list[str] | pd.Index]) -> None:
        """Set feature names for each view."""
        if len(value) != len(self.n_features):
            raise ValueError(f"Expected {len(self.n_features)} views, got {len(value)}")
        self._feature_names = [pd.Index(v) for v in value]

    @property
    def slice_names(self) -> pd.Index:
        """Names of the slices (e.g., cell types, conditions)."""
        if not hasattr(self, "_slice_names") or self._slice_names is None:
            n_slices = self.tensor_data.shape[1] if self.tensor_data is not None else 0
            self._slice_names = pd.Index([f"Slice_{i}" for i in range(n_slices)])
        return self._slice_names

    @slice_names.setter
    def slice_names(self, value: list[str] | pd.Index) -> None:
        """Set slice names."""
        self._slice_names = pd.Index(value)

    # -------------------------------------------------------------------------
    # Accessor methods for embeddings (raw factor matrices)
    # -------------------------------------------------------------------------

    def get_sample_embeddings(
        self,
        factor_idx: int | str | list | None = None,
        as_df: bool = False,
    ) -> torch.Tensor | pd.DataFrame:
        """Get sample embeddings (A1 matrix).

        The sample embeddings represent the sample-level latent factors,
        shared across all views.

        Parameters
        ----------
        factor_idx : int, str, list, optional
            Index of factors to return. If None, returns all factors.
        as_df : bool, optional
            Whether to return as DataFrame, by default False

        Returns
        -------
        torch.Tensor or pd.DataFrame
            Sample embeddings matrix of shape (n_samples, n_factors)
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before accessing embeddings")

        posterior = self._guide.median()
        A1 = posterior["A1"].detach().cpu()

        # Filter factors if requested
        if factor_idx is not None:
            factor_idx = self._normalize_factor_idx(factor_idx)
            A1 = A1[:, factor_idx]

        if as_df:
            cols = self.factor_names if factor_idx is None else self.factor_names[factor_idx]
            return pd.DataFrame(A1.numpy(), index=self.sample_names, columns=cols)

        return A1

    def get_slice_embeddings(
        self,
        factor_idx: int | str | list | None = None,
        as_df: bool = False,
    ) -> torch.Tensor | pd.DataFrame:
        """Get slice embeddings (A2 matrix).

        The slice embeddings represent the second mode of the tensor decomposition,
        typically corresponding to cell types, conditions, or treatments.

        Parameters
        ----------
        factor_idx : int, str, list, optional
            Index of factors to return. If None, returns all factors.
        as_df : bool, optional
            Whether to return as DataFrame, by default False

        Returns
        -------
        torch.Tensor or pd.DataFrame
            Slice embeddings matrix of shape (n_slices, n_factors)
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before accessing embeddings")

        posterior = self._guide.median()
        A2 = posterior["A2"].detach().cpu()

        # Filter factors if requested
        if factor_idx is not None:
            factor_idx = self._normalize_factor_idx(factor_idx)
            A2 = A2[:, factor_idx]

        if as_df:
            cols = self.factor_names if factor_idx is None else self.factor_names[factor_idx]
            return pd.DataFrame(A2.numpy(), index=self.slice_names, columns=cols)

        return A2

    def get_feature_embeddings(
        self,
        factor_idx: int | str | list | None = None,
        as_df: bool = False,
    ) -> torch.Tensor | pd.DataFrame:
        """Get feature embeddings (A3 matrix).

        The feature embeddings represent the feature-level latent factors
        (e.g., genes, peaks, drugs).

        Parameters
        ----------
        factor_idx : int, str, list, optional
            Index of factors to return. If None, returns all factors.
        as_df : bool, optional
            Whether to return as DataFrame, by default False

        Returns
        -------
        torch.Tensor or pd.DataFrame
            Feature embeddings matrix of shape (n_features, n_factors)
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before accessing embeddings")

        posterior = self._guide.median()
        A3 = posterior["A3"].detach().cpu()

        # Filter factors if requested
        if factor_idx is not None:
            factor_idx = self._normalize_factor_idx(factor_idx)
            A3 = A3[:, factor_idx]

        if as_df:
            # Concatenate all feature names
            all_features = pd.Index(
                [f for view_features in self.feature_names for f in view_features]
            )
            cols = self.factor_names if factor_idx is None else self.factor_names[factor_idx]
            return pd.DataFrame(A3.numpy(), index=all_features, columns=cols)

        return A3

    # -------------------------------------------------------------------------
    # Accessor methods for loadings (products of embeddings)
    # -------------------------------------------------------------------------

    def get_loadings(
        self,
        mode1: str = "slice",
        mode2: str = "feature",
        factor_idx: int | str | list | None = None,
        as_df: bool = False,
    ) -> torch.Tensor | pd.DataFrame:
        """Get factor loadings (product of two embedding matrices).

        Loadings are computed as the outer product of two embedding matrices.
        For example, slice-feature loadings show how each feature contributes
        to each factor for each slice (e.g., cell type x gene loadings).

        Parameters
        ----------
        mode1 : str, optional
            First mode: 'sample', 'slice', or 'feature'. By default 'slice'
        mode2 : str, optional
            Second mode: 'sample', 'slice', or 'feature'. By default 'feature'
        factor_idx : int, str, list, optional
            Index of factors to return. If None, returns all factors.
        as_df : bool, optional
            Whether to return as DataFrame (only for 2D case), by default False

        Returns
        -------
        torch.Tensor or pd.DataFrame
            Loadings tensor of shape (n_mode1, n_mode2, n_factors)
            For example, slice-feature loadings: (n_slices, n_features, n_factors)

        Examples
        --------
        >>> # Get cell type x gene loadings
        >>> loadings = model.get_loadings(mode1='slice', mode2='feature')
        >>> # Shape: (n_cell_types, n_genes, n_factors)

        >>> # Get sample x cell type loadings
        >>> loadings = model.get_loadings(mode1='sample', mode2='slice')
        >>> # Shape: (n_samples, n_cell_types, n_factors)
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before accessing loadings")

        # Get the embedding matrices
        mode_map = {
            "sample": ("A1", self.sample_names),
            "slice": ("A2", self.slice_names),
            "feature": ("A3", pd.Index([f for vf in self.feature_names for f in vf])),
        }

        if mode1 not in mode_map:
            raise ValueError(f"mode1 must be one of {list(mode_map.keys())}, got '{mode1}'")
        if mode2 not in mode_map:
            raise ValueError(f"mode2 must be one of {list(mode_map.keys())}, got '{mode2}'")
        if mode1 == mode2:
            raise ValueError("mode1 and mode2 must be different")

        posterior = self._guide.median()
        key1, names1 = mode_map[mode1]
        key2, names2 = mode_map[mode2]

        emb1 = posterior[key1].detach().cpu()
        emb2 = posterior[key2].detach().cpu()

        # Compute loadings: einsum('ir,jr->ijr', emb1, emb2)
        loadings = torch.einsum("ir,jr->ijr", emb1, emb2)

        # Filter factors if requested
        if factor_idx is not None:
            factor_idx = self._normalize_factor_idx(factor_idx)
            loadings = loadings[:, :, factor_idx]

        if as_df and loadings.shape[2] == 1:
            # Can only return DataFrame for single factor
            cols = names2
            return pd.DataFrame(
                loadings[:, :, 0].numpy(),
                index=names1,
                columns=cols,
            )

        return loadings

    # Aliases for backward compatibility
    def get_factor_scores(self, **kwargs) -> torch.Tensor | pd.DataFrame:
        """Alias for get_sample_embeddings(). See that method for documentation."""
        return self.get_sample_embeddings(**kwargs)

    def get_slice_factors(self, **kwargs) -> torch.Tensor | pd.DataFrame:
        """Alias for get_slice_embeddings(). See that method for documentation."""
        return self.get_slice_embeddings(**kwargs)

    # -------------------------------------------------------------------------
    # Other accessor methods
    # -------------------------------------------------------------------------

    def get_reconstructed(self) -> torch.Tensor:
        """Get the reconstructed tensor.

        Computes the tensor reconstruction from the learned factors:
        Y_hat = sum_r A1[:,r] ⊗ A2[:,r] ⊗ A3[:,r]

        Returns
        -------
        torch.Tensor
            Reconstructed tensor of same shape as input
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before accessing reconstruction")

        posterior = self._guide.median()
        A1 = posterior["A1"].detach().cpu()
        A2 = posterior["A2"].detach().cpu()
        A3 = posterior["A3"].detach().cpu()

        return torch.einsum("ir,jr,kr->ijk", A1, A2, A3)

    def get_embeddings(self) -> dict[str, torch.Tensor]:
        """Get all factor matrices as a dictionary.

        This is a convenience method that returns all learned factor matrices.

        Returns
        -------
        dict
            Dictionary with keys 'A1', 'A2', 'A3' containing factor matrices
        """
        if not self._trained:
            raise RuntimeError("Model must be trained before accessing embeddings")

        posterior = self._guide.median()
        return {
            "A1": posterior["A1"].detach().cpu(),
            "A2": posterior["A2"].detach().cpu(),
            "A3": posterior["A3"].detach().cpu(),
        }

    def _normalize_factor_idx(self, factor_idx: int | str | list) -> list[int] | np.ndarray:
        """Normalize factor index to list of integers.

        Parameters
        ----------
        factor_idx : int, str, or list
            Factor index specification

        Returns
        -------
        list or np.ndarray
            List of integer indices
        """
        if isinstance(factor_idx, int):
            return [factor_idx]
        if isinstance(factor_idx, str):
            if factor_idx in self.factor_names:
                return [self.factor_names.get_loc(factor_idx)]
            raise ValueError(f"Factor '{factor_idx}' not found")
        if isinstance(factor_idx, list | np.ndarray | pd.Index):
            result = []
            for idx in factor_idx:
                if isinstance(idx, int):
                    result.append(idx)
                elif isinstance(idx, str):
                    if idx in self.factor_names:
                        result.append(self.factor_names.get_loc(idx))
                    else:
                        raise ValueError(f"Factor '{idx}' not found")
            return result
        raise TypeError(f"Invalid factor_idx type: {type(factor_idx)}")


class MANTRAModel(PyroModule):
    """MANTRA probabilistic generative model.

    Implements the Bayesian tensor decomposition with Horseshoe priors
    for structured sparsity.

    Parameters
    ----------
    tensor_data : torch.Tensor
        The input tensor to decompose
    n_features : List[int]
        Number of features per view
    metadata : pd.DataFrame, optional
        Sample metadata
    p : int, optional
        Number of outcome classes
    R : int, optional
        Tensor rank
    a, b, c, d : float, optional
        Prior hyperparameters
    device : str, optional
        Computation device
    """

    def __init__(
        self,
        tensor_data: torch.Tensor,
        n_features: list[int],
        metadata: pd.DataFrame | None,
        p: int = 1,
        R: int = 10,
        a: float = 1,
        b: float = 0.5,
        c: float = 1,
        d: float = 0.5,
        device: str | None = None,
    ) -> None:
        super().__init__(name="MANTRAModel")

        self.tensor_data = tensor_data
        self.tensor_size = list(tensor_data.shape)
        self.metadata = metadata
        self.n_features = n_features
        self.feature_offsets = [0, *np.cumsum(self.n_features).tolist()]
        self.n_views = len(self.n_features)
        self.RegressionModel = None
        self.R = R
        self.a = a
        self.b = b
        self.c = c
        self.d = d
        self.p = p
        self.Ws: list = []
        self.beta_ms: list = []
        self.A3s: list = []
        self.output_dict: dict[str, torch.Tensor] = {}

        self.device = device
        self.to(self.device)

    def get_plate(self, name: str, **kwargs: Any) -> pyro.plate:
        """Get a Pyro plate for the specified dimension.

        Parameters
        ----------
        name : str
            Name of the plate ('rank', 'view', 'factor1', 'factor2', 'factor3', 'class')

        Returns
        -------
        pyro.plate
            The configured plate
        """
        plate_kwargs = {
            "rank": {"name": "R", "size": self.R, "dim": -1},
            "view": {"name": "view", "size": self.n_views, "dim": -2},
            "factor1": {"name": "samples", "size": self.tensor_size[0]},
            "factor2": {"name": "slices", "size": self.tensor_size[1]},
            "factor3": {"name": "features", "size": self.tensor_size[2]},
            "class": {"name": "class", "size": self.p},
        }
        return pyro.plate(device=self.device, **{**plate_kwargs[name], **kwargs})

    @config_enumerate
    def forward(
        self,
        obs: torch.Tensor | None = None,
        obs_outcome: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
        A2: torch.Tensor | None = None,
        A3: torch.Tensor | None = None,
        scale_factor: float = 1.1,
    ) -> dict[str, torch.Tensor]:
        """Generate samples from the model.

        Parameters
        ----------
        obs : torch.Tensor, optional
            Observations to condition on
        obs_outcome : torch.Tensor, optional
            Outcome observations
        mask : torch.Tensor, optional
            Missing value mask
        A2 : torch.Tensor, optional
            Fixed slice factors
        A3 : torch.Tensor, optional
            Fixed feature factors
        scale_factor : float, optional
            Scale for outcome loss

        Returns
        -------
        dict
            Dictionary of sampled values
        """
        rank_plate = self.get_plate("rank")
        view_plate = self.get_plate("view")
        factor1_plate = self.get_plate("factor1")
        factor2_plate = self.get_plate("factor2")
        factor3_plate = self.get_plate("factor3")

        # Sample noise precision
        tau = pyro.sample("tau", dist.InverseGamma(self.a, self.b))
        self.output_dict["tau"] = tau.to(self.device)

        # View-level shrinkage
        with view_plate:
            view_shrinkage = pyro.sample(
                "view_shrinkage",
                dist.HalfCauchy(torch.ones(1, device=self.device)),
            )
            self.output_dict["view_shrinkage"] = view_shrinkage.to(self.device)

        # Rank-level parameters
        with rank_plate:
            with view_plate:
                rank_scale = pyro.sample(
                    "rank_scale",
                    dist.HalfCauchy(torch.ones(1, device=self.device)),
                )
                self.output_dict["rank_scale"] = rank_scale.to(self.device)

            lmbda = pyro.sample("lmbda", dist.InverseGamma(self.c, self.d))
            self.output_dict["lmbda"] = lmbda.to(self.device)

            # Sample factor matrices
            with factor1_plate:
                a = pyro.sample(
                    "A1",
                    dist.Normal(
                        torch.zeros(self.R, device=self.device),
                        torch.ones(1, device=self.device),
                    ),
                )
                self.output_dict["A1"] = a.to(self.device)

            with factor2_plate:
                if A2 is not None:
                    self.output_dict["A2"] = pyro.deterministic("A2", A2).to(self.device)
                else:
                    self.output_dict["A2"] = pyro.sample(
                        "A2",
                        dist.Normal(
                            torch.zeros(self.R, device=self.device),
                            torch.ones(1, device=self.device),
                        ),
                    ).to(self.device)

            with factor3_plate:
                if A3 is not None:
                    self.output_dict["A3"] = pyro.deterministic("A3", A3).to(self.device)
                else:
                    local_scale = pyro.sample(
                        "local_scale",
                        dist.HalfCauchy(torch.ones(1, device=self.device)),
                    )
                    self.output_dict["local_scale"] = local_scale.to(self.device)

                    # Horseshoe prior variance
                    var = torch.cat(
                        [
                            (
                                self.output_dict["lmbda"]
                                / (
                                    self.output_dict["view_shrinkage"][i]
                                    * self.output_dict["rank_scale"][i]
                                    * self.output_dict["local_scale"][
                                        self.feature_offsets[i] : self.feature_offsets[i + 1]
                                    ]
                                )
                            )
                            for i in range(self.n_views)
                        ],
                        0,
                    )
                    self.output_dict["A3"] = pyro.sample(
                        "A3",
                        dist.Normal(torch.zeros(self.R, device=self.device), var),
                    ).to(self.device)

        # Reconstruct tensor and sample observations
        reconstruction = torch.einsum(
            "ir,jr,kr->ijk",
            self.output_dict["A1"],
            self.output_dict["A2"],
            self.output_dict["A3"],
        )

        if mask is None:
            logger.debug("No mask provided")
            self.output_dict["Y"] = pyro.sample(
                "Y",
                dist.Normal(
                    reconstruction,
                    1 / torch.sqrt(self.output_dict["tau"]),
                ).to_event(3),
                obs=obs,
                infer={"is_auxiliary": True},
            ).to(self.device)
        else:
            mask = mask.int()
            self.output_dict["Y"] = pyro.sample(
                "Y",
                dist.Normal(
                    reconstruction,
                    1 / torch.sqrt(self.output_dict["tau"]),
                )
                .mask(mask)
                .to_event(3),
                obs=obs,
                infer={"is_auxiliary": True},
            ).to(self.device)

        self.to(self.device)

        return self.output_dict
