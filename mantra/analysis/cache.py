"""Cache for downstream analysis results.

Bridges MANTRA's tensor world with scanpy's AnnData ecosystem by wrapping
sample embeddings (A1) into an AnnData object.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mantra.model.core import MANTRA

logger = logging.getLogger(__name__)


class Cache:
    """Cache for downstream analysis results.

    Wraps A1 (sample embeddings) into an AnnData object that scanpy functions
    can operate on. Stores UMAP coords in .obsm, cluster labels in .obs,
    and factor-level metrics in .varm.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    """

    META_KEY = "metadata"

    def __init__(self, model: MANTRA) -> None:
        self.factor_adata = None
        self.uns: dict = {}
        self.use_rep = "X"
        self._setup(model)

    def _setup(self, model: MANTRA) -> None:
        """Create AnnData from A1 embeddings.

        Parameters
        ----------
        model : MANTRA
            A trained MANTRA model.
        """
        import anndata as ad

        A1 = model.get_sample_embeddings().numpy()

        self.factor_adata = ad.AnnData(
            X=A1.astype(np.float32),
            obs=pd.DataFrame(index=model.sample_names),
            var=pd.DataFrame(index=model.factor_names),
        )
        logger.info(
            "Cache initialized: %d samples, %d factors",
            A1.shape[0],
            A1.shape[1],
        )

    @property
    def factor_metadata(self) -> pd.DataFrame | None:
        """Return factor-level metadata DataFrame from .varm."""
        if self.factor_adata is None:
            return None
        return self.factor_adata.varm.get(self.META_KEY)

    def update_factor_metadata(self, scores: pd.DataFrame) -> None:
        """Merge new scores into factor-level metadata.

        Parameters
        ----------
        scores : pd.DataFrame
            DataFrame indexed by factor names with new columns to add.
        """
        existing = self.factor_metadata
        if existing is not None:
            merged = existing.copy()
            for col in scores.columns:
                merged[col] = scores[col]
            self.factor_adata.varm[self.META_KEY] = merged
        else:
            self.factor_adata.varm[self.META_KEY] = scores


def setup_cache(model: MANTRA, overwrite: bool = False) -> Cache:
    """Initialize or retrieve model cache.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    overwrite : bool, optional
        If True, recreate cache even if one exists. By default False.

    Returns
    -------
    Cache
        The model's cache instance.
    """
    if not model._trained:
        raise RuntimeError("Model must be trained before setting up cache")

    if model._cache is not None and not overwrite:
        logger.debug("Returning existing cache")
        return model._cache

    model._cache = Cache(model)
    return model._cache


def _get_cache(model: MANTRA) -> Cache:
    """Get existing cache or auto-create one.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.

    Returns
    -------
    Cache
        The model's cache instance.
    """
    if model._cache is None:
        return setup_cache(model)
    return model._cache


def add_metadata(
    model: MANTRA,
    name: str,
    values: pd.Series | np.ndarray | list,
    overwrite: bool = False,
) -> None:
    """Add sample-level metadata column to cache.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    name : str
        Name for the metadata column.
    values : pd.Series, np.ndarray, or list
        Metadata values, one per sample.
    overwrite : bool, optional
        If True, overwrite existing column. By default False.
    """
    cache = _get_cache(model)

    if name in cache.factor_adata.obs.columns and not overwrite:
        raise ValueError(
            f"Metadata column '{name}' already exists. Use overwrite=True to replace."
        )

    if isinstance(values, (np.ndarray, list)):
        if len(values) != cache.factor_adata.n_obs:
            raise ValueError(
                f"Length mismatch: got {len(values)} values for "
                f"{cache.factor_adata.n_obs} samples"
            )
        values = pd.Series(values, index=cache.factor_adata.obs_names)

    cache.factor_adata.obs[name] = values


def get_metadata(model: MANTRA, name: str) -> pd.Series:
    """Retrieve sample-level metadata column.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    name : str
        Name of the metadata column.

    Returns
    -------
    pd.Series
        Metadata values.
    """
    cache = _get_cache(model)

    if name not in cache.factor_adata.obs.columns:
        raise KeyError(f"Metadata column '{name}' not found in cache")

    return cache.factor_adata.obs[name]
