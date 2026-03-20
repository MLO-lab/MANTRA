"""Tensor preprocessing transformations for MANTRA.

Provides normalization, pseudo-bulk aggregation, and feature selection.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger(__name__)


def normalize(
    tensor: torch.Tensor,
    center: bool = True,
    scale: bool = True,
) -> torch.Tensor:
    """Center and scale a tensor along the feature dimension.

    Operates on a 3D tensor of shape (n_samples, n_slices, n_features).
    Centers by subtracting feature means (across samples and slices).
    Scales by dividing by the global standard deviation.

    NaN values are ignored during mean/std computation and preserved in output.

    Parameters
    ----------
    tensor : torch.Tensor
        3D input tensor of shape (n_samples, n_slices, n_features).
    center : bool, optional
        Whether to center by subtracting feature means. By default True.
    scale : bool, optional
        Whether to scale by dividing by global std. By default True.

    Returns
    -------
    torch.Tensor
        Normalized tensor of same shape.
    """
    if tensor.ndim != 3:
        raise ValueError(f"Expected 3D tensor, got {tensor.ndim}D")

    result = tensor.clone()

    # Create NaN mask
    nan_mask = torch.isnan(result)

    if center:
        # Compute per-feature mean (across samples and slices), ignoring NaN
        masked = result.clone()
        masked[nan_mask] = 0.0
        count = (~nan_mask).float().sum(dim=(0, 1))  # (n_features,)
        count = count.clamp(min=1)
        feature_mean = masked.sum(dim=(0, 1)) / count  # (n_features,)
        result = result - feature_mean.unsqueeze(0).unsqueeze(0)

    if scale:
        # Compute global std, ignoring NaN
        masked = result.clone()
        masked[nan_mask] = 0.0
        n_valid = (~nan_mask).float().sum()
        global_std = torch.sqrt((masked**2).sum() / n_valid.clamp(min=1))
        if global_std > 0:
            result = result / global_std

    # Restore NaN values
    result[nan_mask] = float("nan")

    logger.debug("Normalized tensor: center=%s, scale=%s", center, scale)
    return result


def pseudobulk(
    adata,
    sample_key: str,
    slice_key: str,
    agg_func: str = "mean",
    min_cells: int = 10,
    layer: str | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Aggregate single-cell data to pseudo-bulk tensor.

    Groups cells by (sample_key, slice_key), aggregates expression,
    and returns a 3D tensor (n_samples x n_slices x n_features).
    Groups with fewer than ``min_cells`` cells are set to NaN.

    Parameters
    ----------
    adata : anndata.AnnData
        Single-cell AnnData object.
    sample_key : str
        Column in ``adata.obs`` for sample grouping (e.g., "patient_id").
    slice_key : str
        Column in ``adata.obs`` for slice grouping (e.g., "cell_type").
    agg_func : str, optional
        Aggregation function: "mean" or "sum". By default "mean".
    min_cells : int, optional
        Minimum number of cells per group. Groups below this are NaN.
        By default 10.
    layer : str, optional
        Which layer to use. If None, uses ``adata.X``.

    Returns
    -------
    tensor : torch.Tensor
        3D tensor of shape (n_samples, n_slices, n_features).
    metadata : dict
        Dictionary containing sample_names, slice_names, feature_names.
    """
    import scipy.sparse

    if agg_func not in ("mean", "sum"):
        raise ValueError(f"agg_func must be 'mean' or 'sum', got '{agg_func}'")

    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if scipy.sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    samples = adata.obs[sample_key].values
    slices = adata.obs[slice_key].values

    unique_samples = sorted(set(samples))
    unique_slices = sorted(set(slices))
    feature_names = adata.var_names.tolist()

    sample_to_idx = {s: i for i, s in enumerate(unique_samples)}
    slice_to_idx = {s: i for i, s in enumerate(unique_slices)}

    n_samples = len(unique_samples)
    n_slices = len(unique_slices)
    n_features = X.shape[1]

    tensor = np.full((n_samples, n_slices, n_features), np.nan, dtype=np.float32)

    for si, sample in enumerate(unique_samples):
        for sli, slc in enumerate(unique_slices):
            mask = (samples == sample) & (slices == slc)
            n_cells = mask.sum()

            if n_cells < min_cells:
                continue

            group_X = X[mask]
            if agg_func == "mean":
                tensor[si, sli, :] = group_X.mean(axis=0)
            else:
                tensor[si, sli, :] = group_X.sum(axis=0)

    metadata = {
        "sample_names": unique_samples,
        "slice_names": unique_slices,
        "feature_names": feature_names,
    }

    n_valid = np.sum(~np.isnan(tensor[:, :, 0]))
    n_total = n_samples * n_slices
    logger.info(
        "Pseudo-bulk tensor: (%d, %d, %d), %d/%d valid groups (min_cells=%d)",
        n_samples,
        n_slices,
        n_features,
        n_valid,
        n_total,
        min_cells,
    )

    return torch.tensor(tensor), metadata


def highly_variable_features(
    adata,
    n_top: int = 2000,
    **kwargs: Any,
) -> np.ndarray:
    """Select highly variable features using scanpy.

    Thin wrapper around ``scanpy.pp.highly_variable_genes``.

    Parameters
    ----------
    adata : anndata.AnnData
        AnnData object.
    n_top : int, optional
        Number of top highly variable features. By default 2000.
    **kwargs
        Additional keyword arguments passed to
        ``scanpy.pp.highly_variable_genes``.

    Returns
    -------
    np.ndarray
        Boolean mask indicating highly variable features.
    """
    import scanpy as sc

    sc.pp.highly_variable_genes(adata, n_top_genes=n_top, **kwargs)
    return adata.var["highly_variable"].values
