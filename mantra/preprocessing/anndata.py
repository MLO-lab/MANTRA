"""AnnData and MuData loading utilities for MANTRA.

Constructs 3D tensors from long-format single-cell AnnData objects
by pivoting on sample and slice keys.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)


def from_anndata(
    adata,
    sample_key: str,
    slice_key: str,
    layer: str | None = None,
    feature_names: list[str] | None = None,
) -> tuple[torch.Tensor, dict[str, Any]]:
    """Construct a 3D tensor from a long-format AnnData.

    Pivots AnnData into shape (n_samples x n_slices x n_features).
    Missing sample x slice combinations are filled with NaN.

    Parameters
    ----------
    adata : anndata.AnnData
        Input AnnData object where each row is one observation (e.g., one
        cell or one sample-slice combination).
    sample_key : str
        Column in ``adata.obs`` identifying samples (e.g., "patient_id").
    slice_key : str
        Column in ``adata.obs`` identifying slices (e.g., "cell_type").
    layer : str, optional
        Which layer to use. If None, uses ``adata.X``.
    feature_names : list of str, optional
        Subset of ``adata.var_names`` to include. If None, uses all features.

    Returns
    -------
    tensor : torch.Tensor
        3D tensor of shape (n_samples, n_slices, n_features).
    metadata : dict
        Dictionary containing:
        - ``'sample_names'``: list of sample identifiers
        - ``'slice_names'``: list of slice identifiers
        - ``'feature_names'``: list of feature names
        - ``'sample_metadata'``: pd.DataFrame with one row per unique sample
    """
    import scipy.sparse

    if sample_key not in adata.obs.columns:
        raise KeyError(f"sample_key '{sample_key}' not found in adata.obs")
    if slice_key not in adata.obs.columns:
        raise KeyError(f"slice_key '{slice_key}' not found in adata.obs")

    # Get expression matrix
    if layer is not None:
        X = adata.layers[layer]
    else:
        X = adata.X

    if scipy.sparse.issparse(X):
        X = X.toarray()
    X = np.asarray(X, dtype=np.float32)

    # Subset features if requested
    if feature_names is not None:
        feature_mask = adata.var_names.isin(feature_names)
        X = X[:, feature_mask]
        var_names = adata.var_names[feature_mask].tolist()
    else:
        var_names = adata.var_names.tolist()

    samples = adata.obs[sample_key].values
    slices = adata.obs[slice_key].values

    unique_samples = sorted(set(samples))
    unique_slices = sorted(set(slices))

    sample_to_idx = {s: i for i, s in enumerate(unique_samples)}
    slice_to_idx = {s: i for i, s in enumerate(unique_slices)}

    n_samples = len(unique_samples)
    n_slices = len(unique_slices)
    n_features = X.shape[1]

    # Build tensor (fill missing with NaN)
    tensor = np.full((n_samples, n_slices, n_features), np.nan, dtype=np.float32)

    for obs_idx in range(len(adata)):
        si = sample_to_idx[samples[obs_idx]]
        sli = slice_to_idx[slices[obs_idx]]

        if not np.isnan(tensor[si, sli, 0]):
            # Average duplicate entries
            tensor[si, sli, :] = (tensor[si, sli, :] + X[obs_idx]) / 2.0
            logger.warning(
                "Duplicate entry for sample=%s, slice=%s. Averaging.",
                samples[obs_idx],
                slices[obs_idx],
            )
        else:
            tensor[si, sli, :] = X[obs_idx]

    # Build sample metadata (one row per unique sample)
    sample_meta_rows = []
    for s in unique_samples:
        mask = samples == s
        row = adata.obs.loc[mask].iloc[0].to_dict()
        sample_meta_rows.append(row)
    sample_metadata = pd.DataFrame(sample_meta_rows, index=unique_samples)
    # Drop the grouping columns from metadata
    sample_metadata = sample_metadata.drop(columns=[sample_key, slice_key], errors="ignore")

    metadata = {
        "sample_names": unique_samples,
        "slice_names": unique_slices,
        "feature_names": var_names,
        "sample_metadata": sample_metadata,
    }

    logger.info(
        "Created tensor of shape (%d, %d, %d) from AnnData",
        n_samples,
        n_slices,
        n_features,
    )

    return torch.tensor(tensor), metadata


def from_mudata(
    mdata,
    sample_key: str,
    slice_key: str,
    layer: str | None = None,
) -> tuple[list[torch.Tensor], dict[str, Any]]:
    """Construct tensors from multi-view MuData.

    Each modality becomes a separate view/tensor.

    Parameters
    ----------
    mdata : mudata.MuData
        Input MuData object with multiple modalities.
    sample_key : str
        Column in obs identifying samples.
    slice_key : str
        Column in obs identifying slices.
    layer : str, optional
        Which layer to use in each modality. If None, uses ``.X``.

    Returns
    -------
    tensors : list of torch.Tensor
        List of 3D tensors, one per modality.
    metadata : dict
        Dictionary containing:
        - ``'sample_names'``: list of sample identifiers (shared across views)
        - ``'slice_names'``: list of slice identifiers (shared across views)
        - ``'feature_names'``: list of lists (feature names per view)
        - ``'view_names'``: list of modality names
        - ``'sample_metadata'``: pd.DataFrame with one row per unique sample
    """
    tensors = []
    all_feature_names = []
    view_names = list(mdata.mod.keys())

    # First pass: get shared samples and slices
    all_samples = set()
    all_slices = set()
    for mod_name in view_names:
        mod = mdata.mod[mod_name]
        if sample_key in mod.obs.columns:
            all_samples.update(mod.obs[sample_key].unique())
            all_slices.update(mod.obs[slice_key].unique())
        elif sample_key in mdata.obs.columns:
            # Try global obs
            mod_mask = mdata.obs.index.isin(mod.obs.index)
            all_samples.update(mdata.obs.loc[mod_mask, sample_key].unique())
            all_slices.update(mdata.obs.loc[mod_mask, slice_key].unique())

    # Second pass: build tensors
    first_metadata = None
    for mod_name in view_names:
        mod = mdata.mod[mod_name]
        # Ensure obs has the required keys
        if sample_key not in mod.obs.columns and sample_key in mdata.obs.columns:
            mod.obs[sample_key] = mdata.obs.loc[mod.obs.index, sample_key]
        if slice_key not in mod.obs.columns and slice_key in mdata.obs.columns:
            mod.obs[slice_key] = mdata.obs.loc[mod.obs.index, slice_key]

        tensor, meta = from_anndata(mod, sample_key, slice_key, layer=layer)
        tensors.append(tensor)
        all_feature_names.append(meta["feature_names"])

        if first_metadata is None:
            first_metadata = meta

    metadata = {
        "sample_names": first_metadata["sample_names"],
        "slice_names": first_metadata["slice_names"],
        "feature_names": all_feature_names,
        "view_names": view_names,
        "sample_metadata": first_metadata["sample_metadata"],
    }

    logger.info("Created %d tensors from MuData", len(tensors))
    return tensors, metadata
