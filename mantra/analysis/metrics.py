"""Evaluation metrics for MANTRA."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy import stats
from sklearn.metrics import silhouette_score
from statsmodels.stats import multitest

if TYPE_CHECKING:
    from mantra.model.core import MANTRA

logger = logging.getLogger(__name__)


def rmse_loss(yhat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """Compute Root Mean Squared Error.

    Parameters
    ----------
    yhat : torch.Tensor
        Predicted values
    y : torch.Tensor
        True values

    Returns
    -------
    torch.Tensor
        RMSE value
    """
    return torch.sqrt(torch.mean((yhat - y) ** 2))


def accuracy_outcome(
    pred: torch.Tensor,
    ground_truth: torch.Tensor,
) -> float:
    """Compute classification accuracy.

    Parameters
    ----------
    pred : torch.Tensor
        Predicted probabilities or logits
    ground_truth : torch.Tensor
        Ground truth labels (one-hot encoded)

    Returns
    -------
    float
        Accuracy as percentage (0-100)
    """
    pred_labels = torch.argmax(pred, dim=1)
    true_labels = torch.argmax(ground_truth, dim=1)
    correct = (true_labels == pred_labels).sum().item()
    accuracy = (correct / len(ground_truth)) * 100
    logger.debug("Accuracy: %.2f%%", accuracy)
    return accuracy


def compute_silhouette_score(
    samples: pd.DataFrame,
    metadata_col: str,
    metadata_array: pd.Series,
) -> float:
    """Compute silhouette score for factor embeddings.

    Parameters
    ----------
    samples : pd.DataFrame
        Factor embeddings
    metadata_col : str
        Column name for metadata
    metadata_array : pd.Series
        Metadata values for each sample

    Returns
    -------
    float
        Silhouette score
    """
    factors = samples.copy()
    factors = factors.add_prefix("Factor")
    factors[metadata_col] = metadata_array.to_list()

    adata_samples = sc.AnnData(factors.iloc[:, :-1])
    adata_samples.obs = factors[[metadata_col]]

    cluster_labels = adata_samples.obs[metadata_col].astype("category").cat.codes
    data_matrix = adata_samples.X

    silhouette_avg = silhouette_score(data_matrix, cluster_labels)
    logger.debug("Silhouette score: %.4f", silhouette_avg)

    return silhouette_avg


# -----------------------------------------------------------------------------
# Variance Explained (R²)
# -----------------------------------------------------------------------------


def variance_explained(
    model: MANTRA,
    factor_wise: bool = True,
    view_wise: bool = True,
    as_df: bool = True,
) -> dict[str, np.ndarray | pd.DataFrame]:
    """Compute variance explained (R²) by the model.

    Calculates R² scores showing how much variance in the data is explained
    by each factor and/or by the total model.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model
    factor_wise : bool, optional
        Whether to compute R² for each factor separately, by default True
    view_wise : bool, optional
        Whether to compute R² per view (if multiple views), by default True
    as_df : bool, optional
        Whether to return results as DataFrame, by default True

    Returns
    -------
    dict
        Dictionary containing:
        - 'total': Total R² (scalar or per view)
        - 'per_factor': R² per factor (if factor_wise=True)

    Examples
    --------
    >>> from mantra import MANTRA
    >>> from mantra.analysis import variance_explained
    >>> model = MANTRA(...)
    >>> model.fit(n_epochs=100)
    >>> r2 = variance_explained(model)
    >>> print(r2['total'])  # Total variance explained
    >>> print(r2['per_factor'])  # Per-factor R²
    """
    if not model._trained:
        raise RuntimeError("Model must be trained before computing variance explained")

    # Get data and reconstruction
    Y = model.tensor_data.cpu().numpy()
    Y_hat = model.get_reconstructed().numpy()

    # Compute total R²
    ss_tot = np.nansum((Y - np.nanmean(Y)) ** 2)
    ss_res = np.nansum((Y - Y_hat) ** 2)
    total_r2 = 1.0 - (ss_res / ss_tot)

    result = {"total": total_r2}

    # Compute per-factor R²
    if factor_wise:
        embeddings = model.get_embeddings()
        A1 = embeddings["A1"].numpy()
        A2 = embeddings["A2"].numpy()
        A3 = embeddings["A3"].numpy()

        n_factors = A1.shape[1]
        factor_r2 = np.zeros(n_factors)

        for r in range(n_factors):
            # Reconstruction using only factor r
            Y_hat_r = np.einsum("i,j,k->ijk", A1[:, r], A2[:, r], A3[:, r])
            # Residual after removing this factor
            Y_residual = Y - Y_hat + Y_hat_r
            ss_res_r = np.nansum((Y - Y_residual) ** 2)
            # R² for this factor (proportion of variance captured)
            factor_r2[r] = 1.0 - (ss_res_r / ss_tot)

        if as_df:
            factor_r2 = pd.DataFrame(
                {"r2": factor_r2},
                index=model.factor_names,
            )
            factor_r2 = factor_r2.sort_values("r2", ascending=False)

        result["per_factor"] = factor_r2

    # Compute per-view R² if multiple views
    if view_wise and model.n_views > 1:
        view_r2 = {}
        feature_offsets = [0, *np.cumsum(model.n_features).tolist()]

        for v in range(model.n_views):
            start, end = feature_offsets[v], feature_offsets[v + 1]
            Y_view = Y[:, :, start:end]
            Y_hat_view = Y_hat[:, :, start:end]

            ss_tot_v = np.nansum((Y_view - np.nanmean(Y_view)) ** 2)
            ss_res_v = np.nansum((Y_view - Y_hat_view) ** 2)
            view_r2[f"view_{v}"] = 1.0 - (ss_res_v / ss_tot_v)

        result["per_view"] = view_r2

    return result


def variance_explained_per_factor(
    model: MANTRA,
    cumulative: bool = False,
) -> pd.DataFrame:
    """Compute variance explained per factor, sorted by importance.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model
    cumulative : bool, optional
        Whether to also return cumulative R², by default False

    Returns
    -------
    pd.DataFrame
        DataFrame with factor names as index and R² values as columns
    """
    r2 = variance_explained(model, factor_wise=True, view_wise=False, as_df=True)
    df = r2["per_factor"].copy()

    if cumulative:
        df["cumulative_r2"] = df["r2"].cumsum()

    return df


# -----------------------------------------------------------------------------
# Association Testing
# -----------------------------------------------------------------------------


def test(
    model: MANTRA,
    metadata: pd.DataFrame | pd.Series,
    factor_idx: int | str | list | None = None,
    method: str = "spearman",
    p_adj_method: str = "fdr_bh",
) -> pd.DataFrame:
    """Test association between factor scores and sample metadata.

    Performs statistical tests to identify which factors are significantly
    associated with sample metadata (e.g., cell type, treatment condition).

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model
    metadata : pd.DataFrame or pd.Series
        Sample metadata to test associations against.
        If DataFrame, tests each column separately.
        Index should match sample names.
    factor_idx : int, str, list, optional
        Factors to test. If None, tests all factors.
    method : str, optional
        Statistical test method:
        - 'spearman': Spearman correlation (for continuous metadata)
        - 'pearson': Pearson correlation (for continuous metadata)
        - 'kruskal': Kruskal-Wallis test (for categorical metadata)
        - 'anova': One-way ANOVA (for categorical metadata)
        By default 'spearman'
    p_adj_method : str, optional
        Method for multiple testing correction:
        - 'fdr_bh': Benjamini-Hochberg FDR
        - 'bonferroni': Bonferroni correction
        - 'holm': Holm-Bonferroni
        By default 'fdr_bh'

    Returns
    -------
    pd.DataFrame
        DataFrame with test results containing:
        - 'statistic': Test statistic
        - 'pvalue': Raw p-value
        - 'pvalue_adj': Adjusted p-value
        - 'significant': Boolean indicating significance (p_adj < 0.05)

    Examples
    --------
    >>> from mantra import MANTRA
    >>> from mantra.analysis import test
    >>> model = MANTRA(...)
    >>> model.fit(n_epochs=100)
    >>> # Test association with cell type
    >>> results = test(model, metadata=cell_types, method='kruskal')
    >>> print(results[results['significant']])
    """
    if not model._trained:
        raise RuntimeError("Model must be trained before testing associations")

    # Get factor scores
    factor_scores = model.get_factor_scores(factor_idx=factor_idx, as_df=True)

    # Convert metadata to DataFrame if Series
    if isinstance(metadata, pd.Series):
        metadata = metadata.to_frame()

    # Align indices
    common_idx = factor_scores.index.intersection(metadata.index)
    if len(common_idx) == 0:
        # Try to match by position if indices don't match
        if len(factor_scores) == len(metadata):
            logger.warning(
                "Sample indices don't match, aligning by position. "
                "Consider setting sample_names on the model."
            )
            metadata = metadata.copy()
            metadata.index = factor_scores.index
            common_idx = factor_scores.index
        else:
            raise ValueError(
                f"No common samples between factor scores ({len(factor_scores)}) "
                f"and metadata ({len(metadata)})"
            )

    factor_scores = factor_scores.loc[common_idx]
    metadata = metadata.loc[common_idx]

    results = []

    for meta_col in metadata.columns:
        meta_values = metadata[meta_col]

        # Determine if categorical or continuous
        is_categorical = (
            meta_values.dtype == "object"
            or meta_values.dtype.name == "category"
            or meta_values.nunique() < 10
        )

        for factor_name in factor_scores.columns:
            factor_values = factor_scores[factor_name].values

            # Remove NaN values
            mask = ~(np.isnan(factor_values) | pd.isna(meta_values))
            fv = factor_values[mask]
            mv = meta_values[mask]

            if len(fv) < 3:
                logger.warning(f"Skipping {factor_name} x {meta_col}: too few valid samples")
                continue

            # Perform test
            if is_categorical:
                if method in ("kruskal", "anova"):
                    groups = [fv[mv == cat] for cat in mv.unique() if len(fv[mv == cat]) > 0]
                    if len(groups) < 2:
                        continue
                    if method == "kruskal":
                        stat, pval = stats.kruskal(*groups)
                    else:  # anova
                        stat, pval = stats.f_oneway(*groups)
                else:
                    # Default to Kruskal for categorical
                    groups = [fv[mv == cat] for cat in mv.unique() if len(fv[mv == cat]) > 0]
                    if len(groups) < 2:
                        continue
                    stat, pval = stats.kruskal(*groups)
            else:
                if method == "spearman":
                    stat, pval = stats.spearmanr(fv, mv.astype(float))
                elif method == "pearson":
                    stat, pval = stats.pearsonr(fv, mv.astype(float))
                else:
                    stat, pval = stats.spearmanr(fv, mv.astype(float))

            results.append(
                {
                    "factor": factor_name,
                    "metadata": meta_col,
                    "statistic": stat,
                    "pvalue": pval,
                }
            )

    if not results:
        return pd.DataFrame(columns=["factor", "metadata", "statistic", "pvalue", "pvalue_adj"])

    df = pd.DataFrame(results)

    # Multiple testing correction
    _, pvals_adj, _, _ = multitest.multipletests(
        df["pvalue"].values,
        method=p_adj_method,
        alpha=0.05,
    )
    df["pvalue_adj"] = pvals_adj
    df["significant"] = df["pvalue_adj"] < 0.05

    # Sort by adjusted p-value
    df = df.sort_values("pvalue_adj")

    return df


def test_metadata(
    model: MANTRA,
    metadata: pd.DataFrame | pd.Series,
    **kwargs,
) -> pd.DataFrame:
    """Alias for test() function.

    See :func:`test` for documentation.
    """
    return test(model, metadata, **kwargs)


# -----------------------------------------------------------------------------
# Factor Filtering
# -----------------------------------------------------------------------------


def filter_factors(
    model: MANTRA,
    r2_thresh: float | int = 0.95,
    min_factors: int = 1,
) -> list[str]:
    """Filter factors based on cumulative variance explained.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model
    r2_thresh : float or int, optional
        If float < 1: threshold for cumulative R² (e.g., 0.95 = 95%)
        If int >= 1: number of top factors to keep
        By default 0.95
    min_factors : int, optional
        Minimum number of factors to return, by default 1

    Returns
    -------
    list[str]
        Names of factors to keep
    """
    r2_df = variance_explained_per_factor(model, cumulative=True)

    if isinstance(r2_thresh, float) and r2_thresh < 1:
        # Keep factors until cumulative R² reaches threshold
        mask = r2_df["cumulative_r2"] <= r2_thresh
        n_keep = max(mask.sum() + 1, min_factors)  # +1 to include the threshold-crossing factor
    else:
        # Keep top N factors
        n_keep = max(int(r2_thresh), min_factors)

    return r2_df.index[:n_keep].tolist()
