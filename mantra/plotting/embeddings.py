"""Embedding and factor visualization functions for MANTRA.

Provides scatter plots, heatmaps, strip/box/violin plots, UMAP/tSNE embeddings,
and variance-explained visualizations. All scanpy imports are lazy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

if TYPE_CHECKING:
    from mantra.model.core import MANTRA

from mantra.analysis.cache import _get_cache
from mantra.analysis.metrics import variance_explained as compute_r2

logger = logging.getLogger(__name__)


def variance_explained(
    model: MANTRA,
    top: int | None = None,
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """Heatmap of R-squared per factor.

    Single-view: horizontal bar chart sorted by R-squared.
    Multi-view: heatmap with views on y-axis, factors on x-axis.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    top : int, optional
        Show only the top N factors by R-squared. If None, shows all.
    figsize : tuple, optional
        Figure size. Auto-determined if None.
    **kwargs
        Additional keyword arguments passed to the plot function.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    r2 = compute_r2(model, factor_wise=True, view_wise=True, as_df=True)
    factor_r2 = r2["per_factor"]

    if top is not None:
        factor_r2 = factor_r2.head(top)

    if figsize is None:
        figsize = (max(6, len(factor_r2) * 0.5), 4)

    if "per_view" in r2 and model.n_views > 1:
        # Multi-view: build a heatmap matrix (views x factors)
        # Recompute per-view per-factor R²
        Y = model.tensor_data.cpu().numpy()
        Y_hat = model.get_reconstructed().numpy()
        embeddings = model.get_embeddings()
        A1 = embeddings["A1"].numpy()
        A2 = embeddings["A2"].numpy()
        A3 = embeddings["A3"].numpy()

        feature_offsets = [0, *np.cumsum(model.n_features).tolist()]
        factor_names = factor_r2.index.tolist()
        factor_indices = [model.factor_names.get_loc(f) for f in factor_names]

        view_names = model.view_names or [f"View {i}" for i in range(model.n_views)]
        heatmap_data = np.zeros((model.n_views, len(factor_names)))

        for vi in range(model.n_views):
            start, end = feature_offsets[vi], feature_offsets[vi + 1]
            Y_v = Y[:, :, start:end]
            ss_tot_v = np.nansum((Y_v - np.nanmean(Y_v)) ** 2)

            for fi, r in enumerate(factor_indices):
                Y_hat_r = np.einsum("i,j,k->ijk", A1[:, r], A2[:, r], A3[start:end, r])
                # Transpose to match (samples, slices, features)
                Y_hat_r = np.einsum("i,j,k->ijk", A1[:, r], A2[:, r], A3[start:end, r])
                ss_factor = np.nansum(Y_hat_r**2)
                heatmap_data[vi, fi] = ss_factor / ss_tot_v

        df_heatmap = pd.DataFrame(
            heatmap_data,
            index=view_names,
            columns=factor_names,
        )

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            df_heatmap,
            annot=True,
            fmt=".3f",
            cmap="Blues",
            ax=ax,
            **kwargs,
        )
        ax.set_title("Variance Explained (R²) per Factor and View")
        ax.set_ylabel("View")
        ax.set_xlabel("Factor")
    else:
        # Single-view: horizontal bar chart
        fig, ax = plt.subplots(figsize=figsize)
        ax.barh(
            range(len(factor_r2)),
            factor_r2["r2"].values,
            color="steelblue",
            **kwargs,
        )
        ax.set_yticks(range(len(factor_r2)))
        ax.set_yticklabels(factor_r2.index)
        ax.set_xlabel("R²")
        ax.set_title("Variance Explained per Factor")
        ax.invert_yaxis()

    plt.tight_layout()
    return fig


def factor_weights(
    model: MANTRA,
    factor_idx: int | str | list[int] | list[str],
    view: int | str | None = None,
    top: int = 25,
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """Heatmap of slice x feature loadings per factor.

    MANTRA-specific: shows the outer product A2[:,r] x A3[:,r] as a heatmap
    with slices on y-axis, features on x-axis.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    factor_idx : int, str, or list
        Factor(s) to visualize.
    view : int, str, optional
        View to use for features. If None and single-view, uses that view.
    top : int, optional
        Show only top features by absolute loading. Set to 0 to show all.
        By default 25.
    figsize : tuple, optional
        Figure size. Auto-determined if None.
    **kwargs
        Additional keyword arguments passed to ``sns.heatmap``.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    # Normalize factor_idx to list
    if isinstance(factor_idx, (int, str)):
        factor_idx = [factor_idx]

    resolved = model._normalize_factor_idx(factor_idx)

    A2 = model.get_slice_embeddings(as_df=True)
    A3 = model.get_feature_embeddings(view=view, as_df=True)

    n_factors = len(resolved)
    if figsize is None:
        figsize = (max(8, min(top, A3.shape[0]) * 0.3) * n_factors, max(4, A2.shape[0] * 0.4))

    fig, axes = plt.subplots(1, n_factors, figsize=figsize, squeeze=False)

    for i, r in enumerate(resolved):
        ax = axes[0, i]
        factor_name = model.factor_names[r]

        a2 = A2.iloc[:, [r]].values  # (n_slices, 1)
        a3 = A3.iloc[:, [r]].values  # (n_features, 1)

        # Outer product: slices x features
        loading_matrix = a2 @ a3.T  # (n_slices, n_features)

        feature_names = A3.index
        if top > 0 and top < len(feature_names):
            # Select top features by max absolute loading across slices
            max_abs = np.abs(loading_matrix).max(axis=0)
            top_idx = np.argsort(max_abs)[-top:][::-1]
            loading_matrix = loading_matrix[:, top_idx]
            feature_names = feature_names[top_idx]

        df = pd.DataFrame(
            loading_matrix,
            index=A2.index,
            columns=feature_names,
        )

        sns.heatmap(
            df,
            center=0,
            cmap="RdBu_r",
            ax=ax,
            **kwargs,
        )
        ax.set_title(factor_name)
        ax.set_ylabel("Slice" if i == 0 else "")
        ax.set_xlabel("Feature")

    plt.tight_layout()
    return fig


def slice_weights(
    model: MANTRA,
    factor_idx: int | str | list[int] | list[str],
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """Bar chart of slice embeddings (A2) per factor.

    Shows how each slice (cell type, condition) loads onto selected factors.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    factor_idx : int, str, or list
        Factor(s) to visualize.
    figsize : tuple, optional
        Figure size. Auto-determined if None.
    **kwargs
        Additional keyword arguments passed to ``ax.bar``.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    if isinstance(factor_idx, (int, str)):
        factor_idx = [factor_idx]

    A2 = model.get_slice_embeddings(factor_idx=factor_idx, as_df=True)

    n_factors = A2.shape[1]
    if figsize is None:
        figsize = (max(6, A2.shape[0] * 0.6), 3 * n_factors)

    fig, axes = plt.subplots(n_factors, 1, figsize=figsize, squeeze=False)

    for i, col in enumerate(A2.columns):
        ax = axes[i, 0]
        values = A2[col].values
        colors = ["steelblue" if v >= 0 else "coral" for v in values]
        ax.bar(range(len(values)), values, color=colors, **kwargs)
        ax.set_xticks(range(len(values)))
        ax.set_xticklabels(A2.index, rotation=45, ha="right")
        ax.set_ylabel("Loading")
        ax.set_title(col)
        ax.axhline(0, color="gray", linewidth=0.5)

    plt.tight_layout()
    return fig


def scatter(
    model: MANTRA,
    x: int | str,
    y: int | str,
    color: str | None = None,
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """Scatter plot of two sample embedding factors.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    x : int or str
        Factor index or name for x-axis.
    y : int or str
        Factor index or name for y-axis.
    color : str, optional
        Metadata column name for coloring points. Must be in cache metadata.
    figsize : tuple, optional
        Figure size. By default (6, 5).
    **kwargs
        Additional keyword arguments passed to ``ax.scatter`` or ``sns.scatterplot``.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    A1 = model.get_sample_embeddings(as_df=True)

    x_name = x if isinstance(x, str) else model.factor_names[x]
    y_name = y if isinstance(y, str) else model.factor_names[y]

    if figsize is None:
        figsize = (6, 5)

    fig, ax = plt.subplots(figsize=figsize)

    if color is not None:
        cache = _get_cache(model)
        if color not in cache.factor_adata.obs.columns:
            raise KeyError(
                f"'{color}' not found in metadata. "
                f"Available: {list(cache.factor_adata.obs.columns)}"
            )
        hue = cache.factor_adata.obs[color]
        sns.scatterplot(
            x=A1[x_name],
            y=A1[y_name],
            hue=hue,
            ax=ax,
            **kwargs,
        )
    else:
        ax.scatter(A1[x_name], A1[y_name], **kwargs)

    ax.set_xlabel(x_name)
    ax.set_ylabel(y_name)
    ax.set_title(f"{x_name} vs {y_name}")

    plt.tight_layout()
    return fig


def embedding(
    model: MANTRA,
    color: str | list[str] | None = None,
    method: str = "umap",
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """UMAP or tSNE plot of sample embeddings.

    Auto-computes the embedding if not already in cache.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    color : str or list of str, optional
        Metadata column(s) for coloring. Must be in cache metadata.
    method : str, optional
        Dimensionality reduction method: "umap" or "tsne". By default "umap".
    figsize : tuple, optional
        Figure size. By default (6, 5).
    **kwargs
        Additional keyword arguments passed to ``scanpy.pl.umap`` or
        ``scanpy.pl.tsne``.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    import scanpy as sc

    from mantra.analysis.embedding import tsne as compute_tsne
    from mantra.analysis.embedding import umap as compute_umap

    cache = _get_cache(model)

    if method == "umap":
        if "X_umap" not in cache.factor_adata.obsm:
            compute_umap(model)
        fig = sc.pl.umap(cache.factor_adata, color=color, show=False, return_fig=True, **kwargs)
    elif method == "tsne":
        if "X_tsne" not in cache.factor_adata.obsm:
            compute_tsne(model)
        fig = sc.pl.tsne(cache.factor_adata, color=color, show=False, return_fig=True, **kwargs)
    else:
        raise ValueError(f"method must be 'umap' or 'tsne', got '{method}'")

    return fig


def clustermap(
    model: MANTRA,
    factor_idx: int | str | list | None = None,
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,
):
    """Hierarchical clustering heatmap of sample embeddings.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    factor_idx : int, str, list, optional
        Factors to include. If None, uses all factors.
    figsize : tuple, optional
        Figure size. By default (8, 10).
    **kwargs
        Additional keyword arguments passed to ``sns.clustermap``.

    Returns
    -------
    sns.matrix.ClusterGrid
        The seaborn ClusterGrid object.
    """
    A1 = model.get_sample_embeddings(factor_idx=factor_idx, as_df=True)

    if figsize is None:
        figsize = (max(6, A1.shape[1] * 0.5), max(6, A1.shape[0] * 0.15))

    g = sns.clustermap(
        A1,
        cmap="RdBu_r",
        center=0,
        figsize=figsize,
        **kwargs,
    )
    g.fig.suptitle("Sample Embeddings (A1)", y=1.02)

    return g


def _groupplot(
    model: MANTRA,
    factor_idx: int | str | list[int] | list[str],
    groupby: str,
    pl_type: str = "stripplot",
    groups: list[str] | None = None,
    figsize: tuple[int, int] | None = None,
    **kwargs: Any,
) -> plt.Figure:
    """Shared implementation for strip/box/violin plots of A1 by metadata group.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    factor_idx : int, str, or list
        Factor(s) to plot.
    groupby : str
        Metadata column name to group samples by.
    pl_type : str
        Plot type: "stripplot", "boxplot", or "violinplot".
    groups : list of str, optional
        Subset of groups to show. If None, shows all.
    figsize : tuple, optional
        Figure size.
    **kwargs
        Additional keyword arguments passed to the seaborn plot function.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    if isinstance(factor_idx, (int, str)):
        factor_idx = [factor_idx]

    A1 = model.get_sample_embeddings(factor_idx=factor_idx, as_df=True)
    cache = _get_cache(model)

    if groupby not in cache.factor_adata.obs.columns:
        raise KeyError(
            f"'{groupby}' not found in metadata. "
            f"Available: {list(cache.factor_adata.obs.columns)}"
        )

    group_values = cache.factor_adata.obs[groupby]
    if groups is not None:
        mask = group_values.isin(groups)
        A1 = A1.loc[mask]
        group_values = group_values.loc[mask]

    n_factors = A1.shape[1]
    if figsize is None:
        figsize = (max(6, group_values.nunique() * 0.8), 3.5 * n_factors)

    fig, axes = plt.subplots(n_factors, 1, figsize=figsize, squeeze=False)

    plot_fn = {
        "stripplot": sns.stripplot,
        "boxplot": sns.boxplot,
        "violinplot": sns.violinplot,
    }[pl_type]

    for i, col in enumerate(A1.columns):
        ax = axes[i, 0]
        plot_fn(x=group_values, y=A1[col], ax=ax, **kwargs)
        ax.set_ylabel(col)
        ax.set_xlabel(groupby)
        ax.set_title(f"{col} by {groupby}")

    plt.tight_layout()
    return fig


def stripplot(
    model: MANTRA,
    factor_idx: int | str | list[int] | list[str],
    groupby: str,
    **kwargs: Any,
) -> plt.Figure:
    """Strip plot of sample embeddings grouped by metadata.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    factor_idx : int, str, or list
        Factor(s) to plot.
    groupby : str
        Metadata column name to group samples by.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    return _groupplot(model, factor_idx, groupby, pl_type="stripplot", **kwargs)


def boxplot(
    model: MANTRA,
    factor_idx: int | str | list[int] | list[str],
    groupby: str,
    **kwargs: Any,
) -> plt.Figure:
    """Box plot of sample embeddings grouped by metadata.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    factor_idx : int, str, or list
        Factor(s) to plot.
    groupby : str
        Metadata column name to group samples by.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    return _groupplot(model, factor_idx, groupby, pl_type="boxplot", **kwargs)


def violinplot(
    model: MANTRA,
    factor_idx: int | str | list[int] | list[str],
    groupby: str,
    **kwargs: Any,
) -> plt.Figure:
    """Violin plot of sample embeddings grouped by metadata.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    factor_idx : int, str, or list
        Factor(s) to plot.
    groupby : str
        Metadata column name to group samples by.
    **kwargs
        Additional keyword arguments.

    Returns
    -------
    plt.Figure
        The matplotlib figure.
    """
    return _groupplot(model, factor_idx, groupby, pl_type="violinplot", **kwargs)
