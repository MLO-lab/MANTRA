"""Scanpy-based embedding analysis for MANTRA.

Wraps scanpy functions to operate on the cached factor_adata (A1 embeddings).
All scanpy imports are lazy to avoid import-time conflicts with anndata.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from mantra.model.core import MANTRA

from mantra.analysis.cache import _get_cache

logger = logging.getLogger(__name__)


def neighbors(model: MANTRA, **kwargs: Any) -> None:
    """Compute neighbor graph on sample embeddings (A1).

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    **kwargs
        Additional keyword arguments passed to ``scanpy.pp.neighbors``.
    """
    import scanpy as sc

    cache = _get_cache(model)
    sc.pp.neighbors(cache.factor_adata, use_rep=cache.use_rep, **kwargs)
    logger.info("Computed neighbor graph")


def umap(model: MANTRA, **kwargs: Any) -> None:
    """Compute UMAP of sample embeddings.

    Automatically computes neighbors if not already present.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    **kwargs
        Additional keyword arguments passed to ``scanpy.tl.umap``.
    """
    import scanpy as sc

    cache = _get_cache(model)

    if "neighbors" not in cache.factor_adata.uns:
        logger.info("Computing neighbors before UMAP")
        neighbors(model)

    sc.tl.umap(cache.factor_adata, **kwargs)
    logger.info("Computed UMAP embedding")


def tsne(model: MANTRA, **kwargs: Any) -> None:
    """Compute tSNE of sample embeddings.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    **kwargs
        Additional keyword arguments passed to ``scanpy.tl.tsne``.
    """
    import scanpy as sc

    cache = _get_cache(model)
    sc.tl.tsne(cache.factor_adata, use_rep=cache.use_rep, **kwargs)
    logger.info("Computed tSNE embedding")


def leiden(model: MANTRA, **kwargs: Any) -> None:
    """Leiden clustering on sample embeddings.

    Automatically computes neighbors if not already present.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    **kwargs
        Additional keyword arguments passed to ``scanpy.tl.leiden``.
    """
    import scanpy as sc

    cache = _get_cache(model)

    if "neighbors" not in cache.factor_adata.uns:
        logger.info("Computing neighbors before Leiden clustering")
        neighbors(model)

    sc.tl.leiden(cache.factor_adata, **kwargs)
    logger.info("Computed Leiden clustering")


def rank(
    model: MANTRA,
    groupby: str,
    method: str = "wilcoxon",
    **kwargs: Any,
) -> None:
    """Rank factors by group using statistical tests.

    Wraps ``scanpy.tl.rank_genes_groups`` to identify factors that
    distinguish between sample groups.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    groupby : str
        Column name in cache metadata to group samples by.
    method : str, optional
        Statistical method for ranking. By default "wilcoxon".
    **kwargs
        Additional keyword arguments passed to ``scanpy.tl.rank_genes_groups``.
    """
    import scanpy as sc

    cache = _get_cache(model)

    if groupby not in cache.factor_adata.obs.columns:
        raise KeyError(
            f"'{groupby}' not found in metadata. "
            f"Available columns: {list(cache.factor_adata.obs.columns)}"
        )

    sc.tl.rank_genes_groups(
        cache.factor_adata,
        groupby=groupby,
        method=method,
        rankby_abs=True,
        **kwargs,
    )
    logger.info("Ranked factors by '%s' using %s", groupby, method)
