"""Pathway enrichment analysis for MANTRA.

Runs GSEA or ORA on feature embeddings (A3) using gseapy.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from mantra.model.core import MANTRA

logger = logging.getLogger(__name__)


def enrichment(
    model: MANTRA,
    gene_sets: str | dict | list[str],
    view: int | str | None = None,
    factor_idx: int | str | list | None = None,
    method: str = "gsea",
    top_n: int | None = None,
    **kwargs: Any,
) -> dict[str, pd.DataFrame]:
    """Run pathway enrichment on feature embeddings (A3).

    For each factor, ranks features by absolute loading magnitude,
    then runs GSEA (prerank) or ORA (overrepresentation) via gseapy.

    Parameters
    ----------
    model : MANTRA
        A trained MANTRA model.
    gene_sets : str, dict, or list of str
        Gene set specification. Can be:
        - str: path to GMT file or gseapy library name (e.g., "GO_Biological_Process_2021")
        - dict: mapping set names to gene lists
        - list of str: multiple library names
    view : int, str, optional
        Which view's features to analyze. If None, uses all features.
    factor_idx : int, str, list, optional
        Which factors to analyze. If None, analyzes all factors.
    method : str, optional
        Enrichment method: "gsea" (prerank) or "ora" (overrepresentation).
        By default "gsea".
    top_n : int, optional
        For ORA: number of top features per factor. By default None (uses
        top 10% of features).
    **kwargs
        Additional keyword arguments passed to gseapy.prerank or gseapy.enrich.

    Returns
    -------
    dict[str, pd.DataFrame]
        Dictionary mapping factor names to enrichment result DataFrames.

    Raises
    ------
    ImportError
        If gseapy is not installed.
    ValueError
        If method is not "gsea" or "ora".
    """
    try:
        import gseapy as gp
    except ImportError:
        raise ImportError(
            "gseapy is required for enrichment analysis. "
            "Install it with: pip install gseapy"
        )

    if method not in ("gsea", "ora"):
        raise ValueError(f"method must be 'gsea' or 'ora', got '{method}'")

    # Get feature embeddings
    A3 = model.get_feature_embeddings(view=view, factor_idx=factor_idx, as_df=True)
    factor_names = A3.columns.tolist()

    results = {}

    for factor_name in factor_names:
        loadings = A3[factor_name]

        if method == "gsea":
            # GSEA prerank: rank all features by loading
            ranked = loadings.abs().sort_values(ascending=False)
            # Preserve sign for directionality
            ranked_signed = loadings[ranked.index]

            rnk = pd.DataFrame({"gene": ranked_signed.index, "score": ranked_signed.values})

            try:
                res = gp.prerank(
                    rnk=rnk,
                    gene_sets=gene_sets,
                    no_plot=True,
                    **kwargs,
                )
                results[factor_name] = res.res2d
            except Exception as e:
                logger.warning("GSEA failed for %s: %s", factor_name, e)
                results[factor_name] = pd.DataFrame()

        else:  # ora
            # ORA: select top features
            abs_loadings = loadings.abs().sort_values(ascending=False)
            n = top_n if top_n is not None else max(1, len(abs_loadings) // 10)
            top_genes = abs_loadings.head(n).index.tolist()

            try:
                res = gp.enrich(
                    gene_list=top_genes,
                    gene_sets=gene_sets,
                    background=loadings.index.tolist(),
                    no_plot=True,
                    **kwargs,
                )
                results[factor_name] = res.res2d
            except Exception as e:
                logger.warning("ORA failed for %s: %s", factor_name, e)
                results[factor_name] = pd.DataFrame()

    logger.info("Enrichment analysis completed for %d factors", len(results))
    return results
