"""Analysis utilities for MANTRA (accessible as ``mantra.tl``).

Key functions:

- :func:`variance_explained` -- R-squared decomposition (total, per-factor, per-view)
- :func:`test` -- association testing between factors and metadata
- :func:`filter_factors` -- select factors by cumulative R-squared
- :func:`rmse_loss` -- root mean squared error

Example
-------
>>> import mantra
>>> r2 = mantra.tl.variance_explained(model)
>>> associations = mantra.tl.test(model, metadata=sample_df)
"""

from mantra.analysis.cache import add_metadata
from mantra.analysis.cache import get_metadata
from mantra.analysis.cache import setup_cache
from mantra.analysis.embedding import leiden
from mantra.analysis.embedding import neighbors
from mantra.analysis.embedding import rank
from mantra.analysis.embedding import tsne
from mantra.analysis.embedding import umap
from mantra.analysis.enrichment import enrichment
from mantra.analysis.metrics import accuracy_outcome
from mantra.analysis.metrics import compute_silhouette_score
from mantra.analysis.metrics import filter_factors
from mantra.analysis.metrics import rmse_loss
from mantra.analysis.metrics import test
from mantra.analysis.metrics import test_metadata
from mantra.analysis.metrics import variance_explained
from mantra.analysis.metrics import variance_explained_per_factor

__all__ = [
    "accuracy_outcome",
    "add_metadata",
    "compute_silhouette_score",
    "enrichment",
    "filter_factors",
    "get_metadata",
    "leiden",
    "neighbors",
    "rank",
    "rmse_loss",
    "setup_cache",
    "test",
    "test_metadata",
    "tsne",
    "umap",
    "variance_explained",
    "variance_explained_per_factor",
]
