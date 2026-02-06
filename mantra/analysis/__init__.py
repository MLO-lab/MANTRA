"""Analysis utilities for MANTRA."""

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
    "compute_silhouette_score",
    "filter_factors",
    "rmse_loss",
    "test",
    "test_metadata",
    "variance_explained",
    "variance_explained_per_factor",
]
