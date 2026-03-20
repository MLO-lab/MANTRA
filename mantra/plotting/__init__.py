"""Plotting utilities for MANTRA (accessible as ``mantra.pl``).

- :func:`distplots` -- distributions of factor matrices
- :func:`plot_elbo` -- ELBO training history
- :func:`plot_rmse_comparison` -- RMSE comparison across methods
- :func:`variance_explained` -- R² heatmap/bar chart
- :func:`factor_weights` -- slice × feature loading heatmaps
- :func:`slice_weights` -- slice embedding bar charts
- :func:`scatter` -- factor scatter plots
- :func:`embedding` -- UMAP/tSNE plots
- :func:`clustermap` -- hierarchical clustering heatmap
- :func:`stripplot` / :func:`boxplot` / :func:`violinplot` -- group plots

Example
-------
>>> import mantra
>>> mantra.pl.plot_elbo(history)
>>> mantra.pl.variance_explained(model)
>>> mantra.pl.scatter(model, x=0, y=1, color="cell_type")
"""

from mantra.plotting.embeddings import boxplot
from mantra.plotting.embeddings import clustermap
from mantra.plotting.embeddings import embedding
from mantra.plotting.embeddings import factor_weights
from mantra.plotting.embeddings import scatter
from mantra.plotting.embeddings import slice_weights
from mantra.plotting.embeddings import stripplot
from mantra.plotting.embeddings import variance_explained
from mantra.plotting.embeddings import violinplot
from mantra.plotting.factors import distplots
from mantra.plotting.factors import plot_elbo
from mantra.plotting.factors import plot_rmse_comparison

__all__ = [
    "boxplot",
    "clustermap",
    "distplots",
    "embedding",
    "factor_weights",
    "plot_elbo",
    "plot_rmse_comparison",
    "scatter",
    "slice_weights",
    "stripplot",
    "variance_explained",
    "violinplot",
]
