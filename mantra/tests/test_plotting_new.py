"""Tests for new plotting functions."""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pytest

matplotlib.use("Agg")  # Non-interactive backend for testing


@pytest.fixture
def model_with_metadata(trained_mantra_model):
    """Trained model with sample-level metadata added."""
    from mantra.analysis.cache import add_metadata, setup_cache

    setup_cache(trained_mantra_model)
    groups = ["A", "B"] * 5
    add_metadata(trained_mantra_model, "group", groups)
    scores = np.random.randn(10)
    add_metadata(trained_mantra_model, "score", scores)
    return trained_mantra_model


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test to avoid memory leaks."""
    yield
    plt.close("all")


class TestVarianceExplained:
    def test_returns_figure(self, trained_mantra_model):
        """Test that variance_explained returns a Figure."""
        from mantra.plotting.embeddings import variance_explained

        fig = variance_explained(trained_mantra_model)
        assert isinstance(fig, plt.Figure)

    def test_with_top(self, trained_mantra_model):
        """Test variance_explained with top parameter."""
        from mantra.plotting.embeddings import variance_explained

        fig = variance_explained(trained_mantra_model, top=2)
        assert isinstance(fig, plt.Figure)

    def test_with_figsize(self, trained_mantra_model):
        """Test variance_explained with custom figsize."""
        from mantra.plotting.embeddings import variance_explained

        fig = variance_explained(trained_mantra_model, figsize=(10, 5))
        assert isinstance(fig, plt.Figure)


class TestFactorWeights:
    def test_single_factor(self, trained_mantra_model):
        """Test factor_weights for a single factor."""
        from mantra.plotting.embeddings import factor_weights

        fig = factor_weights(trained_mantra_model, factor_idx=0)
        assert isinstance(fig, plt.Figure)

    def test_multiple_factors(self, trained_mantra_model):
        """Test factor_weights for multiple factors."""
        from mantra.plotting.embeddings import factor_weights

        fig = factor_weights(trained_mantra_model, factor_idx=[0, 1])
        assert isinstance(fig, plt.Figure)

    def test_with_top(self, trained_mantra_model):
        """Test factor_weights with top parameter."""
        from mantra.plotting.embeddings import factor_weights

        fig = factor_weights(trained_mantra_model, factor_idx=0, top=5)
        assert isinstance(fig, plt.Figure)

    def test_show_all_features(self, trained_mantra_model):
        """Test factor_weights with top=0 (show all features)."""
        from mantra.plotting.embeddings import factor_weights

        fig = factor_weights(trained_mantra_model, factor_idx=0, top=0)
        assert isinstance(fig, plt.Figure)


class TestSliceWeights:
    def test_single_factor(self, trained_mantra_model):
        """Test slice_weights for a single factor."""
        from mantra.plotting.embeddings import slice_weights

        fig = slice_weights(trained_mantra_model, factor_idx=0)
        assert isinstance(fig, plt.Figure)

    def test_multiple_factors(self, trained_mantra_model):
        """Test slice_weights for multiple factors."""
        from mantra.plotting.embeddings import slice_weights

        fig = slice_weights(trained_mantra_model, factor_idx=[0, 1, 2])
        assert isinstance(fig, plt.Figure)


class TestScatter:
    def test_basic_scatter(self, trained_mantra_model):
        """Test basic scatter plot of two factors."""
        from mantra.plotting.embeddings import scatter

        fig = scatter(trained_mantra_model, x=0, y=1)
        assert isinstance(fig, plt.Figure)

    def test_scatter_with_color(self, model_with_metadata):
        """Test scatter plot colored by metadata."""
        from mantra.plotting.embeddings import scatter

        fig = scatter(model_with_metadata, x=0, y=1, color="group")
        assert isinstance(fig, plt.Figure)

    def test_scatter_by_name(self, trained_mantra_model):
        """Test scatter plot using factor names."""
        from mantra.plotting.embeddings import scatter

        fig = scatter(trained_mantra_model, x="Factor_0", y="Factor_1")
        assert isinstance(fig, plt.Figure)

    def test_scatter_missing_color(self, trained_mantra_model):
        """Test scatter with non-existent color column raises error."""
        from mantra.analysis.cache import setup_cache
        from mantra.plotting.embeddings import scatter

        setup_cache(trained_mantra_model)
        with pytest.raises(KeyError, match="not found"):
            scatter(trained_mantra_model, x=0, y=1, color="nonexistent")


class TestClustermap:
    def test_basic(self, trained_mantra_model):
        """Test basic clustermap."""
        import seaborn as sns

        from mantra.plotting.embeddings import clustermap

        g = clustermap(trained_mantra_model)
        assert isinstance(g, sns.matrix.ClusterGrid)

    def test_with_factor_idx(self, trained_mantra_model):
        """Test clustermap with subset of factors."""
        from mantra.plotting.embeddings import clustermap

        g = clustermap(trained_mantra_model, factor_idx=[0, 1])
        assert g is not None


class TestGroupPlots:
    def test_stripplot(self, model_with_metadata):
        """Test stripplot returns a Figure."""
        from mantra.plotting.embeddings import stripplot

        fig = stripplot(model_with_metadata, factor_idx=0, groupby="group")
        assert isinstance(fig, plt.Figure)

    def test_boxplot(self, model_with_metadata):
        """Test boxplot returns a Figure."""
        from mantra.plotting.embeddings import boxplot

        fig = boxplot(model_with_metadata, factor_idx=0, groupby="group")
        assert isinstance(fig, plt.Figure)

    def test_violinplot(self, model_with_metadata):
        """Test violinplot returns a Figure."""
        from mantra.plotting.embeddings import violinplot

        fig = violinplot(model_with_metadata, factor_idx=0, groupby="group")
        assert isinstance(fig, plt.Figure)

    def test_multiple_factors(self, model_with_metadata):
        """Test group plots with multiple factors."""
        from mantra.plotting.embeddings import boxplot

        fig = boxplot(model_with_metadata, factor_idx=[0, 1], groupby="group")
        assert isinstance(fig, plt.Figure)

    def test_missing_groupby(self, trained_mantra_model):
        """Test group plot with missing groupby raises error."""
        from mantra.analysis.cache import setup_cache
        from mantra.plotting.embeddings import stripplot

        setup_cache(trained_mantra_model)
        with pytest.raises(KeyError, match="not found"):
            stripplot(trained_mantra_model, factor_idx=0, groupby="nonexistent")
