"""Tests for embedding analysis functions (neighbors, umap, tsne, leiden, rank)."""

import numpy as np
import pytest

sc = pytest.importorskip(
    "scanpy",
    reason="scanpy incompatible with current anndata version",
    exc_type=ImportError,
)


@pytest.fixture
def model_with_metadata(trained_mantra_model):
    """Trained model with sample-level metadata added."""
    from mantra.analysis.cache import add_metadata, setup_cache

    setup_cache(trained_mantra_model)
    # Add a categorical metadata column
    groups = ["A", "B"] * 5
    add_metadata(trained_mantra_model, "group", groups)
    return trained_mantra_model


class TestNeighbors:
    def test_neighbors_basic(self, trained_mantra_model):
        """Test that neighbors computes without error."""
        from mantra.analysis.cache import _get_cache
        from mantra.analysis.embedding import neighbors

        neighbors(trained_mantra_model)

        cache = _get_cache(trained_mantra_model)
        assert "neighbors" in cache.factor_adata.uns

    def test_neighbors_custom_params(self, trained_mantra_model):
        """Test neighbors with custom parameters."""
        from mantra.analysis.embedding import neighbors

        neighbors(trained_mantra_model, n_neighbors=5)


class TestUMAP:
    def test_umap_basic(self, trained_mantra_model):
        """Test that UMAP computes and stores result in obsm."""
        from mantra.analysis.cache import _get_cache
        from mantra.analysis.embedding import umap

        umap(trained_mantra_model)

        cache = _get_cache(trained_mantra_model)
        assert "X_umap" in cache.factor_adata.obsm
        assert cache.factor_adata.obsm["X_umap"].shape == (10, 2)

    def test_umap_auto_neighbors(self, trained_mantra_model):
        """Test that UMAP auto-computes neighbors."""
        from mantra.analysis.cache import _get_cache, setup_cache
        from mantra.analysis.embedding import umap

        setup_cache(trained_mantra_model, overwrite=True)
        umap(trained_mantra_model)

        cache = _get_cache(trained_mantra_model)
        assert "neighbors" in cache.factor_adata.uns
        assert "X_umap" in cache.factor_adata.obsm


class TestTSNE:
    def test_tsne_basic(self, trained_mantra_model):
        """Test that tSNE computes and stores result in obsm."""
        from mantra.analysis.cache import _get_cache
        from mantra.analysis.embedding import tsne

        # perplexity must be < n_samples (10)
        tsne(trained_mantra_model, perplexity=5)

        cache = _get_cache(trained_mantra_model)
        assert "X_tsne" in cache.factor_adata.obsm
        assert cache.factor_adata.obsm["X_tsne"].shape == (10, 2)


class TestLeiden:
    def test_leiden_basic(self, trained_mantra_model):
        """Test that Leiden clustering produces labels."""
        from mantra.analysis.cache import _get_cache
        from mantra.analysis.embedding import leiden

        leiden(trained_mantra_model)

        cache = _get_cache(trained_mantra_model)
        assert "leiden" in cache.factor_adata.obs.columns
        assert len(cache.factor_adata.obs["leiden"]) == 10

    def test_leiden_auto_neighbors(self, trained_mantra_model):
        """Test that Leiden auto-computes neighbors."""
        from mantra.analysis.cache import _get_cache, setup_cache
        from mantra.analysis.embedding import leiden

        setup_cache(trained_mantra_model, overwrite=True)
        leiden(trained_mantra_model)

        cache = _get_cache(trained_mantra_model)
        assert "neighbors" in cache.factor_adata.uns


class TestRank:
    def test_rank_basic(self, model_with_metadata):
        """Test that rank produces results."""
        from mantra.analysis.cache import _get_cache
        from mantra.analysis.embedding import neighbors, rank

        neighbors(model_with_metadata)
        rank(model_with_metadata, groupby="group")

        cache = _get_cache(model_with_metadata)
        assert "rank_genes_groups" in cache.factor_adata.uns

    def test_rank_missing_groupby(self, trained_mantra_model):
        """Test that rank raises with invalid groupby."""
        from mantra.analysis.cache import setup_cache
        from mantra.analysis.embedding import rank

        setup_cache(trained_mantra_model)
        with pytest.raises(KeyError, match="not found"):
            rank(trained_mantra_model, groupby="nonexistent")
