"""Tests for the cache system."""

import numpy as np
import pandas as pd
import pytest


class TestCache:
    """Tests for Cache creation and management."""

    def test_setup_cache(self, trained_mantra_model):
        """Test cache creation from trained model."""
        from mantra.analysis.cache import setup_cache

        cache = setup_cache(trained_mantra_model)

        assert cache is not None
        assert cache.factor_adata is not None
        assert cache.factor_adata.shape == (10, 3)  # 10 samples, 3 factors
        assert cache.use_rep == "X"

    def test_setup_cache_untrained(self, mantra_model):
        """Test that cache setup fails on untrained model."""
        from mantra.analysis.cache import setup_cache

        with pytest.raises(RuntimeError, match="trained"):
            setup_cache(mantra_model)

    def test_setup_cache_idempotent(self, trained_mantra_model):
        """Test that setup_cache returns same cache if already set up."""
        from mantra.analysis.cache import setup_cache

        cache1 = setup_cache(trained_mantra_model)
        cache2 = setup_cache(trained_mantra_model)
        assert cache1 is cache2

    def test_setup_cache_overwrite(self, trained_mantra_model):
        """Test that setup_cache with overwrite=True creates new cache."""
        from mantra.analysis.cache import setup_cache

        cache1 = setup_cache(trained_mantra_model)
        cache2 = setup_cache(trained_mantra_model, overwrite=True)
        assert cache1 is not cache2

    def test_factor_adata_content(self, trained_mantra_model):
        """Test that factor_adata contains correct data."""
        from mantra.analysis.cache import setup_cache

        cache = setup_cache(trained_mantra_model)
        A1 = trained_mantra_model.get_sample_embeddings().numpy()

        np.testing.assert_allclose(cache.factor_adata.X, A1, rtol=1e-5)

    def test_factor_adata_names(self, trained_mantra_model):
        """Test that factor_adata has correct index and columns."""
        from mantra.analysis.cache import setup_cache

        cache = setup_cache(trained_mantra_model)

        assert list(cache.factor_adata.obs_names) == list(trained_mantra_model.sample_names)
        assert list(cache.factor_adata.var_names) == list(trained_mantra_model.factor_names)

    def test_add_metadata(self, trained_mantra_model):
        """Test adding metadata to cache."""
        from mantra.analysis.cache import add_metadata, get_metadata

        values = np.random.choice(["A", "B", "C"], size=10)
        add_metadata(trained_mantra_model, "group", values)

        result = get_metadata(trained_mantra_model, "group")
        assert len(result) == 10
        assert list(result.values) == list(values)

    def test_add_metadata_series(self, trained_mantra_model):
        """Test adding metadata as pd.Series."""
        from mantra.analysis.cache import add_metadata, get_metadata, setup_cache

        setup_cache(trained_mantra_model)
        values = pd.Series(
            ["X", "Y"] * 5,
            index=trained_mantra_model.sample_names,
        )
        add_metadata(trained_mantra_model, "category", values)

        result = get_metadata(trained_mantra_model, "category")
        assert list(result.values) == ["X", "Y"] * 5

    def test_add_metadata_no_overwrite(self, trained_mantra_model):
        """Test that adding duplicate metadata raises without overwrite."""
        from mantra.analysis.cache import add_metadata

        add_metadata(trained_mantra_model, "col1", list(range(10)))
        with pytest.raises(ValueError, match="already exists"):
            add_metadata(trained_mantra_model, "col1", list(range(10)))

    def test_add_metadata_overwrite(self, trained_mantra_model):
        """Test that overwrite=True replaces metadata."""
        from mantra.analysis.cache import add_metadata, get_metadata

        add_metadata(trained_mantra_model, "col1", list(range(10)))
        add_metadata(trained_mantra_model, "col1", list(range(10, 20)), overwrite=True)

        result = get_metadata(trained_mantra_model, "col1")
        assert list(result.values) == list(range(10, 20))

    def test_add_metadata_length_mismatch(self, trained_mantra_model):
        """Test that mismatched length raises error."""
        from mantra.analysis.cache import add_metadata, setup_cache

        setup_cache(trained_mantra_model)
        with pytest.raises(ValueError, match="Length mismatch"):
            add_metadata(trained_mantra_model, "bad", [1, 2, 3])

    def test_get_metadata_missing(self, trained_mantra_model):
        """Test that getting non-existent metadata raises error."""
        from mantra.analysis.cache import get_metadata, setup_cache

        setup_cache(trained_mantra_model)
        with pytest.raises(KeyError, match="not found"):
            get_metadata(trained_mantra_model, "nonexistent")

    def test_get_cache_auto_creates(self, trained_mantra_model):
        """Test that _get_cache auto-creates cache if needed."""
        from mantra.analysis.cache import _get_cache

        trained_mantra_model._cache = None
        cache = _get_cache(trained_mantra_model)
        assert cache is not None
        assert trained_mantra_model._cache is cache

    def test_update_factor_metadata(self, trained_mantra_model):
        """Test updating factor-level metadata."""
        from mantra.analysis.cache import setup_cache

        cache = setup_cache(trained_mantra_model)
        scores = pd.DataFrame(
            {"r2": [0.1, 0.2, 0.3]},
            index=trained_mantra_model.factor_names,
        )
        cache.update_factor_metadata(scores)

        assert cache.factor_metadata is not None
        assert "r2" in cache.factor_metadata.columns
