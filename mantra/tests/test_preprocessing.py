"""Tests for preprocessing functions."""

import numpy as np
import pandas as pd
import pytest
import torch


def _scanpy_available() -> bool:
    """Check if scanpy can be imported without errors."""
    try:
        import scanpy  # noqa: F401

        return True
    except ImportError:
        return False


class TestNormalize:
    def test_center_and_scale(self):
        """Test that normalize centers and scales."""
        from mantra.preprocessing.transform import normalize

        tensor = torch.randn(10, 5, 8)
        result = normalize(tensor, center=True, scale=True)

        assert result.shape == tensor.shape
        # After centering, feature means should be ~0
        feature_means = result.nanmean(dim=(0, 1))
        np.testing.assert_allclose(feature_means.numpy(), 0.0, atol=1e-5)

    def test_center_only(self):
        """Test centering without scaling."""
        from mantra.preprocessing.transform import normalize

        tensor = torch.randn(10, 5, 8) + 5.0  # Offset
        result = normalize(tensor, center=True, scale=False)

        feature_means = result.mean(dim=(0, 1))
        np.testing.assert_allclose(feature_means.numpy(), 0.0, atol=1e-5)

    def test_scale_only(self):
        """Test scaling without centering."""
        from mantra.preprocessing.transform import normalize

        tensor = torch.randn(10, 5, 8) * 10.0
        result = normalize(tensor, center=False, scale=True)

        # Global std should be ~1
        global_std = result.std().item()
        assert abs(global_std - 1.0) < 0.2

    def test_nan_handling(self):
        """Test that NaN values are preserved."""
        from mantra.preprocessing.transform import normalize

        tensor = torch.randn(10, 5, 8)
        tensor[0, 0, 0] = float("nan")
        tensor[3, 2, 5] = float("nan")

        result = normalize(tensor)

        assert torch.isnan(result[0, 0, 0])
        assert torch.isnan(result[3, 2, 5])
        # Non-NaN values should not be NaN
        non_nan_count = (~torch.isnan(result)).sum()
        assert non_nan_count == (~torch.isnan(tensor)).sum()

    def test_wrong_dims(self):
        """Test that 2D tensor raises error."""
        from mantra.preprocessing.transform import normalize

        with pytest.raises(ValueError, match="3D"):
            normalize(torch.randn(10, 5))

    def test_no_op(self):
        """Test with center=False, scale=False returns copy."""
        from mantra.preprocessing.transform import normalize

        tensor = torch.randn(10, 5, 8)
        result = normalize(tensor, center=False, scale=False)
        torch.testing.assert_close(result, tensor)


class TestFromAnndata:
    @pytest.fixture
    def simple_adata(self):
        """Create a simple AnnData for testing."""
        import anndata as ad

        n_obs = 20
        n_vars = 10

        obs = pd.DataFrame(
            {
                "sample": [f"S{i // 4}" for i in range(n_obs)],
                "slice": [f"SL{i % 4}" for i in range(n_obs)],
                "batch": ["B1"] * 10 + ["B2"] * 10,
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        )
        var = pd.DataFrame(index=[f"gene_{i}" for i in range(n_vars)])
        X = np.random.randn(n_obs, n_vars).astype(np.float32)

        return ad.AnnData(X=X, obs=obs, var=var)

    def test_basic(self, simple_adata):
        """Test basic tensor construction from AnnData."""
        from mantra.preprocessing.anndata import from_anndata

        tensor, meta = from_anndata(simple_adata, sample_key="sample", slice_key="slice")

        assert tensor.ndim == 3
        # 5 unique samples, 4 unique slices, 10 features
        assert tensor.shape == (5, 4, 10)
        assert len(meta["sample_names"]) == 5
        assert len(meta["slice_names"]) == 4
        assert len(meta["feature_names"]) == 10

    def test_feature_subset(self, simple_adata):
        """Test tensor construction with feature subset."""
        from mantra.preprocessing.anndata import from_anndata

        features = ["gene_0", "gene_1", "gene_2"]
        tensor, meta = from_anndata(
            simple_adata,
            sample_key="sample",
            slice_key="slice",
            feature_names=features,
        )

        assert tensor.shape[2] == 3
        assert meta["feature_names"] == features

    def test_sample_metadata(self, simple_adata):
        """Test that sample metadata is correctly extracted."""
        from mantra.preprocessing.anndata import from_anndata

        _, meta = from_anndata(simple_adata, sample_key="sample", slice_key="slice")

        assert isinstance(meta["sample_metadata"], pd.DataFrame)
        assert len(meta["sample_metadata"]) == 5

    def test_missing_key(self, simple_adata):
        """Test that missing key raises error."""
        from mantra.preprocessing.anndata import from_anndata

        with pytest.raises(KeyError, match="sample_key"):
            from_anndata(simple_adata, sample_key="nonexistent", slice_key="slice")

    def test_layer(self, simple_adata):
        """Test using a specific layer."""
        from mantra.preprocessing.anndata import from_anndata

        simple_adata.layers["log"] = np.log1p(np.abs(simple_adata.X))
        tensor, _ = from_anndata(
            simple_adata,
            sample_key="sample",
            slice_key="slice",
            layer="log",
        )
        assert tensor.ndim == 3


class TestPseudobulk:
    @pytest.fixture
    def sc_adata(self):
        """Create a single-cell AnnData for pseudo-bulk testing."""
        import anndata as ad

        n_cells = 200
        n_genes = 15
        np.random.seed(42)

        obs = pd.DataFrame(
            {
                "patient": np.random.choice(["P1", "P2", "P3"], size=n_cells),
                "cell_type": np.random.choice(["T", "B", "NK"], size=n_cells),
            },
            index=[f"cell_{i}" for i in range(n_cells)],
        )
        X = np.random.randn(n_cells, n_genes).astype(np.float32)

        return ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_genes)]))

    def test_basic(self, sc_adata):
        """Test basic pseudo-bulk aggregation."""
        from mantra.preprocessing.transform import pseudobulk

        tensor, meta = pseudobulk(
            sc_adata,
            sample_key="patient",
            slice_key="cell_type",
            min_cells=1,
        )

        assert tensor.ndim == 3
        assert tensor.shape[0] == 3  # 3 patients
        assert tensor.shape[1] == 3  # 3 cell types
        assert tensor.shape[2] == 15  # 15 genes

    def test_min_cells(self, sc_adata):
        """Test that min_cells filtering produces NaN for sparse groups."""
        from mantra.preprocessing.transform import pseudobulk

        tensor, _ = pseudobulk(
            sc_adata,
            sample_key="patient",
            slice_key="cell_type",
            min_cells=100,  # High threshold
        )

        # Some groups should have NaN
        assert torch.isnan(tensor).any()

    def test_sum_aggregation(self, sc_adata):
        """Test sum aggregation."""
        from mantra.preprocessing.transform import pseudobulk

        tensor, _ = pseudobulk(
            sc_adata,
            sample_key="patient",
            slice_key="cell_type",
            agg_func="sum",
            min_cells=1,
        )
        assert tensor.ndim == 3

    def test_invalid_agg_func(self, sc_adata):
        """Test invalid aggregation function."""
        from mantra.preprocessing.transform import pseudobulk

        with pytest.raises(ValueError, match="agg_func"):
            pseudobulk(sc_adata, "patient", "cell_type", agg_func="median")


class TestHighlyVariableFeatures:
    @pytest.mark.skipif(
        not _scanpy_available(),
        reason="scanpy incompatible with current anndata version",
    )
    def test_basic(self):
        """Test highly variable feature selection."""
        import anndata as ad

        from mantra.preprocessing.transform import highly_variable_features

        np.random.seed(42)
        # Use positive counts for seurat_v3 flavor
        X = np.random.poisson(5, size=(100, 50)).astype(np.float32)
        # Make some features more variable
        X[:, :10] = np.random.poisson(50, size=(100, 10)).astype(np.float32)
        adata = ad.AnnData(X=X)

        mask = highly_variable_features(adata, n_top=20, flavor="seurat_v3")
        assert mask.sum() == 20
        assert len(mask) == 50


class TestRoundTrip:
    def test_anndata_to_mantra(self):
        """Test round-trip: AnnData → tensor → MANTRA."""
        import anndata as ad

        from mantra.preprocessing.anndata import from_anndata
        from mantra.preprocessing.transform import normalize

        np.random.seed(42)
        n_obs = 30
        n_vars = 8
        obs = pd.DataFrame(
            {
                "sample": [f"S{i // 6}" for i in range(n_obs)],
                "slice": [f"SL{i % 6}" for i in range(n_obs)],
            },
            index=[f"cell_{i}" for i in range(n_obs)],
        )
        X = np.random.randn(n_obs, n_vars).astype(np.float32)
        adata = ad.AnnData(X=X, obs=obs, var=pd.DataFrame(index=[f"g{i}" for i in range(n_vars)]))

        tensor, meta = from_anndata(adata, sample_key="sample", slice_key="slice")
        tensor = normalize(tensor)

        # Verify can be used with MANTRA
        from mantra import MANTRA

        model = MANTRA(
            observations=tensor,
            n_features=[n_vars],
            R=3,
            use_gpu=False,
        )
        model.sample_names = meta["sample_names"]
        model.slice_names = meta["slice_names"]
        model.feature_names = [meta["feature_names"]]
        assert model is not None
