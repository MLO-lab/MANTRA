"""Tests for MANTRA model."""

import torch


def test_synthetic_data_generation(small_synthetic_tensor):
    """Test that synthetic data is generated correctly."""
    data = small_synthetic_tensor.get_sim_data()

    # Check all required keys are present
    assert "A1_sim" in data
    assert "A2_sim" in data
    assert "A3_sim" in data
    assert "Y_sim" in data

    # Check shapes
    assert data["A1_sim"].shape == (10, 3)  # n_samples x R
    assert data["A2_sim"].shape == (5, 3)  # n_drugs x R
    assert data["A3_sim"].shape == (8, 3)  # n_features x R
    assert data["Y_sim"].shape == (10, 5, 8)  # n_samples x n_drugs x n_features


def test_tensor_reconstruction(factor_matrices):
    """Test that tensor can be reconstructed from factor matrices."""
    A1 = factor_matrices["A1"]
    A2 = factor_matrices["A2"]
    A3 = factor_matrices["A3"]

    # Reconstruct tensor using einsum
    reconstructed = torch.einsum("ir,jr,kr->ijk", A1, A2, A3)

    # Check shape
    assert reconstructed.shape == (10, 5, 8)


def test_mantra_model_creation(mantra_model):
    """Test that MANTRA model can be created."""
    assert mantra_model is not None
    assert mantra_model.R == 3
    assert mantra_model._built is False
    assert mantra_model._trained is False


def test_data_generator_missingness(small_synthetic_tensor):
    """Test that missingness can be generated."""
    # Generate missingness
    small_synthetic_tensor.generate_missingness(p=0.2)
    data = small_synthetic_tensor.get_sim_data()

    # Check that some values are NaN
    assert torch.isnan(data["Y_sim"]).any()


def test_mantra_model_fit(mantra_model, set_seeds):
    """Test that MANTRA model can be trained."""
    history, stopped_early = mantra_model.fit(
        n_epochs=5,
        n_particles=1,
        learning_rate=0.01,
        optimizer="adam",
        verbose=False,
        seed=42,
    )

    # Check training completed
    assert len(history) == 5
    assert not stopped_early
    assert mantra_model._trained is True


def test_mantra_posterior(trained_mantra_model):
    """Test that posterior can be extracted."""
    posterior = trained_mantra_model.get_posterior()

    # Check required keys
    assert "A1" in posterior
    assert "A2" in posterior
    assert "A3" in posterior


def test_mantra_elbo_decreases(mantra_model, set_seeds):
    """Test that ELBO generally decreases during training."""
    history, _ = mantra_model.fit(
        n_epochs=20,
        n_particles=1,
        learning_rate=0.01,
        optimizer="adam",
        verbose=False,
        seed=42,
    )

    # ELBO should generally decrease (final should be lower than initial)
    # Using a window to account for noise
    initial_avg = sum(history[:5]) / 5
    final_avg = sum(history[-5:]) / 5

    assert final_avg < initial_avg, "ELBO should decrease during training"


# -----------------------------------------------------------------------------
# Tests for embedding accessor methods
# -----------------------------------------------------------------------------


def test_get_sample_embeddings(trained_mantra_model):
    """Test get_sample_embeddings accessor method (A1 matrix)."""
    embeddings = trained_mantra_model.get_sample_embeddings()

    # Check shape (n_samples=10, R=3)
    assert embeddings.shape == (10, 3)
    assert isinstance(embeddings, torch.Tensor)


def test_get_sample_embeddings_as_df(trained_mantra_model):
    """Test get_sample_embeddings returns DataFrame when requested."""
    import pandas as pd

    embeddings_df = trained_mantra_model.get_sample_embeddings(as_df=True)

    assert isinstance(embeddings_df, pd.DataFrame)
    assert embeddings_df.shape == (10, 3)
    assert list(embeddings_df.columns) == list(trained_mantra_model.factor_names)


def test_get_slice_embeddings(trained_mantra_model):
    """Test get_slice_embeddings accessor method (A2 matrix)."""
    embeddings = trained_mantra_model.get_slice_embeddings()

    # Check shape (n_drugs=5, R=3)
    assert embeddings.shape == (5, 3)
    assert isinstance(embeddings, torch.Tensor)


def test_get_feature_embeddings(trained_mantra_model):
    """Test get_feature_embeddings accessor method (A3 matrix)."""
    embeddings = trained_mantra_model.get_feature_embeddings()

    # Check shape (n_features=8, R=3)
    assert embeddings.shape == (8, 3)
    assert isinstance(embeddings, torch.Tensor)


def test_get_loadings(trained_mantra_model):
    """Test get_loadings method (product of two embeddings)."""
    # Default: slice x feature loadings
    loadings = trained_mantra_model.get_loadings()

    # Check shape (n_slices=5, n_features=8, n_factors=3)
    assert loadings.shape == (5, 8, 3)
    assert isinstance(loadings, torch.Tensor)

    # Test sample x slice loadings
    loadings_sf = trained_mantra_model.get_loadings(mode1="sample", mode2="slice")
    assert loadings_sf.shape == (10, 5, 3)


def test_get_reconstructed(trained_mantra_model):
    """Test get_reconstructed accessor method."""
    recon = trained_mantra_model.get_reconstructed()

    # Check shape matches input (n_samples=10, n_drugs=5, n_features=8)
    assert recon.shape == (10, 5, 8)
    assert isinstance(recon, torch.Tensor)


def test_get_embeddings(trained_mantra_model):
    """Test get_embeddings accessor method."""
    embeddings = trained_mantra_model.get_embeddings()

    assert "A1" in embeddings
    assert "A2" in embeddings
    assert "A3" in embeddings
    assert embeddings["A1"].shape == (10, 3)
    assert embeddings["A2"].shape == (5, 3)
    assert embeddings["A3"].shape == (8, 3)


def test_factor_names_property(trained_mantra_model):
    """Test factor_names property."""
    import pandas as pd

    names = trained_mantra_model.factor_names

    assert isinstance(names, pd.Index)
    assert len(names) == 3

    # Test setting custom names
    trained_mantra_model.factor_names = ["MyFactor_0", "MyFactor_1", "MyFactor_2"]
    assert list(trained_mantra_model.factor_names) == [
        "MyFactor_0",
        "MyFactor_1",
        "MyFactor_2",
    ]


def test_slice_names_property(trained_mantra_model):
    """Test slice_names property."""
    import pandas as pd

    names = trained_mantra_model.slice_names

    assert isinstance(names, pd.Index)
    assert len(names) == 5  # n_drugs=5

    # Test setting custom names
    trained_mantra_model.slice_names = [
        "CellType_A",
        "CellType_B",
        "CellType_C",
        "CellType_D",
        "CellType_E",
    ]
    assert next(iter(trained_mantra_model.slice_names)) == "CellType_A"


# -----------------------------------------------------------------------------
# Tests for analysis functions
# -----------------------------------------------------------------------------


def test_variance_explained(trained_mantra_model):
    """Test variance_explained analysis function."""
    from mantra.analysis import variance_explained

    r2 = variance_explained(trained_mantra_model)

    # Check total R²
    assert "total" in r2
    # R² can be slightly negative if model hasn't converged well
    # In practice it should be between -1 and 1
    assert -1 <= r2["total"] <= 1

    # Check per-factor R²
    assert "per_factor" in r2
    assert len(r2["per_factor"]) == 3  # 3 factors


def test_variance_explained_per_factor(trained_mantra_model):
    """Test variance_explained_per_factor function."""
    from mantra.analysis import variance_explained_per_factor

    df = variance_explained_per_factor(trained_mantra_model, cumulative=True)

    assert "r2" in df.columns
    assert "cumulative_r2" in df.columns
    assert len(df) == 3


def test_filter_factors(trained_mantra_model):
    """Test filter_factors function."""
    from mantra.analysis import filter_factors

    # Keep top 2 factors
    kept = filter_factors(trained_mantra_model, r2_thresh=2)

    assert len(kept) == 2
    assert all(isinstance(name, str) for name in kept)
