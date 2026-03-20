# MANTRA

**M**ulti-view **AN**alysis with **T**ensor and mat**R**ix **A**lignment

A Bayesian probabilistic framework for integrating collections of tensors of different orders (e.g., 3rd-order drug-response tensors with 2nd-order RNA-seq matrices). MANTRA combines group factor analysis and tensor decomposition using variational inference with structured sparsity priors.

## Installation

```bash
git clone https://github.com/MLO-lab/MANTRA.git
cd MANTRA
uv sync
```

If you don't have [uv](https://docs.astral.sh/uv/) installed:

```bash
pip install uv
```

Alternatively, install with pip directly:

```bash
pip install .
```

## Quick Start

```python
from mantra import MANTRA, DataGenerator

# Generate synthetic data
generator = DataGenerator(n_samples=100, n_drugs=20, n_features=50, R=5)
generator.generate(seed=42)
data = generator.get_sim_data()

# Fit model
model = MANTRA(
    observations=data["Y_sim"],
    n_features=[50],
    R=5,
)
history, _ = model.fit(n_epochs=1000, learning_rate=0.01)

# Access embeddings
A1 = model.get_sample_embeddings(as_df=True)   # samples x factors
A2 = model.get_slice_embeddings(as_df=True)     # slices x factors
A3 = model.get_feature_embeddings(as_df=True)   # features x factors
```

## Loading Real Data

MANTRA provides a `pp` (preprocessing) module for constructing 3D tensors from AnnData and MuData objects, following the scanpy-style API.

### From AnnData

```python
import mantra

# Build tensor from long-format AnnData (e.g., patients x cell types x genes)
tensor, metadata = mantra.pp.from_anndata(
    adata,
    sample_key="patient_id",
    slice_key="cell_type",
)

# Normalize features
tensor = mantra.pp.normalize(tensor, center=True, scale=True)

# Fit model with metadata
model = mantra.MANTRA(observations=tensor, R=10)
model.sample_names = metadata["sample_names"]
model.slice_names = metadata["slice_names"]
model.feature_names = [metadata["feature_names"]]
history, _ = model.fit(n_epochs=3000, learning_rate=0.005)
```

### From single-cell data (pseudo-bulk)

```python
# Aggregate single-cell data to pseudo-bulk tensor
tensor, metadata = mantra.pp.pseudobulk(
    adata_sc,
    sample_key="patient_id",
    slice_key="cell_type",
    agg_func="mean",
    min_cells=10,
)
```

### From MuData (multi-view)

```python
# Each modality becomes a separate view
tensors, metadata = mantra.pp.from_mudata(
    mdata,
    sample_key="patient_id",
    slice_key="cell_type",
)

model = mantra.MANTRA(
    observations=tensors,
    n_features=[t.shape[2] for t in tensors],
    R=20,
    view_names=metadata["view_names"],
)
```

### Feature selection

```python
# Select highly variable features before tensor construction
hvg_mask = mantra.pp.highly_variable_features(adata, n_top=2000)
adata_hvg = adata[:, hvg_mask]
```

## Downstream Analysis (`tl`)

After training, use `mantra.tl` for downstream analysis. MANTRA bridges to the scanpy ecosystem by caching sample embeddings (A1) as an AnnData object internally.

### Variance explained

```python
import mantra

# R-squared decomposition (total, per-factor, per-view)
r2 = mantra.tl.variance_explained(model)
print(r2["total"])       # Total R²
print(r2["per_factor"])  # Per-factor R², sorted by importance

# Select top factors capturing 95% variance
top_factors = mantra.tl.filter_factors(model, r2_thresh=0.95)
```

### Metadata association testing

```python
# Test which factors associate with sample metadata
results = mantra.tl.test(model, metadata=clinical_df, method="kruskal")
print(results[results["significant"]])
```

### Embedding analysis (scanpy wrappers)

```python
# Add sample-level metadata for downstream analysis
mantra.tl.add_metadata(model, "subtype", patient_subtypes)
mantra.tl.add_metadata(model, "age", patient_ages)

# Compute neighbor graph, UMAP, and clustering
mantra.tl.neighbors(model)
mantra.tl.umap(model)
mantra.tl.leiden(model)

# Rank factors by group
mantra.tl.rank(model, groupby="subtype")
```

### Pathway enrichment

```python
# GSEA on feature embeddings
results = mantra.tl.enrichment(
    model,
    gene_sets="GO_Biological_Process_2021",
    method="gsea",
)

# ORA with top features
results = mantra.tl.enrichment(
    model,
    gene_sets="KEGG_2021_Human",
    method="ora",
    top_n=100,
)
```

## Visualization (`pl`)

MANTRA provides a `pl` module for visualizing model outputs, factor structure, and metadata associations.

### Training diagnostics

```python
import mantra

# ELBO convergence
mantra.pl.plot_elbo(history)

# Factor matrix distributions
mantra.pl.distplots(model.get_posterior(), keyorder=["A1", "A2", "A3"])
```

### Factor interpretation

```python
# Variance explained (bar chart or heatmap for multi-view)
mantra.pl.variance_explained(model, top=10)

# Slice x feature loading heatmap for a factor
mantra.pl.factor_weights(model, factor_idx=0, top=25)

# Slice embeddings (e.g., how cell types load on a factor)
mantra.pl.slice_weights(model, factor_idx=[0, 1, 2])
```

### Sample embeddings

```python
# Scatter plot of two factors, colored by metadata
mantra.pl.scatter(model, x=0, y=1, color="subtype")

# UMAP of sample embeddings
mantra.pl.embedding(model, color="subtype", method="umap")

# Hierarchical clustering heatmap
mantra.pl.clustermap(model)
```

### Group comparisons

```python
# Factor values by sample group
mantra.pl.boxplot(model, factor_idx=[0, 1], groupby="subtype")
mantra.pl.violinplot(model, factor_idx=0, groupby="subtype")
mantra.pl.stripplot(model, factor_idx=0, groupby="subtype")
```

## Save and Load

MANTRA uses a pickle-free format (JSON metadata + NPZ arrays) for safe, portable model persistence.

```python
# Save
model.save("my_model/")

# Load
model = MANTRA.load("my_model/")
```

## API Reference

### Model

| Method | Description |
|--------|-------------|
| `MANTRA(observations, R=10, ...)` | Create model |
| `model.fit(n_epochs, learning_rate, ...)` | Train with SVI |
| `model.get_sample_embeddings(as_df=True)` | A1: samples x factors |
| `model.get_slice_embeddings(as_df=True)` | A2: slices x factors |
| `model.get_feature_embeddings(view=..., as_df=True)` | A3: features x factors |
| `model.get_loadings(mode1, mode2)` | Product of two embedding matrices |
| `model.get_reconstructed()` | Reconstructed tensor |
| `model.save(path)` / `MANTRA.load(path)` | Pickle-free persistence |

### Preprocessing (`mantra.pp`)

| Function | Description |
|----------|-------------|
| `from_anndata(adata, sample_key, slice_key)` | AnnData to 3D tensor |
| `from_mudata(mdata, sample_key, slice_key)` | MuData to list of tensors |
| `normalize(tensor, center, scale)` | Center/scale along features |
| `pseudobulk(adata, sample_key, slice_key)` | Single-cell to pseudo-bulk |
| `highly_variable_features(adata, n_top)` | Feature selection via scanpy |

### Analysis (`mantra.tl`)

| Function | Description |
|----------|-------------|
| `variance_explained(model)` | R-squared decomposition |
| `test(model, metadata)` | Factor-metadata association testing |
| `filter_factors(model, r2_thresh)` | Select factors by cumulative R-squared |
| `setup_cache(model)` | Initialize AnnData cache from A1 |
| `add_metadata(model, name, values)` | Attach sample metadata |
| `get_metadata(model, name)` | Retrieve sample metadata |
| `neighbors(model)` | Compute neighbor graph |
| `umap(model)` / `tsne(model)` | Dimensionality reduction |
| `leiden(model)` | Clustering |
| `rank(model, groupby)` | Rank factors by group |
| `enrichment(model, gene_sets)` | Pathway enrichment (GSEA/ORA) |

### Plotting (`mantra.pl`)

| Function | Description |
|----------|-------------|
| `plot_elbo(history)` | ELBO training curve |
| `distplots(posterior, keyorder)` | Factor matrix distributions |
| `variance_explained(model, top)` | R-squared bar chart / heatmap |
| `factor_weights(model, factor_idx, top)` | Slice x feature loading heatmap |
| `slice_weights(model, factor_idx)` | Slice embedding bar chart |
| `scatter(model, x, y, color)` | Factor scatter plot |
| `embedding(model, color, method)` | UMAP / tSNE plot |
| `clustermap(model)` | Hierarchical clustering heatmap |
| `stripplot` / `boxplot` / `violinplot` | Group comparison plots |

## Development

Run tests:
```bash
pytest mantra/tests/ -v
```

Run linting:
```bash
ruff check mantra/
black mantra/ --check --line-length 100
```

## Paper

If you use MANTRA in your research, please cite:

> **Interpretable multi-omics integration across mixed-order tensors with MANTRA**

## License

MIT
