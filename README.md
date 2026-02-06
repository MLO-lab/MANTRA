# MANTRA

**M**ulti-view **AN**alysis with **T**ensor and mat**R**ix **A**lignment

A Bayesian tensor decomposition framework for multi-view data integration.

## Installation

First install the conda environment and activate it:

```bash
conda env create -f environment.yml
conda activate mantra
```

Then install the packages with poetry:

```bash
poetry install
```

## Usage

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
model.fit(n_epochs=100, learning_rate=0.01)

# Get embeddings
embeddings = model.get_embeddings()
```

## Tutorials

See the `tutorials/` directory for example notebooks.

## Development

Run tests:
```bash
poetry run pytest
```

Run linting:
```bash
poetry run pre-commit run --all-files
```

## License

MIT
