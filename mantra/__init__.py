"""MANTRA: Multi-view ANalysis with Tensor and matRix Alignment.

A Bayesian probabilistic framework for integrating collections of tensors
of different orders (e.g., 3rd-order drug-response tensors with 2nd-order
RNA-seq matrices).

Example
-------
>>> from mantra import MANTRA, DataGenerator
>>> # Generate synthetic data
>>> gen = DataGenerator(n_samples=100, n_drugs=10, n_features=50)
>>> gen.generate(seed=42)
>>> data = gen.get_sim_data()
>>> # Create and fit model
>>> model = MANTRA(data["Y_sim"], n_features=[50], R=5)
>>> history, _ = model.fit(n_epochs=1000)
>>> # Analyze results
>>> from mantra import tl
>>> r2 = tl.variance_explained(model)
>>> associations = tl.test(model, metadata=sample_metadata)
"""

import logging

from mantra import analysis as tl
from mantra import plotting as pl
from mantra.data.synthetic import DataGenerator
from mantra.model.core import MANTRA
from mantra.model.core import MANTRAModel
from mantra.utils.gpu import get_free_gpu_idx
from mantra.utils.seeds import set_all_seeds

__version__ = "0.1.0"
__all__ = [
    "MANTRA",
    "MANTRAModel",
    "DataGenerator",
    "get_free_gpu_idx",
    "set_all_seeds",
    "tl",  # Tools/analysis module (like scanpy.tl)
    "pl",  # Plotting module (like scanpy.pl)
]

# Configure logging
logging.getLogger(__name__).addHandler(logging.NullHandler())
