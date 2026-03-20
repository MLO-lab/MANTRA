"""Data handling utilities for MANTRA.

Provides :class:`DataGenerator` for creating synthetic tensors with known
factor structure, useful for testing and benchmarking.

Example
-------
>>> from mantra.data import DataGenerator
>>> gen = DataGenerator(n_samples=100, n_drugs=10, n_features=50, R=5)
>>> gen.generate(seed=42)
>>> data = gen.get_sim_data()  # dict with Y_sim, A1_sim, A2_sim, A3_sim
"""

from mantra.data.synthetic import DataGenerator

__all__ = ["DataGenerator"]
