"""MANTRA model components.

Provides the main :class:`MANTRA` user-facing class and the underlying
:class:`MANTRAModel` Pyro generative model.

Example
-------
>>> from mantra.model import MANTRA
>>> model = MANTRA(tensor_data, n_features=[50], R=5)
>>> history, _ = model.fit(n_epochs=1000)
>>> model.save("my_model")
>>> loaded = MANTRA.load("my_model")
"""

from mantra.model.core import MANTRA
from mantra.model.core import MANTRAModel

__all__ = ["MANTRA", "MANTRAModel"]
