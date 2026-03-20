"""Inference utilities for MANTRA.

Provides training callbacks for monitoring and controlling the fit loop:

- :class:`EarlyStoppingCallback` -- stop when loss plateaus
- :class:`CheckpointCallback` -- save periodic snapshots
- :class:`LogCallback` -- log progress at regular intervals

Example
-------
>>> from mantra.inference import EarlyStoppingCallback
>>> cb = EarlyStoppingCallback(patience=50)
>>> model.fit(n_epochs=5000, callbacks=[cb])
"""

from mantra.inference.callbacks import Callback
from mantra.inference.callbacks import CheckpointCallback
from mantra.inference.callbacks import EarlyStoppingCallback
from mantra.inference.callbacks import LogCallback

__all__ = [
    "Callback",
    "CheckpointCallback",
    "EarlyStoppingCallback",
    "LogCallback",
]
