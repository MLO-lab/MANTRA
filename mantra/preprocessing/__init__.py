"""Preprocessing utilities for MANTRA (accessible as ``mantra.pp``).

- :func:`from_anndata` -- construct 3D tensor from AnnData
- :func:`from_mudata` -- construct tensors from MuData
- :func:`normalize` -- center and scale tensors
- :func:`pseudobulk` -- aggregate single-cell to pseudo-bulk tensor
- :func:`highly_variable_features` -- select highly variable features

Example
-------
>>> import mantra
>>> tensor, meta = mantra.pp.from_anndata(adata, sample_key="patient", slice_key="cell_type")
>>> tensor = mantra.pp.normalize(tensor)
"""

from mantra.preprocessing.anndata import from_anndata
from mantra.preprocessing.anndata import from_mudata
from mantra.preprocessing.transform import highly_variable_features
from mantra.preprocessing.transform import normalize
from mantra.preprocessing.transform import pseudobulk

__all__ = [
    "from_anndata",
    "from_mudata",
    "highly_variable_features",
    "normalize",
    "pseudobulk",
]
