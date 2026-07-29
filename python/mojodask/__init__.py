"""Mojo kernels behind a compact, Dask-compatible blocked API."""

from . import array, dataframe
from .core import Delayed, compute, delayed

__all__ = ["Delayed", "array", "compute", "dataframe", "delayed"]
__version__ = "0.1.0"
