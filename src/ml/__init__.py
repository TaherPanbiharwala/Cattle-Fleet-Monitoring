"""Phase 3 public-dataset behaviour benchmark.

This package is optional.  Phase 1 deliberately does not import it, so the
simulator continues to run without ML libraries installed.
"""

from .features import FEATURE_NAMES, SIGNAL_NAMES

__all__ = ["FEATURE_NAMES", "SIGNAL_NAMES"]
