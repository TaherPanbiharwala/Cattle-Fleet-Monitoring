"""MmCows daily personal-baseline SPC/CUSUM detector."""

from .config import DetectorConfig
from .pipeline import run_detector

__all__ = ["DetectorConfig", "run_detector"]
