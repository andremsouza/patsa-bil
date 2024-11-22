"""This module contains enums for normalization methods and wrapping types."""

from enum import Enum


class NormalizationMethod(Enum):
    """Enum class for normalization methods"""

    HISTOGRAM = 0
    CLAHE = 1
    QUANTILE = 2
    GLOBAL = 3


class WrappingType(Enum):
    """Enum class for wrapping types"""

    NOWRAP = 0
    LEFT = 1
    BOTH = 2
    RIGHT = 3
