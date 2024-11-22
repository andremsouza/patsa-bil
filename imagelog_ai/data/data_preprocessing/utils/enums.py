"""This module contains the enums used in the data preprocessing module."""

from enum import Enum


class DenoiserMethod(Enum):
    """Enum class for denoising methods"""

    TOTAL_VARIATION = "Total Variation"
    NON_LOCAL_MEANS = "Non Local Means"
    BILATERAL = "Bilateral"
    GAUSSIAN = "Gaussian"
    WAVELET = "Wavelet"
    BM3D = "BM3D"


class WrappingType(Enum):
    """Enum class for wrapping types"""

    NOWRAP = 0
    RIGHT = 1
    LEFT = 2
    BOTH = 3
