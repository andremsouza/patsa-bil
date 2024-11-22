"""This module contains the dictionary of transforms used in the data preprocessing module."""

from typing import Type

from torchvision.transforms import Resize, ToTensor, Normalize, ToPILImage

from imagelog_ai.data.data_preprocessing.image_spectral_color import SpectralColor
from imagelog_ai.data.data_preprocessing.image_equalizer import EqualizerCLAHE
from imagelog_ai.data.data_preprocessing.image_denoiser import Denoiser

Transforms: dict[str, Type] = {
    "SpectralColor": SpectralColor,
    "ToPILImage": ToPILImage,
    "Normalize": Normalize,
    "Denoiser": Denoiser,
    "ToTensor": ToTensor,
    "Resize": Resize,
    "CLAHE": EqualizerCLAHE,
}
