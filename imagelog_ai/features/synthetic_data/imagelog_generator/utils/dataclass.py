from imagelog_ai.features.synthetic_data.imagelog_generator.utils.enum import (
    BackgroundType,
    PatternType,
)

from dataclasses import dataclass
from typing import List, Callable, Tuple

import torch


@dataclass
class PatternProperties:
    mask_function: Callable
    texture: torch.Tensor
    mask: torch.Tensor


@dataclass
class RegionProperties:
    distance_values: List[int]
    distance_range: Tuple[float]

    color_range: Tuple[float]

    background_type: BackgroundType
    pattern_types: List[PatternType]
    pattern_chances: List[float]

    background_texture: torch.Tensor
    patterns: List[PatternProperties]


@dataclass
class GranuleProperties:
    initial_cutoff: float
    cutoff_step: float

    initial_sigma: float
    sigma_step: float

    repetitions: int

    color_range: Tuple[float]
    resolutions: Tuple[int]


class NewRegionProperties:
    def __init__(
        self,
        background_type: BackgroundType,
        pattern_types: List[PatternType],
        pattern_chances: List[float],
        distance_range: Tuple[float],
        color_range: Tuple[float],
    ) -> None:
        self.background_type: BackgroundType = background_type
        self.pattern_types: List[PatternType] = pattern_types
        self.pattern_chances: List[float] = pattern_chances

        self.distance_range: Tuple[float] = distance_range
        self.color_range: Tuple[float] = color_range

        self.background_texture: torch.Tensor = None
        self.patterns: List[PatternProperties] = []
        self.distance_values: List[int] = []

    def _setup_regions(
        self, image_shape: tuple, properties_list: List[RegionProperties]
    ):
        self.background_texture = self.background_generation_methods[
            self.background_type
        ](image_shape, self.color_range)

        self.distance_values = self._bounded_smooth_cummulative_values(
            array_length=image_shape[0], value_range=self.distance_range, speed=0.002
        ).astype(int)

        for pattern_type in self.pattern_types:
            texture_functions = self.pattern_methods[pattern_type]

            pattern_properties = PatternProperties(
                mask_function=texture_functions[0],
                texture=None,
                mask=torch.zeros(image_shape, device=self.device),
            )

            if texture_functions[1] != None:
                pattern_properties.texture = texture_functions[1](
                    self.background_texture
                )
            else:
                pattern_properties.texture = self.background_texture

            self.patterns.append(pattern_properties)
