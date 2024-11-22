from imagelog_ai.features.synthetic_data.imagelog_generator.BaseImagelogLayersGenerator import (
    BaseImagelogLayersGenerator,
)
from imagelog_ai.features.synthetic_data.imagelog_generator.utils.dataclass import (
    RegionProperties,
)
from imagelog_ai.features.synthetic_data.imagelog_generator.utils.enum import (
    PatternType,
    BackgroundType,
)
from imagelog_ai.features.synthetic_data.utils.normalization import (
    min_max_normalize,
    color_range_normalize,
    color_range_scale,
)
from imagelog_ai.features.synthetic_data.utils.filter import gaussian_filter
from imagelog_ai.utils.random import random_float_in_range

from typing import Any, List, Tuple

import torch

from imagelog_ai.features.synthetic_data.imagelog_generator.utils.dataclass import (
    PatternProperties,
)
import math
import random


class LaminadoGenerator(BaseImagelogLayersGenerator):
    """Module to use with pre-trained networks for Transfer Learning on new classification problems.

    Parameters
    ----------
    early_stopping_patience: float
            Initial learning rate
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.pattern_name = "laminar"

    def _define_region_properties(self) -> List[RegionProperties]:
        return [
            RegionProperties(
                distance_range=(6, 40),
                distance_values=[],
                color_range=(0.95, 1),
                background_type=BackgroundType.PERLIN,
                background_texture=None,
                pattern_chances=[0.60, 0.40],
                pattern_types=[PatternType.SIMPLE, PatternType.IMPULSE],
                # pattern_chances = [0.45, 0.3, 0.25],
                # pattern_types = [PatternType.SIMPLE, PatternType.IMPULSE, PatternType.INVERSEBRECHADO],
                patterns=[],
            ),
            RegionProperties(
                distance_range=(6, 100),
                distance_values=[],
                color_range=(0.35, 0.40),
                background_type=BackgroundType.PERLIN,
                background_texture=None,
                pattern_chances=[1.0],
                pattern_types=[PatternType.SIMPLE],
                # pattern_chances = [0.65, 0.35],
                # pattern_types = [PatternType.SIMPLE, PatternType.BRECHADO],
                patterns=[],
            ),
        ]

    def _generate_image(self, image_shape: tuple) -> torch.Tensor:
        res_img = self._make_laminado_patterns_texture(
            image_shape,
            properties_list=self.region_list,
            noise_strenght=0.2,
            smooth_delta=2.5,
        )

        res_img = torch.clip(res_img, 0, 1)
        res_img = self._heighten_contrast(res_img, self.heighten_contrast_range)

        return res_img


class MacicoGenerator(BaseImagelogLayersGenerator):
    """Module to use with pre-trained networks for Transfer Learning on new classification problems.

    Parameters
    ----------
    early_stopping_patience: float
            Initial learning rate
    """

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.pattern_name = "macico"

    def _apply_brechado(self, image: torch.Tensor):
        shape = image.shape
        number_of_windows = shape[0] // shape[1]

        mask = self.noise_generator.perlin_noise(shape, (number_of_windows, 8))
        mask = min_max_normalize(mask)

        wavyness = self.noise_generator.perlin_noise(
            (shape[0], 1), (number_of_windows, 1)
        )
        wavyness = min_max_normalize(wavyness)
        wavyness *= 30
        wavyness -= 15
        wavyness = wavyness.int().cpu().numpy()

        for i in range(shape[0]):
            mask[i] = torch.roll(mask[i], tuple(wavyness[i]))

        # mask = torch.zeros(shape, device=self.torch_device)
        mask[mask < 0.935] = 0
        mask[mask != 0] = 1

        fundo = gaussian_filter(mask, sigma=5)
        fundo = self._heighten_contrast(fundo, (0.3, 1.2))

        return torch.maximum(image, fundo)

    def _add_random_pad_tint(self, padded_image):
        shape = list(padded_image.size())
        number_of_windows = shape[0] // shape[1]
        y_step = shape[1] / 12

        dark_perlin = self._norm_perlin_noise(shape, (number_of_windows, 1))
        dark_perlin -= 0.4
        dark_perlin[dark_perlin < 0] = 0
        dark_perlin *= 1.4

        break_centers = [
            0,
            int(0.5 * y_step),
            int(2 * y_step),
            int(3.5 * y_step),
            int(5 * y_step),
            int(6.5 * y_step),
            int(8 * y_step),
            int(9.5 * y_step),
            int(11 * y_step),
            shape[1],
        ]

        data_collumns = len(break_centers) - 1

        mask = torch.full(shape, False, dtype=bool, device=self.torch_device)

        for collumn_index in range(data_collumns):
            window_selection = self._binary_random_cummulative_values(
                number_of_windows, 0.9, resolution=2, speed=0.2
            )
            for window_number in range(number_of_windows):
                if window_selection[window_number] == 0:
                    continue
                mask[
                    window_number * shape[1] : (window_number + 1) * shape[1],
                    break_centers[collumn_index] : break_centers[collumn_index + 1],
                ] = True

        padded_image[mask] += dark_perlin[mask]

        return padded_image

    def _define_region_properties(self) -> List[RegionProperties]:
        return [
            RegionProperties(
                distance_range=(300, 1200),
                distance_values=[],
                color_range=(0.21, 0.25),
                background_type=BackgroundType.PERLIN,
                background_texture=None,
                pattern_chances=[0.25, 0.25, 0.25, 0.25],
                pattern_types=[
                    PatternType.GRANULE,
                    PatternType.SPOTS,
                    PatternType.SIMPLE,
                    PatternType.BRECHADO,
                ],
                patterns=[],
            ),
            RegionProperties(
                distance_range=(13, 25),
                distance_values=[],
                color_range=(0.95, 1),
                background_type=BackgroundType.FLAT,
                background_texture=None,
                pattern_chances=[1],
                pattern_types=[PatternType.SIMPLE],
                patterns=[],
            ),
        ]

    def _breaking_mask(self, image_shape: Tuple[int], break_cutoff: float):
        broken_mask = self._norm_fractal_noise(
            image_shape, ((image_shape[0] // image_shape[1]) * 1, 6), 4
        )  # (H, 1) * 2^(4-1) = (8*H, 8)
        broken_mask[broken_mask <= break_cutoff] = 0
        broken_mask[broken_mask > break_cutoff] = 1

        return broken_mask

    def _broken_sine(
        self,
        output_mask: torch.Tensor,
        amplitude: float,
        center: int,
        width: float,
        noise_strength: float,
        noise_values: torch.Tensor,
        mask_value: float = 1,
    ):
        image_shape = output_mask.shape
        input_angles = torch.linspace(
            0, 2 * math.pi, image_shape[1], device=self.torch_device
        )

        sine_positions = torch.sin(
            (input_angles) + noise_strength * noise_values[center, :]
        )
        sine_positions *= amplitude
        sine_positions += center
        sine_positions = torch.clip(
            sine_positions, min=0, max=image_shape[0]
        )  # Keep position limited to image

        start = (sine_positions - (width / 2)).long()
        end = (sine_positions + (width / 2)).long()

        for x_index in range(image_shape[1]):
            output_mask[start[x_index] : end[x_index], x_index] = mask_value

        return output_mask

    def _broken_sines_mask(
        self,
        image_shape,
        amplitude_range,
        width_range,
        distance_range,
        cluster_size_range,
        noise_strength,
        break_cutoff,
        high_amplitude_chance=0.2,
        white_chance=0.2,
    ):
        center = random.randint(distance_range[0], distance_range[1])
        noise_values = self._norm_fractal_noise(
            image_shape, ((image_shape[0] // image_shape[1]) * 2, 5), 4
        )  # (H, 1) * 2^(4-1) = (8*H, 8)

        output_mask = torch.zeros(image_shape, device=self.torch_device)

        while center < image_shape[0]:
            cluster_size = random.randint(cluster_size_range[0], cluster_size_range[1])
            if random.random() <= white_chance:
                cluster_color = -1
            else:
                cluster_color = 1

            if cluster_size == 1 and (random.random() <= high_amplitude_chance):
                amplitude = amplitude_range[1] * random_float_in_range((2, 5))
                width = random_float_in_range(width_range)
                self._broken_sine(
                    output_mask,
                    amplitude,
                    center,
                    width,
                    noise_strength,
                    noise_values,
                    cluster_color,
                )
            else:
                for i in range(cluster_size):
                    amplitude = random_float_in_range(amplitude_range)
                    width = random_float_in_range(width_range)
                    self._broken_sine(
                        output_mask,
                        amplitude,
                        center,
                        width,
                        noise_strength,
                        noise_values,
                        cluster_color,
                    )
                    center += int(random_float_in_range((1.5, 3)) * (width + amplitude))
                    if (center + amplitude_range[1]) >= image_shape[0]:
                        break

            center += random.randint(distance_range[0], distance_range[1])

        output_mask *= self._breaking_mask(image_shape, break_cutoff)

        return output_mask

    def _add_sine_structures(self, image):
        image_shape = image.shape

        fracture_mask = self._broken_sines_mask(
            image_shape,
            amplitude_range=(10, 40),
            width_range=(2, 8),
            distance_range=(500, 2000),
            cluster_size_range=(1, 4),
            noise_strength=0.75,
            break_cutoff=(0.1),
            high_amplitude_chance=0.5,
        )
        broken_sine_mask = self._broken_sines_mask(
            image_shape,
            amplitude_range=(10, 20),
            width_range=(2, 9),
            distance_range=(400, 1500),
            cluster_size_range=(1, 3),
            noise_strength=1.5,
            break_cutoff=(0.7),
            high_amplitude_chance=0,
            white_chance=0,
        )

        image += broken_sine_mask
        image += fracture_mask

        return torch.clip(image, 0, 1)

    def _generate_image(self, image_shape: tuple) -> torch.Tensor:
        res_img = self._make_laminado_patterns_texture(
            image_shape,
            properties_list=self.region_list,
            noise_strenght=0.2,
            smooth_delta=2.5,
        )

        res_img = self._heighten_contrast(res_img, self.heighten_contrast_range)
        res_img = self._add_sine_structures(res_img)
        res_img = self._apply_brechado(res_img)

        # res_img = self._add_random_pad_tint(res_img)

        return res_img
