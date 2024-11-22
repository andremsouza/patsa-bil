from imagelog_ai.features.synthetic_data.imagelog_generator.BaseImagelogGenerator import (
    BaseImagelogGenerator,
)
from imagelog_ai.features.synthetic_data.utils.normalization import min_max_normalize
from imagelog_ai.features.synthetic_data.utils.perlin_noise import PerlinNoise2D
from imagelog_ai.features.synthetic_data.utils.filter import gaussian_filter
from imagelog_ai.utils import argcheck

from torchvision.transforms import Resize
from typing import Any, Tuple

import torch


class CaoticoGenerator(BaseImagelogGenerator):
    """Module to use with pre-trained networks for Transfer Learning on new classification problems.

    Parameters
    ----------
    early_stopping_patience: float
            Initial learning rate
    """

    def __init__(
        self,
        low_inter: Tuple[float, float] = [0.40, 0.15],
        top_inter: Tuple[float, float] = [0.70, 1.00],
        brechado_middle_point: float = 0.68,
        **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)
        argcheck.is_sequence(
            float, sequence_length=2, low_inter=low_inter, top_inter=top_inter
        )
        argcheck.is_positive(float, brechado_middle_point=brechado_middle_point)

        self.brechado_middle_point = brechado_middle_point
        self.low_inter = low_inter
        self.top_inter = top_inter
        self.pattern_name = "caotico"

    def _make_brechado_fundo(self, shape, low_interval, top_interval, middle_point=0.5):
        number_of_windows = shape[0] // shape[1]

        noise_property_list = [
            [(8 * number_of_windows, 4), 1, 0.75],
            [(8 * number_of_windows, 4), 2, 0.75],
            [(2 * number_of_windows, 2), 4, 0.75],
            [(8 * number_of_windows, 4), 1, 0.50],
            [(8 * number_of_windows, 4), 2, 0.50],
            [(4 * number_of_windows, 16), 1, 0.75],
            [(4 * number_of_windows, 4), 2, 0.75],
            [(2 * number_of_windows, 1), 4, 0.75],
        ]

        i = 0
        result_image = torch.zeros(shape, device=self.torch_device)
        for properties in noise_property_list:
            i += 1

            image = (
                self._norm_fractal_noise(shape, properties[0], properties[1])
                ** properties[2]
            )
            image[image < 0.4] = 0
            image[(image <= 0.6) & (image >= 0.4)] = 0.5
            image[image > 0.6] = 1
            result_image += image

            del image
        result_image = min_max_normalize(result_image)

        image_mask = result_image < middle_point  # [0,mid[
        result_image[image_mask] /= middle_point  # [0,1[
        result_image[image_mask] *= low_interval[1] - low_interval[0]  # [0,low1-low0[
        result_image[image_mask] += low_interval[0]  # [low0,low1[

        image_mask = result_image >= middle_point  # [mid, 1]
        result_image[image_mask] -= middle_point  # [0,1-mid]
        result_image[image_mask] /= 1 - middle_point  # [0,1]
        result_image[image_mask] *= top_interval[1] - top_interval[0]  # [0,top1-top0]
        result_image[image_mask] += top_interval[0]  # [top0,top1]

        result_image = result_image**0.85

        return gaussian_filter(result_image, sigma=3)

    def generate_imagelog(
        self, tile_batch_size: int, insert_blind_spots: bool
    ) -> torch.Tensor:
        argcheck.is_instance(bool, insert_blind_spots=insert_blind_spots)
        argcheck.is_positive(int, tile_batch_size=tile_batch_size)

        generated_shape = (
            self.generated_tile_shape[0] * tile_batch_size,
            self.generated_tile_shape[1],
        )
        output_shape = (
            self.output_tile_shape[0] * tile_batch_size,
            self.output_tile_shape[1],
        )

        reshape_transform = Resize(size=output_shape, antialias=True)
        self.noise_generator = PerlinNoise2D()

        result_image = self._make_brechado_fundo(
            generated_shape,
            self.low_inter,
            self.top_inter,
            middle_point=self.brechado_middle_point,
        )
        result_image = torch.clip(result_image, 0, 1)
        result_image = self._heighten_contrast(
            result_image, self.heighten_contrast_range
        )

        result_image = reshape_transform(result_image.view((1,) + generated_shape))
        result_image = result_image.view(output_shape)

        if insert_blind_spots:
            return self._insert_blind_spots(result_image)
        else:
            return result_image


class FraturadoGenerator(BaseImagelogGenerator):
    """Module to use with pre-trained networks for Transfer Learning on new classification problems.

    Parameters
    ----------
    early_stopping_patience: float
            Initial learning rate
    """

    def __init__(self, line_thickness: float = 0.03, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        argcheck.is_positive(float, line_thickness=line_thickness)

        self.line_thickness = line_thickness

        self.pattern_name = "fraturado"

    def _set_interval_to_value(self, tensor, center, value=1):
        mask = torch.logical_and(
            (tensor > (center - self.line_thickness)),
            (tensor < (center + self.line_thickness)),
        )

        tensor[mask] = value

    def _fracture_brechado(self, shape, repetitions, octaves=4):
        mask = self._norm_fractal_noise(shape, repetitions, octaves)

        mask[mask == 1] = 0

        self._set_interval_to_value(mask, 0.5)

        mask[mask != 1] = 0

        return mask

    def _fractal_fracture_brechado(
        self, shape, repetitions, octaves=4, fractal=5, interval_range=0.03
    ):
        mask = self._fracture_brechado(shape, repetitions, octaves)

        for i in range(fractal - 1):
            mask += self._fracture_brechado(shape, repetitions)

        return torch.clip(mask, 0, 1)

    def _make_fracture_image(self, image_shape):
        number_of_windows = image_shape[0] // image_shape[1]

        mask = self._fractal_fracture_brechado(
            image_shape, (5 * number_of_windows, 3), fractal=3
        )

        result_image = self._norm_fractal_noise(
            image_shape, (5 * number_of_windows, 5), 4
        )
        result_image *= 0.4
        result_image += 0.2

        result_image[mask == 1] = 1

        return result_image

    def generate_imagelog(
        self, tile_batch_size: int, insert_blind_spots: bool
    ) -> torch.Tensor:
        argcheck.is_instance(bool, insert_blind_spots=insert_blind_spots)
        argcheck.is_positive(int, tile_batch_size=tile_batch_size)

        generated_shape = (
            self.generated_tile_shape[0] * tile_batch_size,
            self.generated_tile_shape[1],
        )
        output_shape = (
            self.output_tile_shape[0] * tile_batch_size,
            self.output_tile_shape[1],
        )

        reshape_transform = Resize(size=output_shape, antialias=True)
        self.noise_generator = PerlinNoise2D()

        result_image = self._make_fracture_image(generated_shape)

        result_image = torch.clip(result_image, 0, 1)
        result_image = self._heighten_contrast(
            result_image, self.heighten_contrast_range
        )

        result_image = reshape_transform(result_image.view((1,) + generated_shape))
        result_image = result_image.view(output_shape)

        if insert_blind_spots:
            return self._insert_blind_spots(result_image)
        else:
            return result_image
