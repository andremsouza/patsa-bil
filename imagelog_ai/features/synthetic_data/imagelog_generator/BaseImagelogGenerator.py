from imagelog_ai.features.synthetic_data.utils.normalization import (
    min_max_normalize,
    color_range_scale,
)
from imagelog_ai.features.synthetic_data.utils.perlin_noise import PerlinNoise2D

from imagelog_ai.utils import argcheck

from typing import Tuple, Any
from abc import ABC, abstractmethod

import numpy as np
import random
import torch
import math


class BaseImagelogGenerator(ABC):
    """Module to use with pre-trained networks for Transfer Learning on new classification problems.


    Parameters
    ----------
    heighten_contrast_range: Tuple[float, float]
            Range used when heightening contrast in output image
    tile_height: int
            Height of tile in pixels
    tile_width: int
            Width of tile in pixels
    pattern_name: String
            Name of pattern being generated
    """

    def __init__(
        self,
        generated_tile_shape: Tuple[int, int],
        output_tile_shape: Tuple[int, int],
        heighten_contrast_range: Tuple[float, float] = (0.6, 1.1),
        **kwargs: Any
    ) -> None:
        """Initialize BaseImagelogGenerator

        Parameters
        ----------
        tile_shape: Tuple[int, int]
                Shape of the output tile
        heighten_contrast_range: Tuple[float, float]
                Range used when heightening contrast in output image
        """
        argcheck.is_sequence(
            float, sequence_length=2, heighten_contrast_range=heighten_contrast_range
        )
        argcheck.is_sequence(int, sequence_length=2, tile_shape=generated_tile_shape)

        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.noise_generator: PerlinNoise2D = None
        self.heighten_contrast_range = heighten_contrast_range
        self.generated_tile_shape = generated_tile_shape
        self.output_tile_shape = output_tile_shape
        self.pattern_name = "EMPTY"

    def print_device(self):
        print("Using device:", self.torch_device)
        if self.torch_device.type == "cuda":
            print(torch.cuda.get_device_name(0))
            print("Memory Usage:")
            print(
                "Allocated:", round(torch.cuda.memory_allocated(0) / 1024**3, 1), "GB"
            )
            print(
                "Cached:   ", round(torch.cuda.memory_reserved(0) / 1024**3, 1), "GB"
            )

    def _norm_fractal_noise(self, shape, repetitions, octaves: int) -> torch.Tensor:
        """

        Parameters
        ----------
        tile_shape: Tuple[int, int]
                Shape of the output tile
        heighten_contrast_range: Tuple[float, float]
                Range used when heightening contrast in output image
        """
        noise = self.noise_generator.generate_fractal_noise_2d(
            shape, repetitions, octaves
        )

        return min_max_normalize(noise)

    def _norm_perlin_noise(self, shape, repetitions) -> torch.Tensor:
        """

        Parameters
        ----------
        tile_shape: Tuple[int, int]
                Shape of the output tile
        heighten_contrast_range: Tuple[float, float]
                Range used when heightening contrast in output image
        """
        noise = self.noise_generator.perlin_noise(shape, repetitions)

        return min_max_normalize(noise)

    def _apply_variance(self, input_array: np.ndarray, variance: float) -> np.ndarray:
        variance_array = np.random.rand(input_array.shape[0])  # [0, 1[
        variance_array *= 2 * variance  # [0, 2*variance[
        variance_array += 1 - variance  # [-variance, +variance[

        return np.multiply(input_array, variance_array)  # Applies variance to values

    def _bounded_smooth_cummulative_values(
        self,
        array_length: int,
        value_range: tuple,
        variance: float = 0.05,
        speed: float = 0.001,
        resolution: int = 5,
    ) -> np.ndarray:
        noise = self.noise_generator.perlin_noise(
            shape=(array_length, 1), resolutions=(resolution, 1)
        )
        noise *= speed * (value_range[1] - value_range[0])
        noise = noise.cpu().numpy()

        result = np.zeros(shape=(array_length,), dtype=np.float32)
        curr_value = random.random()
        curr_value *= value_range[1] - value_range[0]
        curr_value += value_range[0]
        flip_val = 1

        for pos in range(array_length):
            result[pos] = curr_value

            curr_value += flip_val * noise[pos]
            if curr_value > value_range[1]:
                curr_value = value_range[1] - (curr_value % value_range[1])
                flip_val *= -1
            elif curr_value < value_range[0]:
                curr_value = value_range[0] + (value_range[0] - curr_value)
                flip_val *= -1

        if variance != 0:
            result = self._apply_variance(result, variance)
            result = np.clip(result, a_min=value_range[0], a_max=value_range[1])

        return result

    def _binary_random_cummulative_values(
        self, array_length, cutoff, resolution=2, speed=0.1
    ):
        rand_values = self._bounded_smooth_cummulative_values(
            array_length=array_length,
            value_range=(0, 1),
            resolution=resolution,
            speed=speed,
        )

        rand_values[rand_values < cutoff] = 0
        rand_values[rand_values != 0] = 1
        return rand_values.astype(int)

    def _looping_smooth_cummulative_values(
        self,
        array_length: int,
        max_value: float,
        variance: float = 0.05,
        speed: float = 0.001,
        resolution: int = 5,
    ) -> np.ndarray:
        noise = self.noise_generator.perlin_noise(
            shape=(array_length, 1), resolutions=(resolution, 1)
        )
        noise *= speed * max_value
        noise = noise.cpu().numpy()

        result = np.zeros(shape=(array_length,), dtype=np.float32)
        curr_value = random.random() * max_value

        for pos in range(array_length):
            result[pos] = curr_value
            curr_value += noise[pos]

        result %= max_value

        if variance != 0:
            result = self._apply_variance(result, variance)

        return result

    def _blend_tensors(self, tensor_1, tensor_2, mask):
        return torch.multiply(tensor_1, mask) + torch.multiply(tensor_2, 1 - mask)

    def _insert_blind_spots(self, original_image: torch.Tensor) -> torch.Tensor:
        image = original_image.clone()
        shape = list(image.size())

        y_step = shape[1] / 12

        collumn_centers = [
            int(2 * y_step),
            int(5 * y_step),
            int(8 * y_step),
            int(11 * y_step),
        ]

        gap_positions = [
            int(0.5 * y_step),
            int(3.5 * y_step),
            int(6.5 * y_step),
            int(9.5 * y_step),
        ]

        for x_position in collumn_centers:
            image[
                :, x_position - int(y_step * 0.45) : x_position + int(y_step * 0.45)
            ] = 0

        for x_position in gap_positions:
            image[:, x_position - int(y_step / 12) : x_position + int(y_step / 12)] = 0

        return image

    def _heighten_contrast(
        self, img: torch.Tensor, exponent_range: Tuple[float, float]
    ) -> torch.Tensor:
        powers = torch.cos(img * math.pi)  # [-1, 1]
        powers = min_max_normalize(powers)
        powers = color_range_scale(powers, exponent_range)
        return torch.pow(img, powers)

    @abstractmethod
    def generate_imagelog(
        self, tile_batch_size: int, insert_blind_spots: bool
    ) -> torch.Tensor:
        raise NotImplementedError
