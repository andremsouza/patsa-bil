from imagelog_ai.features.synthetic_data.imagelog_generator.BaseImagelogGenerator import (
    BaseImagelogGenerator,
)
from imagelog_ai.features.synthetic_data.imagelog_generator.utils.dataclass import (
    RegionProperties,
    PatternProperties,
    GranuleProperties,
)
from imagelog_ai.features.synthetic_data.imagelog_generator.utils.enum import (
    PatternType,
    BackgroundType,
)
from imagelog_ai.features.synthetic_data.utils.normalization import (
    min_max_normalize,
    color_range_scale,
)
from imagelog_ai.features.synthetic_data.utils.perlin_noise import PerlinNoise2D
from imagelog_ai.features.synthetic_data.utils.filter import gaussian_filter
from imagelog_ai.utils import argcheck

from torchvision.transforms import Resize
from abc import ABC, abstractmethod
from typing import List, Any, Tuple

import random
import torch
import math


class BaseImagelogLayersGenerator(BaseImagelogGenerator, ABC):
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

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self.background_generation_methods = {
            BackgroundType.BRECHADO: self._make_texture_brechado,
            BackgroundType.PERLIN: self._make_texture_perlin,
            BackgroundType.FLAT: self._make_texture_flat,
        }

        self.pattern_methods = {
            PatternType.SIMPLE: (self._simple_mask, None),
            PatternType.IMPULSE: (self._simple_mask, self._combined_texture_impulse),
            PatternType.SPOTS: (self._simple_mask, self._combined_texture_spots),
            PatternType.GRANULE: (self._simple_mask, self._granules_texture),
            PatternType.BRECHADO: (self._simple_mask, self._brechado_texture),
            PatternType.INVERSEBRECHADO: (
                self._simple_mask,
                self._inverse_brechado_texture,
            ),
        }
        self.granule_properties = [
            ### BLACK ###
            GranuleProperties(
                cutoff_step=0,
                initial_cutoff=0.99999,
                sigma_step=1,
                initial_sigma=1,
                repetitions=3,
                color_range=(0.8, 1.0),
                resolutions=(4, 4),
            ),
            ### BIGGER_BLACK ###
            GranuleProperties(
                cutoff_step=0,
                initial_cutoff=0.999999,
                sigma_step=0,
                initial_sigma=5,
                repetitions=1,
                color_range=(0.8, 1.0),
                resolutions=(4, 4),
            ),
        ]

        self.region_list = self._define_region_properties()
        self.y_values = None

    @abstractmethod
    def _define_region_properties(self) -> List[RegionProperties]:
        raise NotImplementedError

    @abstractmethod
    def _generate_image(self, image_shape: tuple) -> torch.Tensor:
        raise NotImplementedError

    def _smooth_heaviside(
        self, x: torch.Tensor, x0: float, delta: float = 1e-02
    ) -> torch.Tensor:
        return 1 / 2 + (1 / math.pi) * torch.arctan((x - x0) / delta)

    def _stacking_func_smooth_square_wave(
        self, x: torch.Tensor, middle: float, width: float, delta: float = 1e-02
    ) -> torch.Tensor:
        return self._smooth_heaviside(
            x, (middle - 0.5 * width), delta=delta
        ) - self._smooth_heaviside(x, (middle + 0.5 * width), delta=delta)

    def _directional_smooth_square_wave(
        self,
        x: torch.Tensor,
        middle: float,
        width: float,
        delta_1: float = 1e-02,
        delta_2: float = 1e-02,
    ) -> torch.Tensor:
        return self._smooth_heaviside(
            x, (middle - 0.5 * width), delta=delta_1
        ) - self._smooth_heaviside(x, (middle + 0.5 * width), delta=delta_2)

    def _make_brechado_mask(self, shape: Tuple[int, int]):
        number_of_windows = shape[0] // shape[1]

        noise_property_list = [
            [(10 * number_of_windows, 10), 1, 0.75],
            [(4 * number_of_windows, 4), 2, 0.75],
            [(10 * number_of_windows, 10), 2, 0.50],
            [(4 * number_of_windows, 4), 3, 0.50],
        ]
        intervals = torch.linspace(0, 1, 5, device=self.torch_device) ** 1.2

        result_image = torch.zeros(shape, device=self.torch_device)
        for properties in noise_property_list:
            image = (
                self._norm_fractal_noise(shape, properties[0], properties[1])
                ** properties[2]
            )
            for i in range(len(intervals) - 1):
                mask = (image > intervals[i]) & (image < intervals[i + 1])
                image[mask] = intervals[i + 1]
            image **= 0.7
            result_image += image

            del image
        result_image = min_max_normalize(result_image)

        intervals = torch.linspace(0, 1, 6, device=self.torch_device) ** 1.1

        for i in range(len(intervals) - 1):
            mask = (result_image > intervals[i]) & (result_image < intervals[i + 1])
            result_image[mask] = intervals[i + 1]

        return gaussian_filter(result_image, sigma=2)

    def _make_texture_flat(
        self, shape: Tuple[int, int], color_range: Tuple[float, float] = (0, 1)
    ):
        return torch.full(shape, color_range[1], device=self.torch_device)

    def _make_texture_perlin(
        self, shape: Tuple[int, int], color_range: Tuple[float, float] = (0, 1)
    ):
        number_of_windows = shape[0] // shape[1]
        perlin_map = self.noise_generator.generate_fractal_noise_2d(
            shape, (2 * number_of_windows, 5), 4
        )

        perlin_map = min_max_normalize(perlin_map)
        perlin_map[perlin_map <= 0.5] **= 1.2
        perlin_map[perlin_map > 0.5] **= 0.8

        perlin_map = gaussian_filter(perlin_map, 3)

        return color_range_scale(perlin_map, color_range)

    def _make_texture_brechado(
        self, shape: Tuple[int, int], color_range: Tuple[float, float] = (0, 1)
    ):
        mask = self._make_brechado_mask(shape)
        return color_range_scale(mask, color_range)

    def _make_texture_impulse(
        self, shape: Tuple[int, int], cutoff: float = 0.9, sigma: float = 2
    ):
        mask = torch.rand(shape[0], shape[1], device=self.torch_device)

        mask[(mask > cutoff)] = 1e4
        mask[mask <= cutoff] = 0

        mask = gaussian_filter(mask, sigma)

        mask = min_max_normalize(mask)
        mask *= 1.7

        return torch.clip(mask, 0, 1)

    def _simple_mask(
        self,
        output_mask: torch.Tensor,
        start_indexes: List[int],
        end_indexes: List[int],
        delta: float,
    ):
        image_shape = output_mask.shape

        for x_index in range(image_shape[1]):
            start_Y = start_indexes[x_index]
            end_Y = end_indexes[x_index]

            if int(start_Y) == int(end_Y):
                continue

            middle_Y = (start_Y + end_Y) / 2
            width_Y = end_Y - start_Y

            output_mask[:, x_index] += self._stacking_func_smooth_square_wave(
                self.y_values, middle_Y, width_Y, delta
            )

    def _make_texture_spots(
        self, shape: Tuple[int, int], cutoff: float = 0.8, sigma: float = 8
    ):
        number_of_windows = shape[0] // shape[1]
        noise_property_list = [
            [(10 * number_of_windows, 10), 1, 0.8],
            [(4 * number_of_windows, 4), 2, 0.8],
            [(10 * number_of_windows, 10), 2, 0.6],
            [(4 * number_of_windows, 4), 3, 0.6],
        ]

        mask = torch.zeros(shape, device=self.torch_device)
        for properties in noise_property_list:
            mask += (
                self._norm_fractal_noise(shape, properties[0], properties[1])
                ** properties[2]
            )

        mask -= torch.min(mask)
        mask /= torch.max(mask)

        mask[mask < cutoff] = 0

        mask *= 3
        mask = gaussian_filter(mask, sigma)

        return torch.clip(mask, 0, 1)

    def _make_texture_fractal_spots(self, shape, cutoff=0.95, repititions=8):
        mask = torch.zeros(shape, device=self.torch_device)

        for i in range(repititions):
            mask += self._make_texture_spots(shape, cutoff)

        mask *= 1.2

        return torch.clip(mask, 0, 1)

    def _combined_texture_impulse(
        self, background_texture: torch.Tensor, cutoff: float = 0.8, sigma: float = 3
    ):
        impulse_texture = self._make_texture_impulse(
            background_texture.shape, cutoff, sigma
        )

        return impulse_texture * background_texture

    def _combined_texture_spots(
        self, background_texture: torch.Tensor, cutoff=0.95, repititions: int = 10
    ):
        spots_texture = self._make_texture_fractal_spots(
            background_texture.shape, cutoff, repititions
        )

        return background_texture * (1 - spots_texture) + spots_texture * spots_texture

    def _clear_textures(self):
        del self.y_values
        self.y_values = None

        for region in self.region_list:
            del region.background_texture
            region.background_texture = None

            for pattern in region.patterns:
                del pattern.texture
                del pattern.mask

            region.patterns = []

    def _brechado_texture(self, background_texture: torch.Tensor, middle_point=0.85):
        shape = background_texture.shape
        number_of_windows = shape[0] // shape[1]
        low_interval = [0.21, 0.25]
        top_interval = [0.80, 1.00]

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

        result_image = gaussian_filter(result_image, sigma=4)

        return result_image

    def _inverse_brechado_texture(
        self, background_texture: torch.Tensor, middle_point=0.87
    ):
        reverse_brechado = 1 - self._brechado_texture(background_texture, middle_point)
        reverse_brechado += 0.1

        reverse_brechado /= torch.max(reverse_brechado)

        return background_texture * reverse_brechado

    def _make_texture_brechado(self, shape, color_range, middle_point=0.82):
        number_of_windows = shape[0] // shape[1]
        low_interval = [0.12, 0.10]
        top_interval = [0.80, 1.00]

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

        return gaussian_filter(result_image, sigma=4)

    def _grain_texture(self, img_shape, val_cutoff, sigma_size):
        x = torch.rand(img_shape, dtype=torch.float32, device=self.torch_device)

        x[x < val_cutoff] = 0
        x[x > val_cutoff] = 50 * sigma_size
        x = gaussian_filter(x, sigma_size)

        return torch.clip(x, 0, 1)

    def _granule_mask(self, img_shape: Tuple[int], properties: GranuleProperties):
        x = torch.zeros(img_shape, dtype=torch.float32, device=self.torch_device)

        curr_cutoff = properties.initial_cutoff
        curr_sigma = properties.initial_sigma

        for i in range(properties.repetitions):
            x += self._grain_texture(img_shape, curr_cutoff, curr_sigma)

            curr_cutoff += properties.cutoff_step
            curr_sigma += properties.sigma_step

        return torch.clip(x, 0, 1)

    def _granules_texture(self, background_texture: torch.Tensor):
        for properties in self.granule_properties:
            mask = self._granule_mask(background_texture.shape, properties)
            texture = self._norm_perlin_noise(
                background_texture.shape, properties.resolutions
            )
            texture = color_range_scale(texture, properties.color_range)

            background_texture = self._blend_tensors(texture, background_texture, mask)

        return background_texture

    def _setup_regions(
        self, image_shape: tuple, properties_list: List[RegionProperties]
    ):
        for region_properties in properties_list:
            region_properties.background_texture = self.background_generation_methods[
                region_properties.background_type
            ](image_shape, region_properties.color_range)

            region_properties.distance_values = self._bounded_smooth_cummulative_values(
                array_length=image_shape[0],
                value_range=region_properties.distance_range,
                speed=0.0021,
            ).astype(int)

            for pattern_type in region_properties.pattern_types:
                texture_functions = self.pattern_methods[pattern_type]

                pattern_properties = PatternProperties(
                    mask_function=texture_functions[0],
                    texture=None,
                    mask=torch.zeros(image_shape, device=self.torch_device),
                )

                if texture_functions[1] != None:
                    pattern_properties.texture = texture_functions[1](
                        region_properties.background_texture
                    )
                else:
                    pattern_properties.texture = region_properties.background_texture

                region_properties.patterns.append(pattern_properties)

    def _make_laminado_patterns_texture(
        self,
        image_shape: tuple,
        properties_list: List[RegionProperties],
        position_range: tuple = None,
        smooth_delta: float = 3,
        noise_strenght: float = 0.5,
        x_phase: float = math.pi / 2,
        amplitude_range: tuple = (10, 75),
    ) -> torch.Tensor:
        if position_range == None:
            position_range = (0, image_shape[1])

        perlin_values = self._norm_fractal_noise(
            image_shape, (image_shape[0] // image_shape[1], 5), 4
        )  # (H, 1) * 2^(4-1) = (8*H, 8)
        input_angles = (
            torch.linspace(0, 2 * math.pi, image_shape[1], device=self.torch_device)
            + x_phase
        )
        last_sine_positions = (
            torch.zeros((image_shape[1]), dtype=int, device=self.torch_device)
            + position_range[0]
        )
        last_center = position_range[0]

        self._setup_regions(image_shape, properties_list)

        amplitude_values = self._bounded_smooth_cummulative_values(
            array_length=image_shape[0], value_range=amplitude_range
        )
        phase_values = self._looping_smooth_cummulative_values(
            array_length=image_shape[0],
            max_value=2 * math.pi,
            variance=0.01,
            resolution=1,
            speed=0.001,
        )

        current_pattern = 0
        gen_count = 0
        while torch.min(last_sine_positions) < image_shape[0]:
            region_properties = properties_list[current_pattern]

            distance = region_properties.distance_values[last_center % image_shape[0]]
            center = last_center + distance

            noise_values = perlin_values[center % image_shape[0], :]
            amplitude = amplitude_values[center % image_shape[0]]
            phase = phase_values[center % image_shape[0]]

            sine_positions = torch.sin(
                (input_angles + phase) + noise_strenght * noise_values
            )
            sine_positions *= amplitude
            sine_positions += center
            sine_positions = torch.clip(
                sine_positions, min=0, max=image_shape[0]
            )  # Keep position limited to image

            pattern: PatternProperties = random.choices(
                region_properties.patterns, region_properties.pattern_chances
            )[0]

            pattern.mask_function(
                pattern.mask, last_sine_positions, sine_positions, smooth_delta
            )

            current_pattern += 1
            current_pattern %= len(properties_list)
            last_sine_positions = sine_positions
            last_amplitude = amplitude
            last_center = center
            gen_count += 1

        result_image = torch.zeros(image_shape, device=self.torch_device)

        for region_properties in properties_list:
            for pattern in region_properties.patterns:
                result_image += pattern.mask * pattern.texture

        return torch.clip(result_image, 0, 1)

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

        self.y_values = torch.linspace(
            0, generated_shape[0], generated_shape[0], device=self.torch_device
        )
        reshape_transform = Resize(size=output_shape, antialias=True)
        self.noise_generator = PerlinNoise2D()

        res_img = self._generate_image(generated_shape)
        self._clear_textures()

        res_img = reshape_transform(res_img.view((1,) + generated_shape))
        res_img = res_img.view(output_shape)

        if insert_blind_spots:
            return self._insert_blind_spots(res_img)
        else:
            return res_img
