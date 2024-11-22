"""This module contains the endpoints for the imagelog_generator API."""

import os
from typing import Any, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from tqdm import tqdm

from imagelog_ai.features.synthetic_data.imagelog_generator.BaseImagelogGenerator import (
    BaseImagelogGenerator,
)
from imagelog_ai.features.synthetic_data.imagelog_generator.BrechadoGenerators import (
    CaoticoGenerator,
    FraturadoGenerator,
)
from imagelog_ai.features.synthetic_data.imagelog_generator.ImagelogLayersGenerators import (
    MacicoGenerator,
    LaminadoGenerator,
)


def _imagelog_generator(
    generators: List[BaseImagelogGenerator],
    insert_blind_spots: bool,
    number_of_tiles: int,
    apply_colormap: bool,
    dataset_name: str,
    batch_size: int,
    **kwargs: Any,
):
    number_of_batches = number_of_tiles // batch_size
    image_count = 0

    for generator in generators:
        folder_path = f"data/interim/{dataset_name}/{generator.pattern_name}"
        os.makedirs(folder_path, exist_ok=True)

        iterator = tqdm(
            range(number_of_batches), desc=f"Generating {generator.pattern_name}"
        )

        for _ in iterator:
            tiled_images = (
                generator.generate_imagelog(batch_size, insert_blind_spots)
                .cpu()
                .numpy()
            )

            if apply_colormap:
                tiled_images = plt.cm.YlOrBr(tiled_images)[..., :3]
            else:
                tiled_images = np.repeat(tiled_images[..., np.newaxis], 3, axis=2)
                tiled_images = 1 - tiled_images

            tiled_images = (tiled_images * 255).astype(np.uint8)

            # Apply vertical split using `np.reshape`
            output_shape = (
                batch_size,
                -1,
                tiled_images.shape[-2],
                tiled_images.shape[-1],
            )
            tiled_images = np.reshape(tiled_images, output_shape)

            for image in tiled_images:
                image = Image.fromarray(image)
                # image = reshape_transform(image)
                image.save(f"{folder_path}/{image_count:06d}.png")
                image_count += 1


def synthetic_imagelog_generation(
    generated_tile_shape: Tuple[int, int],
    output_tile_shape: Tuple[int, int],
    **kwargs: Any,
) -> None:
    """Call the prediction using transfer learning model."""
    generators: List[BaseImagelogGenerator] = [
        CaoticoGenerator(
            generated_tile_shape=generated_tile_shape,
            output_tile_shape=output_tile_shape,
        ),
        FraturadoGenerator(
            generated_tile_shape=generated_tile_shape,
            output_tile_shape=output_tile_shape,
        ),
        MacicoGenerator(
            generated_tile_shape=generated_tile_shape,
            output_tile_shape=output_tile_shape,
        ),
        LaminadoGenerator(
            generated_tile_shape=generated_tile_shape,
            output_tile_shape=output_tile_shape,
        ),
    ]

    kwargs["insert_blind_spots"] = not kwargs["no_blind_spots"]

    _imagelog_generator(generators, **kwargs)


def test_brechado(
    generated_tile_shape: Tuple[int, int],
    output_tile_shape: Tuple[int, int],
    **kwargs: Any,
) -> None:
    """Call the prediction using transfer learning model."""
    generators: List[BaseImagelogGenerator] = [
        CaoticoGenerator(
            generated_tile_shape=generated_tile_shape,
            output_tile_shape=output_tile_shape,
        )
    ]
    kwargs["insert_blind_spots"] = not kwargs["no_blind_spots"]

    _imagelog_generator(generators, **kwargs)


def test_macico(
    generated_tile_shape: Tuple[int, int],
    output_tile_shape: Tuple[int, int],
    **kwargs: Any,
) -> None:
    """Call the prediction using transfer learning model."""
    generators: List[BaseImagelogGenerator] = [
        MacicoGenerator(
            generated_tile_shape=generated_tile_shape,
            output_tile_shape=output_tile_shape,
        )
    ]
    kwargs["insert_blind_spots"] = not kwargs["no_blind_spots"]

    _imagelog_generator(generators, **kwargs)


def test_fracture(
    generated_tile_shape: Tuple[int, int],
    output_tile_shape: Tuple[int, int],
    **kwargs: Any,
) -> None:
    """Call the prediction using transfer learning model."""
    generators: List[BaseImagelogGenerator] = [
        FraturadoGenerator(
            generated_tile_shape=generated_tile_shape,
            output_tile_shape=output_tile_shape,
        )
    ]
    kwargs["insert_blind_spots"] = not kwargs["no_blind_spots"]

    _imagelog_generator(generators, **kwargs)


def test_laminado(
    generated_tile_shape: Tuple[int, int],
    output_tile_shape: Tuple[int, int],
    **kwargs: Any,
) -> None:
    """Call the prediction using transfer learning model."""
    generators: List[BaseImagelogGenerator] = [
        LaminadoGenerator(
            generated_tile_shape=generated_tile_shape,
            output_tile_shape=output_tile_shape,
        )
    ]
    kwargs["insert_blind_spots"] = not kwargs["no_blind_spots"]

    _imagelog_generator(generators, **kwargs)
