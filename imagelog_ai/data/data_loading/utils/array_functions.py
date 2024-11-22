"""This module contains functions for converting arrays to images and tensors."""

from typing import Callable

from matplotlib import pyplot as plt
import numpy as np
import torch

from imagelog_ai.data.data_loading.data_processing import normalization
from imagelog_ai.utils import argcheck


def matrix_2D_gray_1channel(matrix: np.ndarray):
    """Converts a 2D matrix to a 2D gray image with 1 channel.

    Args:
        matrix (np.ndarray): 2D matrix.

    Returns:
        np.ndarray: 2D gray image with 1 channel.
    """
    matrix = normalization.image_from_data(matrix)
    return matrix


def matrix_2D_gray_3channels(matrix: np.ndarray):
    """Converts a 2D matrix to a 2D gray image with 3 channels.

    Args:
        matrix (np.ndarray): 2D matrix.

    Returns:
        np.ndarray: 2D gray image with 3 channels.
    """
    matrix = normalization.image_from_data(matrix)
    matrix = np.repeat(matrix[..., np.newaxis], 3, axis=2)
    return matrix


def matrix_2D_colorized_YlOrBr(matrix: np.ndarray, colormap: Callable = plt.cm.YlOrBr):
    """Converts a 2D matrix to a 2D colorized image with 3 channels.

    Args:
        matrix (np.ndarray): 2D matrix.
        colormap (Callable): Colormap function.

    Returns:
        np.ndarray: 2D colorized image with 3 channels.
    """
    matrix = normalization.image_from_data(matrix, wrap_size=0)
    matrix = colormap(matrix)[:, :, :3]
    matrix = matrix.astype(np.float32)
    return matrix


def array_to_tiled_tensor(input_array: np.ndarray, tile_height: int) -> torch.Tensor:
    """Converts an array to a tiled tensor.

    Args:
        input_array (np.ndarray): Input array.
        tile_height (int): Tile height.

    Returns:
        torch.Tensor: Tiled tensor.
    """
    argcheck.is_instance(np.ndarray, input_tensor=input_array)
    argcheck.is_positive(int, tile_height=tile_height)

    if len(input_array.shape) == 3:
        input_array = np.transpose(input_array, (2, 0, 1))  # PyTorch formating
        num_tiles = input_array.shape[-2] // tile_height
    else:
        num_tiles = input_array.shape[0] // tile_height

    input_shape = input_array.shape
    end = num_tiles * tile_height  # Drop last tile if it is incomplete.

    if len(input_shape) == 1:
        output_shape = [num_tiles, tile_height]
    elif len(input_shape) == 2:
        output_shape = [num_tiles, tile_height, input_shape[-1]]
    else:
        output_shape = np.concatenate(
            (input_shape[:-2], [num_tiles, tile_height], [input_shape[-1]])
        )
    if len(input_shape) == 1:
        input_array = input_array[:end]
    else:
        input_array = input_array[..., :end, :]

    output_shape = tuple(output_shape)

    # Convert from `ndarray` to `Tensor` and apply vertical split using `Tensor.view` because
    # it is faster and uses less memory.
    tiled_tensor = torch.from_numpy(input_array).view(output_shape)

    if len(input_shape) > 2:
        tiled_tensor = torch.movedim(tiled_tensor, -3, 0)

    # Return tiles.
    return tiled_tensor


def array_to_tiles(input_array: np.ndarray, tile_height: int) -> np.ndarray:
    """Converts an array to tiles.

    Args:
        input_array (np.ndarray): Input array.
        tile_height (int): Tile height.

    Returns:
        np.ndarray: Tiles.
    """
    argcheck.is_instance(np.ndarray, input_tensor=input_array)
    argcheck.is_positive(int, tile_height=tile_height)

    input_shape = input_array.shape
    num_tiles = input_shape[0] // tile_height

    if len(input_shape) == 1:
        output_shape = (num_tiles, tile_height)
    else:
        output_shape = np.concatenate(([num_tiles, tile_height], input_shape[1:]))
        output_shape = tuple(output_shape)

    end = num_tiles * tile_height
    input_array = input_array[:end, ...]  # Drop last tile if it is incomplete.

    # Apply vertical split using `numpy.reshape`
    tiled_array = np.reshape(input_array, output_shape)

    # Return tiles.
    return tiled_array
