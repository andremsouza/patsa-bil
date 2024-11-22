"""This module contains utility functions for working with run-length encoded (RLE) masks.
"""

from label_studio_sdk.converter.brush import decode_rle
import numpy as np


def rle_to_mask(rle: list[int], height: int, width: int, print_params: bool = False):
    """
    Converts a Label-Studio run-length encoded (RLE) mask to a binary mask.

    Args:
        rle (list[int]): The run-length encoded mask.
        height (int): The height of the mask.
        width (int): The width of the mask.
        print_params (bool, optional): Whether to print the parameters. Defaults to False.

    Returns:
        numpy.ndarray: The binary mask.
    """
    out = decode_rle(rle, print_params=print_params)
    image = np.reshape(out, [height, width, 4])[:, :, 3]
    return image
