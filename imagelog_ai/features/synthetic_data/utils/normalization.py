from typing import Tuple

import torch


def min_max_normalize(x_tensor: torch.Tensor) -> torch.Tensor:
    x_tensor -= torch.min(x_tensor)
    x_tensor /= torch.max(x_tensor)

    return x_tensor


def color_range_scale(
    noise: torch.Tensor, color_range: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    noise *= color_range[1] - color_range[0]
    noise += color_range[0]
    return noise


def color_range_normalize(
    input_tensor: torch.Tensor, color_range: Tuple[float, float] = (0, 1)
) -> torch.Tensor:
    input_tensor = min_max_normalize(input_tensor)
    return color_range_scale(input_tensor, color_range)


def old_min_max_normalize(x_tensor: torch.Tensor) -> torch.Tensor:
    shape = x_tensor.size()
    x_tensor = x_tensor.view(shape[0], -1)

    x_tensor -= x_tensor.min(1, keepdim=True)[0]
    x_tensor /= x_tensor.max(1, keepdim=True)[0]

    return x_tensor.view(shape)
