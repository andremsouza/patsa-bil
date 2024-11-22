import torch
from torchvision.transforms import GaussianBlur


def gaussian_filter(
    input_tensor: torch.Tensor, sigma: float, truncate: float = 4, radius=None
) -> torch.Tensor:
    if radius is None:
        radius = round(truncate * sigma)

    kernel_size = 2 * (radius * 3) + 1

    gaussian_filter_torch = GaussianBlur(kernel_size=kernel_size, sigma=sigma)

    if len(input_tensor.shape) == 2:
        shape = list(input_tensor.shape)
        return gaussian_filter_torch(input_tensor.view([1] + shape)).view(shape)
    else:
        return gaussian_filter_torch(input_tensor)
