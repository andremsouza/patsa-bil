"""Utility functions for operating on image masks."""

# %% [markdown]
# ## Imports

# %%

import numpy as np

# %% [markdown]
# ## Functions

# %%


def get_bounding_box(
    mask: np.ndarray, background_value: int = 0, perturbation: int = 1
) -> tuple[int, int, int, int]:
    """Returns the bounding box of the mask.

    Args:
        mask (np.ndarray): The mask to get the bounding box from.
            Shape should be 2D, i.e., (height, width).
        background_value (int, optional): The value of the background in the mask.
            Defaults to 0.
        perturbation (int, optional): The perturbation to add to the bounding box coordinates.
            Defaults to 1.

    Returns:
        tuple[int, int, int, int]: The bounding box coordinates (x_min, y_min, x_max, y_max).
    """
    y_indices, x_indices = np.where(mask > background_value)
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    min_perturbation: int = 1 if perturbation > 1 else 0
    x_min = max(0, x_min - np.random.randint(min_perturbation, perturbation))
    x_max = min(W, x_max + np.random.randint(min_perturbation, perturbation))
    y_min = max(0, y_min - np.random.randint(min_perturbation, perturbation))
    y_max = min(H, y_max + np.random.randint(min_perturbation, perturbation))
    bbox = [x_min, y_min, x_max, y_max]

    return bbox
