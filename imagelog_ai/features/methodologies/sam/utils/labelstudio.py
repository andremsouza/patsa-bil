"""Label Studio utility functions for the SAM methodologies."""

import numpy as np

from imagelog_ai.utils.rle_functions import rle_to_mask


def get_masks_from_annotation(annotation: dict) -> dict[str, np.ndarray]:
    """Load and join masks from Label Studio annotation, by label.

    Args:
        annotation (dict): Label Studio annotation.

    Returns:
        dict[str, np.ndarray]: Dictionary of masks, by label.
    """
    masks = {}
    for brushlabel in annotation:
        decoded_mask = rle_to_mask(
            brushlabel["value"]["rle"],
            brushlabel["original_height"],
            brushlabel["original_width"],
        )
        # Get labels
        labels = brushlabel["value"]["brushlabels"]
        for label in labels:
            if label not in masks:
                masks[label] = np.zeros_like(decoded_mask)
            masks[label] = np.logical_or(masks[label], decoded_mask)
    return masks
