import torch


def thresholds_transformations(x_tensor, thresholds, transformations=None):
    if transformations is None:
        transformations = thresholds + [None]

    for idx_transformation, transformation in enumerate(transformations):
        if transformation is not None:
            if idx_transformation == 0:
                mask = x_tensor <= thresholds[idx_transformation]
            elif idx_transformation < len(transformations):
                mask = (x_tensor > thresholds[idx_transformation - 1]) & (
                    x_tensor <= thresholds[idx_transformation]
                )
            else:
                mask = x_tensor > thresholds[idx_transformation - 1]

            x_tensor[mask] = (
                transformation(x_tensor[mask])
                if callable(transformation)
                else transformation
            )

    return x_tensor
