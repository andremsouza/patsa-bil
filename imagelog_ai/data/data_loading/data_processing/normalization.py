"""This module contains functions for normalizing data."""

import numpy as np
import cv2

from imagelog_ai.data.data_loading.utils.enums import NormalizationMethod, WrappingType

INT_PIXEL_CONVERSION = (2**16) - 1
INT_PIXEL_TYPE = np.uint16


def global_normalization(data: np.ndarray):
    """Normalizes data to a [0, 1] range

    Parameters
    ----------
    data : ndarray
        Numpy array to be normalized.
    Returns
    ----------
    result : ndarray
        Normalized data of same shape as input, with values in the [0, 1] range.
    """
    min_value = np.min(data)
    max_value = np.max(data)

    return (data - min_value) / (max_value - min_value)


def quantile_normalization(data: np.ndarray, quantile_parameter: float = 0.001):
    """Normalizes data to a [0, 1] range

    Parameters
    ----------
    data : ndarray
        Numpy array to be normalized.
    quantile_parameter : float, default: 0.001
        Defines the quantiles used for normalization.
    Returns
    ----------
    result : ndarray
        Normalized data of same shape as input, with values in the [0, 1] range
    """
    quantiles = quantile_parameter, 1 - quantile_parameter
    min_value = np.quantile(data, quantiles[0])
    max_value = np.quantile(data, quantiles[1])

    return np.clip((data - min_value) / (max_value - min_value), 0, 1)


def histogram_normalization(data: np.ndarray):
    """Normalizes data to a [0, 1] range whilst making the histogram cdf linear.

    Parameters
    ----------
    data : ndarray
        Numpy array to be normalized.
    Returns
    ----------
    result : ndarray
        Normalized data of same shape as input, with values in the [0, 1] range
    """
    flattened_data = data.flatten()

    histogram_array = np.bincount(flattened_data, minlength=256)

    number_of_pixels = np.sum(histogram_array)
    histogram_array = histogram_array / number_of_pixels

    cumulative_histogram = np.cumsum(histogram_array)

    # Pixel mapping lookup table
    transform_map = np.floor(INT_PIXEL_CONVERSION * cumulative_histogram).astype(
        INT_PIXEL_TYPE
    )

    # transform pixel values to equalize
    equalized_image_list = [transform_map[pixel] for pixel in flattened_data]

    equalized_image = np.reshape(np.asarray(equalized_image_list), data.shape)

    return equalized_image


def wrapping_nothing(data: np.ndarray, wrap_size: int):
    """Returns the input data.
    Needed for "image_from_data"

    Parameters
    ----------
    data : (width, height, 3) or (width, height) ndarray
                    Input data.
    """
    return data


def wrapping_left(data: np.ndarray, wrap_size: int):
    """Wraps input data to create border on the left side

    Parameters
    ----------
    data : (width, height, 3) or (width, height) ndarray
                    Input data.
    """
    return cv2.copyMakeBorder(data, 0, 0, wrap_size, 0, borderType=cv2.BORDER_WRAP)


def wrapping_both(data: np.ndarray, wrap_size: int):
    """Wraps input data to create border on the left and right side

    Parameters
    ----------
    data : (width, height, 3) or (width, height) ndarray
                    Input data.
    """
    return cv2.copyMakeBorder(
        data, 0, 0, wrap_size, wrap_size, borderType=cv2.BORDER_WRAP
    )


def wrapping_right(data: np.ndarray, wrap_size: int):
    """Wraps input data to create border on the right side

    Parameters
    ----------
    data : (width, height, 3) or (width, height) ndarray
                    Input data.
    """
    return cv2.copyMakeBorder(
        data, 0, 0, wrap_size, wrap_size, borderType=cv2.BORDER_WRAP
    )


def normalizing_histogram(
    image_data: np.ndarray,
    interpolation_method,
    image_shape: tuple,
    normalization_first: bool,
):
    """Reshapes the input image data to the specified size.
    Normalizes data to a [0, 1] range whilst making the histogram cdf linear.

    Parameters
    ----------
    image_data : ndarray
        Numpy array to be normalized.
    interpolation_method: cv2.INTER_METHOD
        Interpolation method used for resizing of the data. Uses values directly from OpenCV.
    image_shape:
        Desired shape of the output
    normalization_first: bool
        Defines if the image will be normalized before the initial normalization.
    Returns
    ----------
    result : ndarray
        Normalized data of shape image_shape, with values in the [0, 1] range
    """
    if not isinstance(image_data, np.ndarray):
        raise TypeError("`image_data` should be an numpy.ndarray.")

    if not isinstance(interpolation_method, int):
        raise TypeError("`interpolation_method` should be an integer.")

    if isinstance(image_shape, tuple):
        if not isinstance(image_shape[0], int) or not isinstance(image_shape[1], int):
            raise TypeError("Tuple elements of `image_shape` should be integers.")

        if image_shape[0] < 1 or image_shape[1] < 1:
            raise ValueError(
                "Tuple elements of `image_shape` should be positive integers."
            )

    if int(interpolation_method) not in list(range(0, 8)):
        raise ValueError(
            "`interpolation_method` should be a valid cv2 interpolation method (0-7)."
        )

    if image_shape is None:
        image_data = quantile_normalization(image_data)
    elif normalization_first:
        image_data = quantile_normalization(image_data)
        image_data = cv2.resize(
            image_data, image_shape, interpolation=interpolation_method
        )
    else:
        image_data = cv2.resize(
            image_data, image_shape, interpolation=interpolation_method
        )
        image_data = quantile_normalization(image_data)

    image_data = (image_data * INT_PIXEL_CONVERSION).astype(INT_PIXEL_TYPE)

    image_data = histogram_normalization(image_data)

    return image_data


def normalizing_clahe(
    image_data: np.ndarray,
    interpolation_method,
    image_shape: tuple,
    normalization_first: bool,
):
    """Reshapes the input image data to the specified size.
    Normalizes data to a [0, 1] range whilst making the histogram cdf linear.

    Parameters
    ----------
    image_data : ndarray
        Numpy array to be normalized.
    interpolation_method: cv2.INTER_METHOD
        Interpolation method used for resizing of the data. Uses values directly from OpenCV.
    image_shape:
        Desired shape of the output
    normalization_first: bool
        Defines if the image will be normalized before the initial normalization.
    Returns
    ----------
    result : ndarray
        Normalized data of shape image_shape, with values in the [0, 1] range
    """

    if not isinstance(image_data, np.ndarray):
        raise TypeError("`image_data` should be an numpy.ndarray.")

    if not isinstance(interpolation_method, int):
        raise TypeError("`interpolation_method` should be an integer.")

    if isinstance(image_shape, tuple):
        if not isinstance(image_shape[0], int) or not isinstance(image_shape[1], int):
            raise TypeError("Tuple elements of `image_shape` should be integers.")

        if image_shape[0] < 1 or image_shape[1] < 1:
            raise ValueError(
                "Tuple elements of `image_shape` should be positive integers."
            )

    if int(interpolation_method) not in list(range(0, 8)):
        raise ValueError(
            "`interpolation_method` should be a valid cv2 interpolation method (0-7)."
        )

    if image_shape is None:
        image_data = quantile_normalization(image_data) * INT_PIXEL_CONVERSION
    elif normalization_first:
        image_data = quantile_normalization(image_data) * INT_PIXEL_CONVERSION
        image_data = cv2.resize(
            image_data.astype(INT_PIXEL_TYPE),
            image_shape,
            interpolation=interpolation_method,
        )
    else:
        image_data = cv2.resize(
            image_data, image_shape, interpolation=interpolation_method
        )
        image_data = quantile_normalization(image_data) * INT_PIXEL_CONVERSION

    image_data = image_data.astype(INT_PIXEL_TYPE)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image_data = clahe.apply(image_data).astype(np.float32)

    return image_data / INT_PIXEL_CONVERSION


def normalizing_quantile(
    image_data: np.ndarray,
    interpolation_method,
    image_shape: tuple,
    quantile_parameter: float,
):
    """Reshapes the input image data to the specified size.
    Normalizes data to a [0, 1] range whilst making the histogram cdf linear.

    Parameters
    ----------
    image_data : ndarray
        Numpy array to be normalized.
    interpolation_method: cv2.INTER_METHOD
        Interpolation method used for resizing of the data. Uses values directly from OpenCV.
    image_shape:
        Desired shape of the output
    quantile_parameter: float
        Value of the quantiles used in normalization.
    Returns
    ----------
    result : ndarray
        Normalized data of shape image_shape, with values in the [0, 1] range
    """

    if not isinstance(image_data, np.ndarray):
        raise TypeError("`image_data` should be an numpy.ndarray.")

    if not isinstance(interpolation_method, int):
        raise TypeError("`interpolation_method` should be an integer.")

    if not isinstance(image_shape, tuple):
        raise TypeError("`image_shape` should be a tuple.")

    if not isinstance(image_shape[0], int) or not isinstance(image_shape[1], int):
        raise TypeError("Tuple elements of `image_shape` should be integers.")

    if int(interpolation_method) not in list(range(0, 8)):
        raise ValueError(
            "`interpolation_method` should be a valid cv2 interpolation method (0-7)."
        )

    if image_shape[0] < 1 or image_shape[1] < 1:
        raise ValueError("Tuple elements of `image_shape` should be positive integers.")

    resized = cv2.resize(image_data, image_shape, interpolation=interpolation_method)
    resized = quantile_normalization(resized, quantile_parameter=quantile_parameter)

    return resized.astype(np.float32)


def normalizing_global(
    image_data: np.ndarray, interpolation_method, image_shape: tuple, **kwargs
):
    """Reshapes the input image data to the specified size.
    Normalizes data to a [0, 1] range whilst making the histogram cdf linear.

    Parameters
    ----------
    image_data : ndarray
        Numpy array to be normalized.
    interpolation_method: cv2.INTER_METHOD
        Interpolation method used for resizing of the data. Uses values directly from OpenCV.
    image_shape:
        Desired shape of the output
    Returns
    ----------
    result : ndarray
        Normalized data of shape image_shape, with values in the [0, 1] range
    """

    if not isinstance(image_data, np.ndarray):
        raise TypeError("`image_data` should be an numpy.ndarray.")

    if not isinstance(interpolation_method, int):
        raise TypeError("`interpolation_method` should be an integer.")

    if not isinstance(image_shape, tuple):
        raise TypeError("`image_shape` should be a tuple.")

    if not isinstance(image_shape[0], int) or not isinstance(image_shape[1], int):
        raise TypeError("Tuple elements of `image_shape` should be integers.")

    if int(interpolation_method) not in list(range(0, 8)):
        raise ValueError(
            "`interpolation_method` should be a valid cv2 interpolation method (0-7)."
        )

    if image_shape[0] < 1 or image_shape[1] < 1:
        raise ValueError("Tuple elements of `image_shape` should be positive integers.")

    resized = cv2.resize(image_data, image_shape, interpolation=interpolation_method)
    resized = global_normalization(resized)

    return resized.astype(np.float32)


def image_from_data(
    data: np.ndarray,
    image_upscaling: tuple = (1, 1),
    wrapping: WrappingType = WrappingType.BOTH,
    wrap_size: int = 0,
    normalization: NormalizationMethod = NormalizationMethod.HISTOGRAM,
    interpolation_method=cv2.INTER_LINEAR,
    **kwargs
):
    """Takes a numpy array and transforms it into an image in the [0, 1] range.
    Applies border wrapping, resizing and normalization.

    Parameters
    ----------
    data : Numpy array
        Numpy array of data.
    image_upscaling : tuple, default: (1,1)
        Ratio of upscaling applied to the data.

        Example: input(200, 10), img_upscaling(2, 5) -> scaled(400, 50)
    wrapping: string, default: 'both'
        What side of the image to add a wrapped border.
        Possible values are the available in WrappingType Enum.
    wrap_size: int, default: 1
        Size of the wrapping applied to the data.
        IMPORTANT: this is done before resizing, so keep in mind that the wrapping will be upscaled
    normalization_method: string, default: 'histogram'
        Method used for normalization.
        Possible values are the available in NormalizationMethod Enum.
    interpolation_method: cv2.INTER_METHOD, default: cv2.INTER_LINEAR
        Interpolation method used for resizing of the data. Uses values directly from OpenCV.
    normalization_first: bool
        The 'histogram' and 'CLAHE' normalization methods require the data to be put in a
        [0, 255] range.
        This decides if the data is resized before or after being put in this range.
    quantile_parameter: float
        Value of the quantiles used in normalization. Only applies for the 'quantile' norm_method.
    Returns
    -------
    result : Numpy float32 array
        Resulting image from the processing methods used.
    """
    # Behaviour when asking for a wrapping of size 0 is unclear
    # Changing the wrapping type to NOWRAP results in known behaviour
    # if (wrap_size == 0):
    #     wrapping = WrappingType.NOWRAP

    # wrapping_methods = {
    #     WrappingType.NOWRAP: wrapping_nothing,
    #     WrappingType.LEFT: wrapping_left,
    #     WrappingType.BOTH: wrapping_both,
    #     WrappingType.RIGHT: wrapping_right,
    # }

    # if wrapping not in wrapping_methods.keys():
    #     raise Exception("Invalid Wrapping. Must be from WrappingType Enum")

    # wrapped_data = wrapping_methods[wrapping](data, wrap_size)

    if image_upscaling[0] == 1 and image_upscaling[1] == 1:
        image_shape = None
    else:
        # OpenCV shape is (Width, Height), Numpy is (Heigth, Width)
        # Because of this shape[1] comes first in the tuple
        image_shape = tuple(
            [data.shape[1] * image_upscaling[1], data.shape[0] * image_upscaling[0]]
        )

    normalization_methods = {
        NormalizationMethod.HISTOGRAM: normalizing_histogram,
        NormalizationMethod.CLAHE: normalizing_clahe,
        NormalizationMethod.QUANTILE: normalizing_quantile,
        NormalizationMethod.GLOBAL: normalizing_global,
    }

    if normalization not in normalization_methods.keys():
        raise ValueError("Invalid Wrapping. Must be from WrappingType Enum")

    normalized_data = normalization_methods[normalization](
        data, interpolation_method, image_shape, normalization_first=True
    )

    return ((1 - global_normalization(normalized_data)) * INT_PIXEL_CONVERSION).astype(
        INT_PIXEL_TYPE
    )
