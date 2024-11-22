"""This module provides various data preprocessing transforms.
"""

from abc import ABC, abstractmethod
from typing import Callable, Optional, Tuple, Union

import bm3d
import cv2
import matplotlib.cm as cm
import numpy as np
from PIL import Image
from skimage.restoration import (
    denoise_tv_chambolle,  # pylint: disable=no-name-in-module
    denoise_wavelet,  # pylint: disable=no-name-in-module
)
import torch
from torch import Tensor
import torchvision.transforms.v2 as T

from imagelog_ai.data.data_preprocessing.utils.enums import DenoiserMethod

INT_PIXEL_CONVERSION = 255


class RepeatChannels(torch.nn.Module):
    """Class used to repeat the channels of an image.

    Parameters
    ----------
    n_repeat : int
        Number of times to repeat the channels.

    Methods
    ----------
    forward
        Repeat the channels of the input tensor image.
    """

    def __init__(self, n_repeat: int):
        super().__init__()
        self.n_repeat = n_repeat
        self.transformation = T.Lambda(lambda x: x.repeat(self.n_repeat, 1, 1))

    def forward(self, tensor: Tensor) -> Tensor:
        return self.transformation(tensor)

    def __repr__(self) -> str:
        detail = f"(n_repeat={self.n_repeat})"
        return f"{self.__class__.__name__}{detail}"


class GetChannel(torch.nn.Module):
    """Class used to get a specific channel of an image.

    Parameters
    ----------
    channel : int
        The channel to get.

    Methods
    ----------
    forward
        Get the specific channel of the input tensor image.
    """

    def __init__(self, channel: int):
        super().__init__()
        self.channel = channel

    def forward(self, tensor: Tensor) -> Tensor:
        return tensor[self.channel].unsqueeze(0)

    def __repr__(self) -> str:
        detail = f"(channel={self.channel})"
        return f"{self.__class__.__name__}{detail}"


class Denoiser(torch.nn.Module):
    """Class used to denoise images.


    Parameters
    ----------
    denoiser_method : DenoiserEnum
            Method used to denoise image. Options: DenoiserEnum.GAUSSIAN, DenoiserEnum.BILATERAL,
    DenoiserEnum.TOTAL_VARIATION, DenoiserEnum.WAVELET, DenoiserEnum.NON_LOCAL_MEANS,
    DenoiserEnum.BM3D.

    Methods
    ----------
    denoise
            Denoises the image.

    gaussian_denoise
            Perform Gaussian denoise on image.

    bilateral_denoise
            Perform Bilateral denoise on image.

    total_variation_denoise
            Perform Total Variation denoise on image.

    wavelet_denoise
            Perform Wavelet denoise on image.

    non_local_means_denoise
            Perform Non Local Means denoise on image.

    bm3d_denoise
            Perform BM3D denoise on image.
    """

    def __init__(self, denoiser_method: DenoiserMethod, **kwargs):
        """
        Initiate class to denoise images.

        Parameters
        ----------
        denoiser_method : DenoiserEnum
                Method used to denoise image. Options: DenoiserEnum.GAUSSIAN, DenoiserEnum.BILATERAL,
        DenoiserEnum.TOTAL_VARIATION, DenoiserEnum.WAVELET, DenoiserEnum.NON_LOCAL_MEANS,
        DenoiserEnum.BM3D.
        """
        super().__init__()

        denoiser_methods = {
            DenoiserMethod.TOTAL_VARIATION: self.total_variation_denoise,
            DenoiserMethod.NON_LOCAL_MEANS: self.non_local_means_denoise,
            DenoiserMethod.BILATERAL: self.bilateral_denoise,
            DenoiserMethod.GAUSSIAN: self.gaussian_denoise,
            DenoiserMethod.WAVELET: self.wavelet_denoise,
            DenoiserMethod.BM3D: self.bm3d_denoise,
        }

        if denoiser_method not in denoiser_methods.keys():
            try:
                denoiser_method = DenoiserMethod[denoiser_method]
            except KeyError as exc:
                raise KeyError(
                    f"Exception: {exc}\n"
                    "Invalid denoiser method, choose one of the DenoiserEnum options."
                ) from exc

        self.denoiser_function: Callable[..., np.ndarray] = denoiser_methods[denoiser_method]  # type: ignore
        self.denoiser_method = denoiser_method
        self.kwargs = kwargs

    def forward(self, tensor: Tensor, **kwargs) -> Tensor:
        """
        Denoises the ``image`` according to the ``denoiser_method`` attribute.

        Parameters
        ----------
        image : array
                Noise image that will be denoised.

        Each method parameter is described in the method Docstring.
        """
        image = tensor.numpy()

        if len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))

        denoised_image = self.denoiser_function(image=image, **self.kwargs)
        denoised_image = np.transpose(denoised_image, (2, 0, 1))

        return torch.from_numpy(denoised_image)

    def gaussian_denoise(
        self,
        image: np.ndarray,
        kernel_size: tuple[int, int] = (5, 5),
        x_gaussian_standard_deviation: float = 0,
        y_gaussian_standard_deviation: float = 0,
        border_type=cv2.BORDER_REFLECT101,
        **kwargs,
    ):
        """
        Perform Gaussian denoise on image.

        Parameters
        ----------
        image : array
                Noise image that will be denoised.

        kernel_size : tuple, default: (5, 5)
                Positive and odd numbers,  Gaussian kernel size.
                ksize.width and ksize.height can differ but they both must be positive and odd.
                Or, they can be zero's and then they are computed from sigma.

        x_gaussian_standard_deviation : float, default: 0
                Gaussian kernel standard deviation in X direction.

        y_gaussian_standard_deviation : float, default: 0
                Gaussian kernel standard deviation in Y direction; if sigmaY is zero, it is set
                to be equal to sigmaX,
        if both sigmas are zeros, they are computed from ksize.width and ksize.height,
        respectively (see #getGaussianKernel for details); to fully control the result regardless
        of possible future modifications of all this semantics, it is recommended
        to specify all of ksize, sigmaX, and sigmaY.

        border_type : cv2.Border, default: cv2.BORDER_REFLECT101
                Pixel extrapolation method, see cv2 BorderTypes. cv2.BORDER_WRAP is not supported.
        """

        denoised_image = cv2.GaussianBlur(
            src=image,
            ksize=kernel_size,
            sigmaX=x_gaussian_standard_deviation,
            sigmaY=y_gaussian_standard_deviation,
            borderType=border_type,
        )

        return denoised_image

    def bilateral_denoise(
        self,
        image: np.ndarray,
        diameter: int = 5,
        sigma_color: float = 10,
        sigma_space: float = 10,
        border_type=cv2.BORDER_REFLECT101,
        **kwargs,
    ):
        """
        Perform Bilateral denoise on image.

        Parameters
        ----------
        image : array
                Noise image that will be denoised.

        diameter : int, default: 5
                Diameter of each pixel neighborhood that is used during filtering.
                If it is non-positive, it is computed from sigmaSpace.

        sigma_color : float, default: 10
                Filter sigma in the color space. A larger value of the parameter means that
                farther colors within the pixel neighborhood (see sigmaSpace) will be mixed
                together, resulting in larger areas of semi-equal color.

        sigma_space : float, default: 10
                Filter sigma in the coordinate space. A larger value of the parameter means
                that farther pixels will influence each other as long as their colors are close
                enough (see sigmaColor ). When d&gt;0, it specifies the neighborhood
                size regardless of sigmaSpace. Otherwise, d is proportional to sigmaSpace.

        border_type : cv2.Border, default: cv2.BORDER_REFLECT101
                Pixel extrapolation method, see cv2 BorderTypes. cv2.BORDER_WRAP is not supported.
        """

        denoised_image = cv2.bilateralFilter(
            src=image,
            d=diameter,
            sigmaColor=sigma_color,
            sigmaSpace=sigma_space,
            borderType=border_type,
        )

        return denoised_image

    def total_variation_denoise(
        self,
        image: np.ndarray,
        weight: float = 0.1,
        stop_criterion: float = 0.0002,
        maximum_iterations_number: int = 200,
        channel_axis: Optional[int] = None,
        **kwargs,
    ):
        """
        Perform Total Variation denoise on image.

        Parameters
        ----------
        image : array
                Noise image that will be denoised.

        weight : float, default: 0.1
                Denoising weight. The greater weight, the more denoising
                (at the expense of fidelity to input).

        stop_criterion : float, default: 0.0002
                Relative difference of the value of the cost function that determines the stop
                criterion. The algorithm stops when: (E_(n-1) - E_n) < eps * E_0.

        maximum_iterations_number : int, default: 200
                Maximal number of iterations used for the optimization.

        channel_axis : int or None, default: None
                If None, the image is assumed to be a grayscale (single channel) image. Otherwise,
                this parameter indicates which axis of the array corresponds to channels.
        """

        denoised_image = denoise_tv_chambolle(
            image,
            weight=weight,
            eps=stop_criterion,
            max_num_iter=maximum_iterations_number,
            channel_axis=channel_axis,
        )

        return denoised_image

    def wavelet_denoise(
        self,
        image: np.ndarray,
        noise_standard_deviation: Optional[float] = None,
        wavelet_type: str = "db1",
        mode: str = "soft",
        method: str = "BayesShrink",
        wavelet_levels: Optional[int] = None,
        convert_to_ycbcr: bool = False,
        rescale_sigma: bool = True,
        channel_axis: Optional[int] = None,
        **kwargs,
    ):
        """
        Perform Wavelet denoise on image.

        Parameters
        ----------
        image : array
                Noise image that will be denoised.

        noise_standard_deviation : Float or None, default: None
                The noise standard deviation used when computing the wavelet detail coefficient
                threshold(s). When None (default), the noise standard deviation is estimated via
                the method in [2]_.

        wavelet_type : str, default: 'db1'
                The type of wavelet to perform and can be any of the options pywt.wavelist
                outputs. The default is 'db1'. For example, wavelet can be any of
                {'db2', 'haar', 'sym9'} and many more.

        mode : str, default: 'soft'
                An optional argument to choose the type of denoising performed. It noted that
                choosing soft thresholding given additive noise finds the best approximation
                of the original image.

        method : str, default: 'BayesShrink'
                Thresholding method to be used. The currently supported methods are
                "BayesShrink" [1]_ and "VisuShrink" [2]_. Defaults to "BayesShrink".

        wavelet_levels : int or None, default: None
                The number of wavelet decomposition levels to use. The default is three less than
                the maximum number of possible decomposition levels.

        convert_to_ycbcr : bool, default: False
                If True and multichannel True, do the wavelet denoising in the YCbCr colorspace
                instead of the RGB color space. This typically results in better performance for
                RGB images.

        rescale_sigma : bool, default: True
                If False, no rescaling of the user-provided sigma will be performed. The default
                of True rescales sigma appropriately if the image is rescaled internally.
                (Only on Wavelet Method)

        channel_axis : int, default: None
                If None, the image is assumed to be a grayscale (single channel) image.
                Otherwise, this parameter indicates which axis of the array corresponds to channels
        """

        denoised_image = denoise_wavelet(
            image,
            sigma=noise_standard_deviation,
            wavelet=wavelet_type,
            mode=mode,
            method=method,
            wavelet_levels=wavelet_levels,
            convert2ycbcr=convert_to_ycbcr,
            rescale_sigma=rescale_sigma,
            channel_axis=channel_axis,
        )

        return denoised_image

    def non_local_means_denoise(
        self,
        image: np.ndarray,
        filter_strength: float = 3,
        template_window_size: int = 7,
        search_window_size: int = 21,
        **kwargs,
    ):
        """
        Perform Non Local Means denoise on image.

        Parameters
        ----------
        image : array
                Noise image that will be denoised.

        filter_strength : float, default: 3
                Parameter regulating filter strength. Big ``filter_strength`` value perfectly
                removes noise but also removes image details, smaller ``filter_strength`` value
                preserves details but also preserves some noise.

        template_window_size : int, default: 7
                Size in pixels of the template patch that is used to compute weights.
                Should be odd. Recommended value 7 pixels.

        search_window_size : int, default: 21
                Size in pixels of the window that is used to compute weighted average for
                given pixel.
                Should be odd.
                Affect performance linearly: greater searchWindowsSize - greater denoising time.
                Recommended value 21 pixels.
        """

        denoised_image = (
            cv2.fastNlMeansDenoising(
                src=np.uint8(image * 255),
                h=filter_strength,
                templateWindowSize=template_window_size,
                searchWindowSize=search_window_size,
            )
            / 255
        )

        return denoised_image

    def bm3d_denoise(
        self,
        image: np.ndarray,
        noise_standard_deviation: float = 0.2,
        profile: str = "np",
        stage_argument: bm3d.BM3DStages = bm3d.BM3DStages.ALL_STAGES,
        blockmatches=(False, False),
        **kwargs,
    ):
        """
        Perform BM3D denoise on image.

        Parameters
        ----------
        image : array
                Noise image that will be denoised.

        noise_standard_deviation : float, default: 0.2
                Noise PSD, either MxN or MxNxC (different PSDs for different channels)
                or sigma_psd: Noise standard deviation, either float, or length C list of floats.

        profile : str, default: 'np'
                Settings for BM3D: BM3DProfile object or a string
                ('np', 'refilter', 'vn', 'vn_old', 'high', 'deb'). Default 'np'.

        stage_argument : bm3d stages, default: bm3d.BM3DStages.ALL_STAGES
                Determines whether to perform hard-thresholding or wiener filtering.
                either BM3DStages.HARD_THRESHOLDING, BM3DStages.ALL_STAGES or an estimate of the
                noise-free image.
                - BM3DStages.ALL_STAGES: Perform both.
                - BM3DStages.HARD_THRESHOLDING: Perform hard-thresholding only. - ndarray,
                size of z: Perform Wiener Filtering with stage_arg as pilot.

        blockmatches : tuple, default: (False, False)
                denoised image, same size as z: if blockmatches == (False, False) denoised image,
                blockmatch data: if either element of blockmatches is True.
        """

        denoised_image = bm3d.bm3d(
            image,
            sigma_psd=noise_standard_deviation,
            profile=profile,
            stage_arg=stage_argument,
            blockmatches=blockmatches,
        )

        return denoised_image

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(method={self.denoiser_method})"


class EqualizerCLAHE(torch.nn.Module):
    """Class used to denoise images.


    Parameters
    ----------
    denoiser_method : DenoiserEnum
            Method used to denoise image. Options: DenoiserEnum.GAUSSIAN, DenoiserEnum.BILATERAL,
    DenoiserEnum.TOTAL_VARIATION, DenoiserEnum.WAVELET, DenoiserEnum.NON_LOCAL_MEANS,
    DenoiserEnum.BM3D.

    Methods
    ----------
    denoise
            Denoises the image.

    gaussian_denoise
            Perform Gaussian denoise on image.

    bilateral_denoise
            Perform Bilateral denoise on image.

    total_variation_denoise
            Perform Total Variation denoise on image.

    wavelet_denoise
            Perform Wavelet denoise on image.

    non_local_means_denoise
            Perform Non Local Means denoise on image.

    bm3d_denoise
            Perform BM3D denoise on image.
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: Tuple[int, int] = (8, 8),
        **kwargs,
    ):
        """
        Initiate class to denoise images.

        Parameters
        ----------
        denoiser_method : DenoiserEnum
                Method used to denoise image. Options: DenoiserEnum.GAUSSIAN,
                DenoiserEnum.BILATERAL, DenoiserEnum.TOTAL_VARIATION, DenoiserEnum.WAVELET,
                DenoiserEnum.NON_LOCAL_MEANS, DenoiserEnum.BM3D.
        """
        super().__init__()
        self.clahe_object = cv2.createCLAHE(
            clipLimit=clip_limit, tileGridSize=tile_grid_size
        )

    def forward(self, tensor: Tensor, **kwargs) -> Tensor:
        """
        Denoises the ``image`` according to the ``denoiser_method`` attribute.

        Parameters
        ----------
        image : array
                Noise image that will be denoised.

        Each method parameter is described in the method Docstring.
        """
        equalized_image = tensor.numpy()
        original_shape = equalized_image.shape

        equalized_image = self._float_to_int_image(equalized_image)
        if len(original_shape) == 3:
            equalized_image = np.transpose(equalized_image, (1, 2, 0))
            equalized_image = self._make_single_channel(equalized_image)

        equalized_image = self.clahe_object.apply(src=equalized_image)

        equalized_image = self._int_to_float_image(equalized_image)
        if len(original_shape) == 3:
            equalized_image = self._make_triple_channel(equalized_image)
            equalized_image = np.transpose(equalized_image, (2, 0, 1))

        return torch.from_numpy(equalized_image)

    def _make_single_channel(self, image: np.ndarray):
        return image[:, :, 0]

    def _make_triple_channel(self, image: np.ndarray):
        return np.repeat(image[:, :, np.newaxis], 3, axis=2)

    def _float_to_int_image(self, image: np.ndarray):
        return (image * INT_PIXEL_CONVERSION).astype(np.uint8)

    def _int_to_float_image(self, image: np.ndarray):
        return (image.astype(np.float32)) / INT_PIXEL_CONVERSION


class SpectralColor(torch.nn.Module):
    """
    A PyTorch module that applies a spectral colormap to a grayscale image.

    """

    def __init__(self, color_map: str) -> None:
        """

        Initialize the SpectralColor class.

        Args:
            color_map (str): The name of the colormap to use.

        Returns:
            final_image (PIL.Image): The resulting colorized image.
        """
        super().__init__()
        self.cmap = cm.get_cmap(
            color_map
        )  # get_cmap: function from matplotlib.cm is deprecated
        self.color_map = color_map

        self.pil_transform = T.ToPILImage()
        self.tensor_transform = T.ToImage()

    def forward(self, image: Tensor, **kwargs) -> Tensor:
        """
        Applies the spectral colormap to the input tensor image.

        Args:
            image (torch.Tensor): The input grayscale tensor image.
            Shape should be (1, height, width) for grayscale images.

        Returns:
            color_tensor (torch.Tensor): The resulting colorized tensor image.
            Shape will be (3, height, width).
        """
        image_pil = self.pil_transform(image)

        # Convert image to grayscale
        grayscale_image = image_pil.convert("L")

        # Convert grayscale image to numpy array
        np_img = np.array(grayscale_image)

        # Apply spectral colormap
        colored_img = (self.cmap(np_img) * 255).astype(np.uint8)

        # Convert back to PIL Image and remove alpha channel
        final_image = Image.fromarray(colored_img[:, :, :3])

        return self.tensor_transform(final_image)

    def __repr__(self) -> str:
        """
        is a special method that returns a string representation of an object.

        Returns:
          a string representation of the object.
        """
        return f"{self.__class__.__name__}(Spectral color={self.color_map})"


class MorphologicalTranformation(ABC, torch.nn.Module):
    """Module that applies morphological opening (erosion followed by dilation) to an image."""

    def __init__(
        self,
        kernel_size: tuple[int, int],
    ) -> None:
        """Initialize the MorphologicalTranformation class.

        Args:
            slic_algorithm (int): The algorithm to use for SLIC.
                Options: cv2.ximgproc.SLIC, cv2.ximgproc.SLICO, cv2.ximgproc.MSLIC.
                Default is cv2.ximgproc.SLIC.
            kernel_size (tuple[int, int]): Size of the kernel.
        """
        super().__init__()
        self.kernel_size: tuple[int, int] = kernel_size

        self.pil_transform = T.ToPILImage()
        self.tensor_transform = T.ToImage()
        self.morth_operation = self._build_morth_operation()

    def forward(
        self, image: Tensor
    ) -> Union[Tensor, list[cv2.ximgproc_SuperpixelSLIC]]:
        """Applies morphological transformation to the input tensor image.

        The method is individually applied to each channel of the input image.

        Args:
            image (torch.Tensor): The input tensor image.
                Shape should be (C, height, width).

        Returns:
            Tensor: The resulting tensor image mask.
        """
        # Convert tensor to numpy array for each channel
        np_image: np.ndarray = np.array(
            [
                np.array(self.pil_transform(image[i].unsqueeze(0).int()).convert("L"))
                for i in range(image.shape[0])
            ]
        )
        # Apply morphological transformation to each channel
        out = [
            self._apply_morphological_operation(np_image[i, ..., np.newaxis])
            for i in range(image.shape[0])
        ]
        out = np.transpose(np.array(out), (1, 2, 0))
        out = self.tensor_transform(out)

        return out

    def _apply_morphological_operation(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Applies SLIC to the input numpy array image."""
        kernel = np.ones(self.kernel_size, np.uint8)

        out = cv2.morphologyEx(image, self.morth_operation, kernel)

        return out

    @abstractmethod
    def _build_morth_operation(self):
        raise NotImplementedError


class Opening(MorphologicalTranformation):
    """Module that applies morphological opening (erosion followed by dilation) to an image."""

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super().__init__(kernel_size)

    def _build_morth_operation(self):
        return cv2.MORPH_OPEN


class Closing(MorphologicalTranformation):
    """Module that applies morphological clossing (dilation followed by erosion) to an image."""

    def __init__(self, kernel_size: Tuple[int, int]) -> None:
        super().__init__(kernel_size)

    def _build_morth_operation(self):
        return cv2.MORPH_CLOSE
