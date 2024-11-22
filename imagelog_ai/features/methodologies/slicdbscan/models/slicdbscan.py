from concurrent.futures import ProcessPoolExecutor
import os
from typing import Callable, Optional, Union
import warnings

import cv2
import numpy as np
import pandas as pd
import radiomics
import SimpleITK as sitk
from skimage import measure
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
)
import torch
from torch import Tensor
import torchvision.transforms.v2 as T

from imagelog_ai.utils.pca import broken_stick


class LabelSizeFilter(torch.nn.Module):
    """A PyTorch module that filters labels based on their size.

    Args:
        min_label_size (Optional[int]): The minimum size of labels to keep. If a label has a size smaller than
            this value, it will be replaced with the background label. Default is None.
        max_label_size (Optional[int]): The maximum size of labels to keep. If a label has a size larger than
            this value, it will be replaced with the background label. Default is None.
        background_label (Union[int, float]): The value to replace filtered labels with. Default is 0.

    Raises:
        ValueError: If both min_label_size and max_label_size are None.
        ValueError: If min_label_size is greater than max_label_size.

    Returns:
        Tensor: The input image with labels filtered based on their size.
    """

    def __init__(
        self,
        min_label_size: Optional[int],
        max_label_size: Optional[int],
        background_label: Union[int, float] = 0,
    ) -> None:
        """Initializes a new instance of the Transform class.

        Args:
            min_label_size (Optional[int]): The minimum size of the label.
            max_label_size (Optional[int]): The maximum size of the label.
            background_label (Union[int, float], optional): The value to use for the background label. Defaults to 0.
        """
        super().__init__()
        self.min_label_size: Optional[int] = min_label_size
        self.max_label_size: Optional[int] = max_label_size
        self.background_label: Union[int, float] = background_label
        self._validate_args()

    def forward(self, image: Tensor) -> Tensor:
        """Applies forward transformation on the input image.

        Args:
            image (Tensor): The input image tensor.

        Returns:
            Tensor: The transformed image tensor.
        """
        unique_labels, label_counts = torch.unique(image, return_counts=True)
        for label, count in zip(unique_labels, label_counts):
            if (
                self.min_label_size
                and count < self.min_label_size
                and label != self.background_label
            ):
                image[image == label] = self.background_label
            if (
                self.max_label_size
                and count > self.max_label_size
                and label != self.background_label
            ):
                image[image == label] = self.background_label
        return image

    def _validate_args(self) -> None:
        """Validate the arguments passed to the class."""
        # At least one of min_label_size and max_label_size must be provided
        if self.min_label_size is None and self.max_label_size is None:
            raise ValueError(
                "At least one of min_label_size and max_label_size must be provided. "
                f"Received min_label_size={self.min_label_size} and "
                f"max_label_size={self.max_label_size}."
            )
        # min_label_size <= max_label_size
        if self.min_label_size is not None and self.max_label_size is not None:
            if self.min_label_size > self.max_label_size:
                raise ValueError(
                    "min_label_size must be less than or equal to max_label_size. "
                    f"Received min_label_size={self.min_label_size} and "
                    f"max_label_size={self.max_label_size}."
                )


class LabelIntensityFilter(torch.nn.Module):
    """A PyTorch Module that filters labels based on their intensity in the original image."""

    reduce_functions: dict[str, Callable] = {
        "mean": torch.mean,
        "median": torch.median,
        "max": torch.max,
        "min": torch.min,
        "sum": torch.sum,
    }

    def __init__(
        self,
        min_intensity: Optional[Union[int, float]],
        max_intensity: Optional[Union[int, float]],
        reduce: Optional[str] = "mean",
        background_label: Union[int, float] = 0,
    ) -> None:
        super().__init__()
        self.min_intensity: Optional[Union[int, float]] = min_intensity
        self.max_intensity: Optional[Union[int, float]] = max_intensity
        self.reduce: Optional[str] = reduce
        self.background_label: Union[int, float] = background_label
        self._validate_args()

    def forward(self, label_image: Tensor, intensity_image: Tensor) -> Tensor:
        """Applies forward transformation on the input image.

        Args:
            label_image (Tensor): The input label image tensor.
            intensity_image (Tensor): The input intensity image tensor.

        Returns:
            Tensor: The transformed label image tensor.
        """
        unique_labels = torch.unique(label_image)
        for label in unique_labels:
            # Get intensities in label_mask region
            label_mask = label_image == label
            intensities = intensity_image[label_mask]
            # Reduce intensities
            reduced_intensity = self.reduce_functions[self.reduce](intensities)
            # Filter labels based on intensity
            if (
                self.min_intensity
                and reduced_intensity < self.min_intensity
                and label != self.background_label
            ):
                label_image[label_image == label] = self.background_label
            if (
                self.max_intensity
                and reduced_intensity > self.max_intensity
                and label != self.background_label
            ):
                label_image[label_image == label] = self.background_label
        return label_image

    def _validate_args(self) -> None:
        """Validate the arguments passed to the class."""
        # At least one of min_intensity and max_intensity must be provided
        if self.min_intensity is None and self.max_intensity is None:
            raise ValueError(
                "At least one of min_intensity and max_intensity must be provided. "
                f"Received min_intensity={self.min_intensity} and "
                f"max_intensity={self.max_intensity}."
            )
        # min_intensity <= max_intensity
        if self.min_intensity is not None and self.max_intensity is not None:
            if self.min_intensity > self.max_intensity:
                raise ValueError(
                    "min_intensity must be less than or equal to max_intensity. "
                    f"Received min_intensity={self.min_intensity} and "
                    f"max_intensity={self.max_intensity}."
                )
        # Reduce function must be valid
        if self.reduce not in self.reduce_functions:
            raise ValueError(
                f"Invalid reduce function: {self.reduce}. "
                f"Valid options are {list(self.reduce_functions.keys())}."
            )


class LabelGapAdjustment(torch.nn.Module):
    """A PyTorch module that adjusts the gap between labels.

    Given a Tensor, if there is a missing label from 0 to max, decrement all labels greater than
    the missing label iteratively, until len(unique_labels) == labels.max().

    This module presumes that the labels are consecutive integers starting from 1.
    """

    def __init__(self, shift: bool = False) -> None:
        super().__init__()
        self.shift = shift

    def forward(self, image: Tensor) -> Tensor:
        """Applies forward transformation on the input image.

        Args:
            image (Tensor): The input image tensor.

        Returns:
            Tensor: The transformed image tensor.
        """
        # If image.min() <= 0, increment all labels by so that image.min() == 1
        image_shift = 1 - image.min()
        if image.min() <= 0:
            image += image_shift
        unique_labels = torch.unique(image, sorted=True)
        image_max = image.max()
        while len(unique_labels) != image_max:
            # Find next missing label
            missing_label = torch.arange(1, image_max + 1)[
                ~torch.isin(torch.arange(1, image_max + 1), unique_labels)
            ][0]
            # Decrement all labels greater than missing_label
            image[image > missing_label] -= 1
            unique_labels = torch.unique(image, sorted=True)
            image_max = image.max()
        if not self.shift:
            # Return labels to original values without shifting
            image -= image_shift
        return image


class DBSCANSegmentation(torch.nn.Module):
    """A PyTorch module that applies DBSCAN for image segmentation."""

    def __init__(
        self,
        eps: float = 0.5,
        min_samples: int = 5,
        metric: str = "euclidean",
        metric_params: Optional[dict] = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: Optional[float] = None,
        n_jobs: Optional[int] = None,
        feature_extractor: str = "regionprops",
        feature_extractor_kwargs: Optional[dict] = None,
        feature_extractor_features: list[str] = None,
        use_pca: bool = False,
        background_label: Optional[int] = None,
    ) -> None:
        super().__init__()
        self.dbscan = DBSCAN(
            eps=eps,
            min_samples=min_samples,
            metric=metric,
            metric_params=metric_params,
            algorithm=algorithm,
            leaf_size=leaf_size,
            p=p,
            n_jobs=n_jobs,
        )
        self.n_jobs: int = os.cpu_count() if n_jobs is None else n_jobs
        self.feature_extractor: str = feature_extractor
        self.feature_extractor_kwargs: Optional[dict] = (
            feature_extractor_kwargs if feature_extractor_kwargs else {}
        )
        self.feature_extractor_features: list[str] = feature_extractor_features
        self.use_pca: bool = use_pca
        self.background_label: Optional[int] = background_label
        self.pil_transform = T.ToPILImage()
        self.tensor_transform = T.ToImage()
        self.last_feature_set: Optional[pd.DataFrame] = None
        self.last_feature_set_pca: Optional[pd.DataFrame] = None
        self.last_metrics: dict = {}

    def forward(self, image: Tensor, labels: Tensor) -> Tensor:
        """Applies forward transformation on the input image.

        Args:
            image (Tensor): The input image tensor. Shape should be (height, width).
            labels (Tensor): The input image labels tensor. Shape should be (height, width).

        Returns:
            Tensor: The transformed image tensor.
        """
        np_image: np.ndarray = image.numpy()
        np_labels: np.ndarray = labels.numpy()
        if self.background_label is not None:
            # Set background label to 0
            # Skimage will ignore 0 labels
            # In Radiomics, we need to skip 0 labels in feature extraction
            labels[labels == self.background_label] = 0
            # Adjust gaps between labels
            labels = LabelGapAdjustment()(labels)
            assert labels.min() == 0, f"forward: " f"labels.min()={labels.min()} != 0"
        superpixel_features: np.ndarray = self._summarize_superpixels(
            np_image, np_labels
        )
        dbscan_labels: np.ndarray = self._apply_dbscan(superpixel_features)
        # If labels.min() <= 0, increment labels until min is 1
        if dbscan_labels.min() <= 0:
            dbscan_labels += 1 - dbscan_labels.min()
        dbscan_labels = self._desummarize_superpixels(np_labels, dbscan_labels)
        # Calculate metrics
        self.last_metrics = self._calculate_metrics()
        return torch.from_numpy(dbscan_labels)

    def _summarize_superpixels(
        self,
        image: np.ndarray,
        labels: np.ndarray,
    ) -> np.ndarray:
        """Summarize superpixels using either skimage or radiomics.

        Args:
            image (np.ndarray): The input numpy array image.
                Shape should be (height, width).
            labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).

        Returns:
            np.ndarray: The resulting superpixel features.
        """
        if self.feature_extractor == "radiomics":
            return self._summarize_superpixels_radiomics(image, labels)
        elif self.feature_extractor == "regionprops":
            return self._summarize_superpixels_regionprops(image, labels)
        else:
            raise ValueError(
                f"Invalid feature_extractor: {self.feature_extractor}. "
                "Valid options are 'radiomics' and 'regionprops'."
            )

    def _summarize_superpixels_regionprops(
        self, image: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Summarize superpixels using skimage.measure.regionprops.

        Args:
            image (np.ndarray): The input numpy array image.
                Shape should be (height, width).
            labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).

        Returns:
            np.ndarray: The resulting superpixel features.
        """
        props = measure.regionprops(
            label_image=labels,
            intensity_image=image,
            **self.feature_extractor_kwargs,
        )
        assert len(props) == labels.max(), (
            f"_summarize_superpixels_regionprops: "
            f"len(label_measures)={len(props)} "
            f"!= labels.max()={labels.max()}"
        )

        regionprops = []
        for region in props:
            row = {}
            if self.feature_extractor_features is None:
                # By default, use only scalar features
                # Ignore diagnostics features
                ignored_features = [
                    "coords",
                    "image",
                    "image_convex",
                    "image_filled",
                    "image_intensity",
                    "label",
                ]
                for prop in region:
                    if np.isscalar(region[prop]) and prop not in ignored_features:
                        row[prop] = region[prop]
            else:
                for feature in self.feature_extractor_features:
                    # If tuple or ndarray, append each value separately
                    if isinstance(region[feature], (tuple, list)):
                        for i, val in enumerate(region[feature]):
                            row[f"{feature}_{i}"] = val
                    elif isinstance(region[feature], np.ndarray):
                        for i, val in enumerate(region[feature].flatten()):
                            row[f"{feature}_{i}"] = val
                    else:
                        row[feature] = region[feature]
            regionprops.append(row)
        regionprops = pd.DataFrame(regionprops, index=np.unique(labels))
        self.last_feature_set = regionprops
        self.last_feature_set_pca = None
        # Normalize the features
        superpixel_features = regionprops.to_numpy(dtype=np.float64, copy=True)
        superpixel_features = (
            superpixel_features - superpixel_features.mean(axis=0)
        ) / (superpixel_features.std(axis=0) + 1e-6)
        # If use_pca is True, apply PCA to the features
        if self.use_pca:
            # Estimate intrinsic dimensionality with broken_stick
            num_pca = broken_stick(superpixel_features, normalize=False)
            pca = PCA(n_components=num_pca)
            superpixel_features = pca.fit_transform(superpixel_features)
            self.last_feature_set_pca = pd.DataFrame(
                superpixel_features, index=np.unique(labels), columns=range(num_pca)
            )

        return superpixel_features

    def _summarize_superpixels_radiomics(
        self, image: np.ndarray, labels: np.ndarray
    ) -> np.ndarray:
        """Summarize superpixels using radiomics.

        Args:
            image (np.ndarray): The input numpy array image.
                Shape should be (height, width).
            labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).

        Returns:
            np.ndarray: The resulting superpixel features.
        """
        superpixel_features = []
        results = []
        sitk_image: sitk.Image = sitk.GetImageFromArray(image[..., np.newaxis])
        with ProcessPoolExecutor(
            None if self.n_jobs == -1 else self.n_jobs
        ) as executor:
            futures = []
            for label in np.unique(labels):
                if self.background_label is not None and label == 0:
                    continue
                mask = labels == label
                mask = mask.astype(int)[..., np.newaxis]
                mask = sitk.GetImageFromArray(mask)
                mask.CopyInformation(sitk_image)
                future = executor.submit(
                    self._extract_features_radiomics,
                    sitk_image,
                    mask,
                )
                futures.append(future)
            results = [future.result(timeout=60) for future in futures]
        # superpixel_features: np.ndarray = np.array(results)
        if len(results) == len(np.unique(labels)[1:]):
            self.last_feature_set = pd.DataFrame(results, index=np.unique(labels)[1:])
        elif len(results) == len(np.unique(labels)):
            self.last_feature_set = pd.DataFrame(results, index=np.unique(labels))
        else:
            raise ValueError(
                f"Number of results ({len(results)}) does not match number of labels "
                f"({len(np.unique(labels))})."
            )
        self.last_feature_set_pca = None
        # Filter features with self.feature_extractor_features and convert to array
        if self.feature_extractor_features is None:
            superpixel_features = self.last_feature_set.loc[
                :,
                [
                    col
                    for col in self.last_feature_set.columns
                    if "diagnostics_" not in col
                ],
            ]
        else:
            superpixel_features = self.last_feature_set.loc[
                :, self.feature_extractor_features
            ]
        # Convert to NumPy array and normalize the features
        superpixel_features = superpixel_features.to_numpy(dtype=np.float64, copy=True)
        superpixel_features = (
            superpixel_features - superpixel_features.mean(axis=0)
        ) / (superpixel_features.std(axis=0) + 1e-6)
        # If use_pca is True, apply PCA to reduce the number of features
        if self.use_pca:
            # Estimate intrinsic dimensionality with broken_stick
            num_pca = broken_stick(superpixel_features, normalize=False)
            pca = PCA(n_components=min(len(superpixel_features), num_pca))
            superpixel_features = pca.fit_transform(superpixel_features)
            self.last_feature_set_pca = pd.DataFrame(
                superpixel_features, index=np.unique(labels)[1:], columns=range(num_pca)
            )

        return superpixel_features

    def _extract_features_radiomics(
        self,
        image: sitk.Image,
        mask: sitk.Image,
    ) -> np.ndarray:
        """Extract features using radiomics.

        Args:
            image (sitk.Image): The input SimpleITK image.
            mask (sitk.Image): The input SimpleITK mask.

        Returns:
            np.ndarray: The resulting features.
        """
        feature_extractor = radiomics.featureextractor.RadiomicsFeatureExtractor(
            normalize=True, force2D=True, force2Ddimension=2, minimumROIDimensions=1
        )
        feature_extractor.enableAllFeatures()
        result = feature_extractor.execute(image, mask)
        return result

    def _desummarize_superpixels(
        self, labels: np.ndarray, dbscan_labels: np.ndarray
    ) -> np.ndarray:
        """Desummarize superpixels.

        Args:
            labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).
            dbscan_labels (np.ndarray): The input numpy array image labels.
                Shape should be (height, width).

        Returns:
            np.ndarray: The resulting desummarized numpy array image labels.
        """
        assert len(dbscan_labels) == labels.max(), (
            f"_desummarize_superpixels: "
            f"len(dbscan_labels)={len(dbscan_labels)} "
            f"!= labels.max()={labels.max()}"
        )
        desummarized_mask = np.zeros_like(labels)
        for i, dbscan_label in enumerate(dbscan_labels):
            mask = labels == i + 1
            assert mask.sum() > 0, (
                f"_desummarize_superpixels: " f"mask.sum()={mask.sum()} == 0"
            )
            desummarized_mask[mask] = dbscan_label
            # Update self.last_feature_set
            # Each row in self.last_feature_set corresponds to a label in labels
            # New column will have the corresponding dbscan_label
            if self.last_feature_set is not None:
                self.last_feature_set.loc[i + 1, "dbscan_label"] = dbscan_label
            if self.last_feature_set_pca is not None:
                self.last_feature_set_pca.loc[i + 1, "dbscan_label"] = dbscan_label
        return desummarized_mask

    def _apply_dbscan(self, features: np.ndarray) -> np.ndarray:
        """Applies DBSCAN to the input numpy array features.

        Args:
            features (np.ndarray): The input numpy array features.

        Returns:
            np.ndarray: The resulting numpy array labels.
        """
        self.dbscan.fit(features)
        labels = self.dbscan.labels_

        return labels

    def _calculate_metrics(self) -> dict:
        """Calculate clustering metrics for the DBSCAN segmentation.

        Returns:
            dict: The resulting metrics.
        """
        metrics = {}
        if self.last_feature_set is not None:
            feature_array: np.ndarray = self.last_feature_set.drop(
                columns=[
                    col
                    for col in self.last_feature_set.columns
                    if "diagnostics_" in col or "dbscan_label" in col
                ]
            ).to_numpy()
            labels: np.ndarray = self.last_feature_set["dbscan_label"].to_numpy()
            try:
                metrics["silhouette_score"] = silhouette_score(
                    X=feature_array, labels=labels
                )
            except ValueError as exc:
                warnings.warn(
                    f"Error calculating silhouette_score: {exc}. "
                    "Skipping silhouette_score calculation."
                )
                metrics["silhouette_score"] = None
            try:
                metrics["calinski_harabasz_score"] = calinski_harabasz_score(
                    X=feature_array, labels=labels
                )
            except ValueError as exc:
                warnings.warn(
                    f"Error calculating calinski_harabasz_score: {exc}. "
                    "Skipping calinski_harabasz_score calculation."
                )
                metrics["calinski_harabasz_score"] = None
            try:
                metrics["davies_bouldin_score"] = davies_bouldin_score(
                    X=feature_array, labels=labels
                )
            except ValueError as exc:
                warnings.warn(
                    f"Error calculating davies_bouldin_score: {exc}. "
                    "Skipping davies_bouldin_score calculation."
                )
                metrics["davies_bouldin_score"] = None
        if self.last_feature_set_pca is not None:
            feature_array: np.ndarray = self.last_feature_set_pca.drop(
                columns=["dbscan_label"]
            ).to_numpy()
            labels: np.ndarray = self.last_feature_set_pca["dbscan_label"].to_numpy()
            try:
                metrics["silhouette_score_pca"] = silhouette_score(
                    X=feature_array, labels=labels
                )
            except ValueError as exc:
                warnings.warn(
                    f"Error calculating silhouette_score_pca: {exc}. "
                    "Skipping silhouette_score_pca calculation."
                )
                metrics["silhouette_score_pca"] = None
            try:
                metrics["calinski_harabasz_score_pca"] = calinski_harabasz_score(
                    X=feature_array, labels=labels
                )
            except ValueError as exc:
                warnings.warn(
                    f"Error calculating calinski_harabasz_score_pca: {exc}. "
                    "Skipping calinski_harabasz_score_pca calculation."
                )
                metrics["calinski_harabasz_score_pca"] = None
            try:
                metrics["davies_bouldin_score_pca"] = davies_bouldin_score(
                    X=feature_array, labels=labels
                )
            except ValueError as exc:
                warnings.warn(
                    f"Error calculating davies_bouldin_score_pca: {exc}. "
                    "Skipping davies_bouldin_score_pca calculation."
                )
                metrics["davies_bouldin_score_pca"] = None

        return metrics


class SLICSegmentation(torch.nn.Module):
    """A PyTorch module that applies SLIC to an image."""

    def __init__(
        self,
        slic_algorithm: int = cv2.ximgproc.SLIC,
        region_size: int = 10,
        ruler: float = 10.0,
        slic_iterations: int = 10,
        gaussian_blur: Optional[Union[tuple[int, int], int]] = 3,
        sigma: Union[tuple[float, float], float] = 0.0,
        enforce_connectivity: bool = True,
        min_element_size: int = 25,
        return_labels: bool = True,
        labels_new_dimension: bool = False,
        return_slic_object: bool = False,
    ) -> None:
        """Initialize the SLICSegmentation class.

        Args:
            slic_algorithm (int): The algorithm to use for SLIC.
                Options: cv2.ximgproc.SLIC, cv2.ximgproc.SLICO, cv2.ximgproc.MSLIC.
                Default is cv2.ximgproc.SLIC.
            region_size (int): The average superpixel size measured in pixels.
                Ignored if slic_algorithm is cv2.ximgproc.SLICO or cv2.ximgproc.MSLIC.
                Default is 10.
            ruler (float): The enforcement of superpixel smoothness factor of superpixel.
                Default is 10.0.
            slic_iterations (int): The number of iterations.
                Default is 10.
            gaussian_blur (Optional[Union[tuple[int, int], int]]): The kernel size for Gaussian blur.
                If None, no Gaussian blur is applied.
                Default is 3.
            sigma (tuple[float, float] | float): The standard deviation for Gaussian blur.
                If gaussian_blur is None, this parameter is ignored.
                Default is 0.0.
            enforce_connectivity (bool): If True, enforce connectivity between superpixels.
                Default is True.
            min_element_size (int): The minimum element size in percents that should be
                absorbed into a bigger superpixel. Ignored if enforce_connectivity is False.
                Default is 25.
            return_labels (bool): If True, return the superpixel labels.
                Mutually exclusive with return_slic_object.
                At least one of return_labels and return_slic_object must be True.
                Default is True.
            labels_new_dimension (bool): If True, add a new dimension to the labels.
                Ignored if return_labels is False.
                Default is False.
            return_slic_object (bool): If True, return the SLIC object.
                Mutually exclusive with return_labels.
                At least one of return_labels and return_slic_object must be True.
                Default is False.
        """
        super().__init__()
        self.slic_algorithm: int = slic_algorithm
        self.region_size: int = region_size
        self.ruler: float = ruler
        self.slic_iterations: int = slic_iterations
        self.gaussian_blur: Optional[tuple[int, int]] = (
            (gaussian_blur, gaussian_blur)
            if isinstance(gaussian_blur, int)
            else gaussian_blur
        )
        self.sigma: tuple[float, float] = (
            (sigma, sigma) if isinstance(sigma, float) else sigma
        )
        self.enforce_connectivity: bool = enforce_connectivity
        self.min_element_size: int = min_element_size
        self.return_labels: bool = return_labels
        self.labels_new_dimension: bool = labels_new_dimension
        self.return_slic_object: bool = return_slic_object
        self.pil_transform = T.ToPILImage()
        self.tensor_transform = T.ToImage()
        self._validate_args()

    def forward(
        self, image: Tensor
    ) -> Union[Tensor, list[cv2.ximgproc_SuperpixelSLIC]]:
        """Applies SLIC to the input tensor image.

        The method is individually applied to each channel of the input image.

        Args:
            image (torch.Tensor): The input tensor image.
                Shape should be (C, height, width).

        Returns:
            Tensor | list[cv2.ximgproc_SuperpixelSLIC]: The resulting tensor image mask
                or the SLIC object for each channel.
        """
        # Convert tensor to numpy array for each channel
        np_image: np.ndarray = np.array(
            [
                self.pil_transform(image[i].unsqueeze(0)).convert("L")
                for i in range(image.shape[0])
            ]
        )
        # Apply SLIC to each channel
        slic = [
            self._apply_slic(np_image[i, ..., np.newaxis])
            for i in range(image.shape[0])
        ]
        if self.return_labels:
            slic_labels = np.transpose(np.array(slic), (1, 2, 0))
            slic_labels = self.tensor_transform(slic_labels)
            if self.labels_new_dimension:
                slic_labels = torch.stack((image, slic_labels))
            return slic_labels
        return slic

    def _apply_slic(
        self,
        image: np.ndarray,
    ) -> np.ndarray:
        """Applies SLIC to the input numpy array image."""
        # If gaussian_blur is not None, apply Gaussian blur to the image
        if self.gaussian_blur is not None:
            image = cv2.GaussianBlur(
                src=image,
                ksize=self.gaussian_blur,
                sigmaX=self.sigma[0],
                sigmaY=self.sigma[1],
            )
        slic_object = cv2.ximgproc.createSuperpixelSLIC(
            image, self.slic_algorithm, self.region_size, self.ruler
        )
        slic_object.iterate(self.slic_iterations)
        if self.enforce_connectivity:
            slic_object.enforceLabelConnectivity(self.min_element_size)
        # Get superpixel mask
        if self.return_labels:
            return slic_object.getLabels()
        return slic_object

    def _validate_args(self) -> None:
        """Validate the arguments passed to the class."""
        # Validate return_mask and return_segmented_image
        if not isinstance(self.return_labels, bool):
            raise TypeError(
                f"Invalid type for return_mask: {type(self.return_labels)}. "
                f"Expected bool."
            )
        if not isinstance(self.return_slic_object, bool):
            raise TypeError(
                f"Invalid type for return_segmented_image: {type(self.return_slic_object)}. "
                f"Expected bool."
            )
        if self.return_labels and self.return_slic_object:
            raise ValueError(
                "return_mask and return_segmented_image cannot both be True."
            )
        if not self.return_labels and not self.return_slic_object:
            raise ValueError(
                "At least one of return_mask and return_segmented_image must be True."
            )


class SLICDBSCANSegmentation(torch.nn.Module):
    """A PyTorch module that applies SLIC and DBSCAN to an image.

    The method is applied individually to each channel of the input image.

    The method first applies SLIC to the input image to generate superpixels. Then,
    it applies DBSCAN to the superpixels to segment the image.

    The method can return the binary mask, the labels, or the segmented image.
    """

    def __init__(
        self,
        slic_algorithm: int = cv2.ximgproc.SLIC,
        region_size: int = 10,
        ruler: float = 10.0,
        slic_iterations: int = 10,
        gaussian_blur: Optional[Union[tuple[int, int], int]] = 3,
        sigma: Union[tuple[float, float], float] = 0.0,
        enforce_connectivity: bool = True,
        min_element_size: int = 25,
        min_label_size: Optional[int] = None,
        use_radiomics: bool = False,
        dbscan_features: Optional[list[str]] = None,
        use_pca: bool = False,
        eps: float = 0.00005,
        min_samples: Optional[int] = None,
        metric: str = "euclidean",
        metric_params: Optional[dict] = None,
        algorithm: str = "auto",
        leaf_size: int = 30,
        p: Optional[float] = None,
        n_jobs: int = 1,
        return_mask: bool = True,
        labels_new_dimension: bool = False,
        return_segmented_image: bool = False,
    ) -> None:
        """Initialize the SLICDBSCANSegmentation class.

        Args:
            slic_algorithm (int): The algorithm to use for SLIC.
                Options: cv2.ximgproc.SLIC, cv2.ximgproc.SLICO, cv2.ximgproc.MSLIC.
                Default is cv2.ximgproc.SLIC.
            region_size (int): The average superpixel size measured in pixels.
                Ignored if slic_algorithm is cv2.ximgproc.SLICO or cv2.ximgproc.MSLIC.
                Default is 10.
            ruler (float): The enforcement of superpixel smoothness factor of superpixel.
                Default is 10.0.
            slic_iterations (int): The number of iterations.
                Default is 10.
            gaussian_blur (Optional[Union[tuple[int, int], int]]): The kernel size for Gaussian blur.
                If None, no Gaussian blur is applied.
                Default is 3.
            sigma (tuple[float, float] | float): The standard deviation for Gaussian blur.
                If gaussian_blur is None, this parameter is ignored.
                Default is 0.0.
            enforce_connectivity (bool): If True, enforce connectivity between superpixels.
                Default is True.
            min_element_size (int): The minimum element size in percents that should be
                absorbed into a bigger superpixel. Ignored if enforce_connectivity is False.
                Default is 25.
            min_label_size (Optional[int]): The minimum label size.
                If None, no filtering is applied.
                Default is None.
            use_radiomics (bool): If True, use radiomics to summarize superpixels.
                Default is False.
            dbscan_features (Optional[list[str]]): The features to use for DBSCAN.
                If None, all non-metadata features are used.
                Default is None.
            use_pca (bool): If True, use PCA to reduce the dimensionality of the features.
            eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
                Default is 0.00005.
            min_samples (Optional[int]): The number of samples in a neighborhood for a point to be considered as a core point.
                If None, min_samples is set to region_size.
                Default is None.
            metric (str): The metric to use when calculating distance between instances in a feature array.
                Options: "euclidean", "manhattan", "..."
                Default is "euclidean".
            metric_params (Optional[dict]): Additional keyword arguments for the metric function.
                Default is None.
            algorithm (str): The algorithm to use for DBSCAN.
                Options: "auto", "ball_tree", "kd_tree", "brute".
                Default is "auto".
            leaf_size (int): Leaf size passed to BallTree or KDTree.
                Default is 30.
            p (Optional[float]): The power of the Minkowski metric to be used to calculate distance between points.
                Default is None.
            n_jobs (int): The number of parallel jobs to run.
                Default is 1.
            return_mask (bool): If True, return the binary mask.
                Mutually exclusive with return_segmented_image.
                At least one of return_mask and return_segmented_image must be True.
                Default is True.
            labels_new_dimension (bool): If True, add a new dimension to the labels.
                Ignored if return_mask is False.
                Default is False.
            return_segmented_image (bool): If True, return the segmented image.
                Mutually exclusive with return_mask.
                At least one of return_mask and return_segmented_image must be True.
                Default is False.
        """
        super().__init__()
        self.slic_algorithm: int = slic_algorithm
        self.region_size: int = region_size
        self.ruler: float = ruler
        self.slic_iterations: int = slic_iterations
        self.gaussian_blur: Optional[tuple[int, int]] = (
            (gaussian_blur, gaussian_blur)
            if isinstance(gaussian_blur, int)
            else gaussian_blur
        )
        self.sigma: tuple[float, float] = (
            (sigma, sigma) if isinstance(sigma, float) else sigma
        )
        self.enforce_connectivity: bool = enforce_connectivity
        self.min_element_size: int = min_element_size
        self.min_label_size: Optional[int] = min_label_size
        self.use_radiomics: bool = use_radiomics
        self.dbscan_features: Optional[list[str]] = dbscan_features
        if self.dbscan_features is None:
            self.dbscan_features = None if self.use_radiomics else ["intensity_mean"]
        self.use_pca: bool = use_pca
        self.eps: float = eps
        self.min_samples: int = (
            min_samples if min_samples is not None else self.region_size
        )
        self.metric: str = metric
        self.metric_params: Optional[dict] = metric_params
        self.algorithm: str = algorithm
        self.leaf_size: int = leaf_size
        self.p: Optional[float] = p
        self.n_jobs: int = n_jobs
        self.return_mask: bool = return_mask
        self.labels_new_dimension: bool = labels_new_dimension
        self.return_segmented_image: bool = return_segmented_image
        self.pil_transform = T.ToPILImage()
        self.tensor_transform = T.ToImage()
        self.dbscan = DBSCANSegmentation(
            eps=self.eps,
            min_samples=self.min_samples,
            metric=self.metric,
            metric_params=self.metric_params,
            algorithm=self.algorithm,
            leaf_size=self.leaf_size,
            p=self.p,
            n_jobs=self.n_jobs,
            feature_extractor=(
                "regionprops" if not self.use_radiomics else "radiomics"
            ),
            feature_extractor_kwargs={},
            feature_extractor_features=self.dbscan_features,
            use_pca=self.use_pca,
            background_label=1,
        )
        self._validate_args()

    def forward(self, image: Tensor) -> Tensor:
        """Applies SLIC and DBSCAN to the input tensor image.

        The method is applied individually to each channel of the input image.

        Args:
            image (torch.Tensor): The input tensor image.
                Shape should be (C, height, width).

        Returns:
            Tensor: The resulting tensor image mask.
                Shape will be (C, height, width).
        """
        # Instantiate SLICSegmentation and forward tensor image
        segmentation = SLICSegmentation(
            slic_algorithm=self.slic_algorithm,
            region_size=self.region_size,
            ruler=self.ruler,
            slic_iterations=self.slic_iterations,
            gaussian_blur=self.gaussian_blur,
            sigma=self.sigma,
            enforce_connectivity=self.enforce_connectivity,
            min_element_size=self.min_element_size,
            return_labels=True,
            labels_new_dimension=False,
            return_slic_object=False,
        )
        labels = segmentation(image)
        filter_size_transform = LabelSizeFilter(
            min_label_size=self.min_label_size,
            max_label_size=None,
            background_label=0,
        )
        filter_intensity_transform = LabelIntensityFilter(
            min_intensity=float(image.max()),
            max_intensity=None,
            reduce="sum",
            background_label=0,
        )
        label_gap_transform = LabelGapAdjustment(shift=True)
        # Apply DBSCAN to each channel
        slic_dbscan = []
        for i in range(image.shape[0]):
            labels[i] = filter_size_transform(labels[i])
            labels[i] = filter_intensity_transform(labels[i], image[i])
            labels[i] = label_gap_transform(labels[i])
            dbscan_labels = self.dbscan(image[i], labels[i])
            slic_dbscan.append(dbscan_labels)
        slic_dbscan_labels = np.transpose(np.array(slic_dbscan), (1, 2, 0)).astype(
            np.int32
        )
        slic_dbscan_labels = self.tensor_transform(slic_dbscan_labels)
        if self.return_mask:
            if self.labels_new_dimension:
                slic_dbscan_labels = torch.stack((image, slic_dbscan_labels.float()))
            return slic_dbscan_labels
        return slic_dbscan_labels * image

    def _validate_args(self) -> None:
        """Validate the arguments passed to the class."""
        # Validate return_mask and return_segmented_image
        if not isinstance(self.return_mask, bool):
            raise TypeError(
                f"Invalid type for return_mask: {type(self.return_mask)}. "
                f"Expected bool."
            )
        if not isinstance(self.return_segmented_image, bool):
            raise TypeError(
                f"Invalid type for return_segmented_image: {type(self.return_segmented_image)}. "
                f"Expected bool."
            )
        if self.return_mask and self.return_segmented_image:
            raise ValueError(
                "return_mask and return_segmented_image cannot both be True."
            )
        if not self.return_mask and not self.return_segmented_image:
            raise ValueError(
                "At least one of return_mask and return_segmented_image must be True."
            )
