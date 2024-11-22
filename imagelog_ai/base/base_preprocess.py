"""This module contains the base class for preprocessing image data and labels in the project."""

from abc import ABC, abstractmethod
import os
import shutil
from typing import List, Tuple, Optional, Callable, Final, Any, Union

from torch import Tensor
from tqdm import tqdm

from imagelog_ai.utils.argcheck import dir_exists, same_extension
from imagelog_ai.utils.io_functions import json_save, pil_save, pil_load, tensor_save
from imagelog_ai.utils.os_functions import get_files_paths, copytree


def check_same_file_names_without_extension(
    list_of_paths1: List[str], list_of_paths2: List[str]
) -> Any:
    """Checks whether two lists of files have the same names, disregarding extensions.

        Methods to be implemented when inheriting:
                `_process_label_data()`: Method for calculating the loss of results.

    Parameters
    ----------
    list_of_paths1 : List[str]
        First list of file paths.

    list_of_paths2 : List[str]
        Second list of file paths.
    """

    names = [
        [path.split("/")[-1].split(".")[0] for path in list_of_paths]
        for list_of_paths in (list_of_paths1, list_of_paths2)
    ]

    if names[0] != names[1]:
        raise TypeError("The two lists do not have the same files.")


class BaseImagesProcessor(ABC):
    """Base abstract class for preprocessing image data and labels for
        clasification, object detection and segmentation tascks.

            Functions to be implemented when inheriting:
             `_process_label_data()`: Method that processes labels. For example,
                                      loading json containing boundboxes, or loading
                                      masks used for segmentation.

    Parameters
    ----------
    project_name: str
            Project name located in the `projects` directory,
            where the preprocessing setup json file is located.

    image_format: Optional[str]
            Sets the image format to be saved to `processed`.
            If it is None (Default), then it will be saved as Tensor.

    input_pil_mode: Optional[str] = "RGB"
            Select image reading mode via PIL. Default is RGB.

    override: Optional[str] = bool = False
            Option to override preprocessing. Default is False.

    """

    def __init__(
        self,
        project_name: str,
        preprocess_name: str,
        image_format: Optional[str] = None,
        input_pil_mode: Optional[str] = "RGB",
        override: bool = False,
    ):

        self.project_name: Final = project_name
        self.preprocess_name: Final = preprocess_name
        self.image_format: Final = image_format
        self.input_pil_mode: Final = input_pil_mode
        self.override: Final = override

        self.project_path: Final = os.path.join(
            "data/processed", self.project_name, self.preprocess_name
        )
        self.save_image_fn = pil_save if self.image_format else tensor_save
        self.dst_file_extension = (
            f".{self.image_format}" if self.image_format else ".pt"
        )

        self._override()

    def _override(self):
        """Method to override preprocessing."""
        if dir_exists(dir_path=self.project_path, override=self.override):
            shutil.rmtree(self.project_path, ignore_errors=True)

    def _build_paths_and_trees(
        self, list_datasource_names: List[str], dirs_exist_ok: bool = False
    ) -> Tuple[List, List]:
        """Method to build the directory tree in the 'processed/PROJECT_NAME/' directory
            with the same file structure as the source directory in 'interim'.
            And, it returns lists of image file paths and labels that will be processed.

        Parameters
        ----------
        list_datasource_names : List[str]
            List of data sources that will be processed, and are found in 'interim/'
        """
        list_src_files_paths = []
        for datasource_name in list_datasource_names:
            src_path = os.path.join("data/interim", datasource_name)
            copytree(
                src_path=src_path,
                dst_path=self.project_path,
                dirs_exist_ok=dirs_exist_ok,
            )
            list_src_files_paths.extend(get_files_paths(src_path))

        list_images_paths = list(
            filter(lambda x: "labels" not in x, list_src_files_paths)
        )
        list_labels_paths = list(filter(lambda x: "labels" in x, list_src_files_paths))

        return list_images_paths, list_labels_paths

    def _check_consistency(
        self, list_images_paths: List[str], list_labels_paths: List[str]
    ) -> None:
        """Method checks whether all image files and labels have, respectively,
            the same formats. And check if there is a label for each image
            (they must have the same name).

        Parameters
        ----------
        list_images_paths : List[str]
            List of images that will be processed, and are found in 'interim/'

        list_labels_paths : List[str]
            List of labels that will be processed, and are found in 'interim/'
        """
        if list_images_paths:
            same_extension(list_images_paths)

        if list_labels_paths:
            same_extension(list_labels_paths)
            check_same_file_names_without_extension(
                list_images_paths, list_labels_paths
            )

    def _built_dst_file_path(self, src_file_path: str) -> str:
        """Method to set the path where the data will be saved in 'processed/PROJECT_NAME'.

        Parameters
        ----------
        src_file_path : List[str]
            Source path of the data that will be processed
        """
        return src_file_path.replace("data/interim", self.project_path).split(".")[0]

    def _process_image_data(
        self, src_file_path: str, input_transform: Optional[Union[Callable, str]] = None
    ) -> Tensor:
        """Method that loads image data from its source path, processes it
            with 'input_transform', and saves it within the directory
            structure created in 'processed/'.

        Parameters
        ----------
        src_file_path : List[str]
            Source path of the data that will be processed

        input_transform : Optional[Union[Callable, str]] = None
            Transformation that will be applied to each image.

        Returns
        -------
        Tensor : processed image.
        """
        self.current_tensor_image = pil_load(
            src_file_path, pil_mode=self.input_pil_mode
        )

        if input_transform == "hard_copy":
            shutil.copyfile(
                src=src_file_path,
                dst=src_file_path.replace("data/interim", self.project_path),
            )
            tensor_image = self.current_tensor_image
        else:
            tensor_image = (
                input_transform(self.current_tensor_image)
                if callable(input_transform)
                else (
                    self.current_tensor_image
                    if input_transform
                    else self.current_tensor_image
                )
            )

            dst_file_path = self._built_dst_file_path(src_file_path)
            self.save_image_fn(tensor_image, dst_file_path + self.dst_file_extension)
        return tensor_image

    def _process_validation(
        self,
        current_tensor_image: Tensor,
        tensor_image: Tensor,
        labels: Any,
        input_transform_validation: Optional[list[Callable]] = None,
    ) -> None:
        """Method that will execute each validation function and store the results.

        Parameters
        ----------
        current_tensor_image : Tensor
            Current tensor image.

        tensor_image : Tensor
            Processed image.

        labels : Any
            Labels for the image.

        input_transform_validation : Optional[Callable] = None
            Transformation that will be applied to each image.

        Returns
        -------
        None
        """
        # Store validation results in a dictionary
        validation_results = {}
        # Execute each validation function
        if input_transform_validation:
            for validation_fn in input_transform_validation:
                validation_results[validation_fn.__name__] = validation_fn(
                    current_tensor_image, tensor_image, labels
                )
            # Create directory tree if it does not exist
            os.makedirs(os.path.join("logs", "preprocessing_validation"), exist_ok=True)
            # Store the results in a json file
            json_save(
                validation_results,
                os.path.join(
                    "logs", "preprocessing_validation", "validation_results.json"
                ),
            )

    @abstractmethod
    def _process_label_data(
        self, src_file_path: str, **kwargs_process_label: Any
    ) -> None:
        """Abstract method that loads label data from its source path,
            processes it, and saves it within the directory
            structure created in 'processed/'.

        Parameters
        ----------
        src_file_path : List[str]
            Source path of the data that will be processed

        kwargs_process_label : Any
            Attributes necessary for the method to perform transformations on label data.
        """
        raise NotImplementedError

    def __call__(
        self,
        list_datasource_names: List[str],
        input_transform: Optional[Union[Callable, str]] = None,
        input_transform_validation: Optional[list[Callable]] = None,
        **kwargs_process_label: Any,
    ) -> None:
        """Method that will loop through each image and label data,
            loading them and applying the transformations.

        Parameters
        ----------
        list_datasource_names : List[str]
            List of data sources that will be processed, and are found in 'interim/'

        input_transform : Optional[Union[Callable, str]] = None
            Transformation that will be applied to each image.

        kwargs_process_label : Any
            Attributes necessary for the method to perform transformations on label data.
        """
        list_images_paths, list_labels_paths = self._build_paths_and_trees(
            list_datasource_names
        )

        self._check_consistency(list_images_paths, list_labels_paths)

        for file_idx, src_file_path in tqdm(
            enumerate(list_images_paths),
            desc="Preprocessing",
            total=len(list_images_paths),
        ):
            self._process_idx(
                input_transform,
                input_transform_validation,
                kwargs_process_label,
                list_labels_paths,
                file_idx,
                src_file_path,
            )

    def _process_idx(
        self,
        input_transform,
        input_transform_validation,
        kwargs_process_label,
        list_labels_paths,
        file_idx,
        src_file_path,
    ):
        """Method that will process each image and label data.

        Parameters
        ----------
        input_transform : Optional[Union[Callable, str]] = None
            Transformation that will be applied to each image.

        input_transform_validation : Optional[list[Callable]] = None
            Transformation that will be applied to each image.

        kwargs_process_label : Any
            Attributes necessary for the method to perform transformations on label data.

        list_labels_paths : List[str]
            List of labels that will be processed, and are found in 'interim/'

        file_idx : int
            Index of the file being processed.

        src_file_path : str
            Source path of the data that will be processed
        """
        tensor_image = self._process_image_data(src_file_path, input_transform)

        if list_labels_paths:
            labels = self._process_label_data(
                list_labels_paths[file_idx], **kwargs_process_label
            )
        self._process_validation(
            self.current_tensor_image,
            tensor_image,
            labels,
            input_transform_validation,
        )
