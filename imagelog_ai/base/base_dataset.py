"""This module contains the BaseDataset class.

The BaseDataset class is the base class for custom datasets in the imagelog project.
"""

from abc import ABC, abstractmethod
import os
from typing import Any, Callable, Final, Optional

from torch.utils.data import Dataset

from imagelog_ai.utils.os_functions import get_files_paths
from imagelog_ai.utils.io_functions import tensor_load


class BaseDataset(ABC, Dataset):
    """Base class for custom datasets.

    Args:
        project_name (str): The name of the project.
        preprocess_name (str): The name of the preprocessing.
        list_datasource_names (list[str]): A list of datasource names.
        class_list (list[str]): A list of class names.
        others_class_list (Optional[list[str]], optional): A list of other class names.
            Defaults to None.
        transform (Optional[Callable], optional): A function that applies transformations to
        the data.
            Defaults to None.
    """

    def __init__(
        self,
        project_name: str,
        preprocess_name: str,
        list_datasource_names: list[str],
        class_list: list[str],
        others_class_list: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__()
        self.project_name: str = project_name
        self.preprocess_name: str = preprocess_name
        self.list_datasource_names: list[str] = list_datasource_names
        self.class_list: list[str] = sorted(class_list)
        self.others_class_list: Optional[list[str]] = others_class_list
        self.transform: Optional[Callable] = transform

        self.project_path: Final = os.path.join(
            "data/processed", self.project_name, self.preprocess_name
        )

        self._build_paths()

    def _build_paths(self) -> None:
        """Creates lists of image file paths and labels that will be processed."""
        list_src_files_paths: list[str] = []
        for datasource_name in self.list_datasource_names:
            src_path = os.path.join(self.project_path, datasource_name)

            list_src_files_paths.extend(get_files_paths(src_path))

        self.list_images_paths = list(
            filter(lambda x: "labels" not in x, list_src_files_paths)
        )
        # If len(self.list_images_paths) == 0, raise an error (no images found in the dataset)
        if len(self.list_images_paths) == 0:
            raise ValueError(
                f"No images found in the dataset. Check the path: {src_path}."
            )

        self.list_labels_paths = list(
            filter(lambda x: "labels" in x, list_src_files_paths)
        )

    @abstractmethod
    def _load_label(self, label_path: str, *args, **kwargs) -> Any:
        """Abstract method to load the label from a given path.

        Args:
            label_path (str): The path to the label file.

        Returns:
            Any: The loaded label.
        """
        raise NotImplementedError

    @abstractmethod
    def dataset_info(self):
        """Abstract method to provide information about the dataset."""
        raise NotImplementedError

    def __len__(self) -> int:
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.list_images_paths)

    def __getitem__(self, idx: int):
        """Returns the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple or Any: The image and label (if available) at the given index.
        """
        img = tensor_load(self.list_images_paths[idx])

        if self.transform:
            img = self.transform(img)

        if self.list_labels_paths:
            labels = self._load_label(self.list_labels_paths[idx])

            return img, labels
        else:
            return img
