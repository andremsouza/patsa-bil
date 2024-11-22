from imagelog_ai.base.base_dataset import BaseDataset
from imagelog_ai.utils.io_functions import json_load
from imagelog_ai.utils.encoders import LabelEncoder, OneHotEncoder
import torch
from typing import Optional, Callable


class ClassificationDataset(BaseDataset):
    """Class for creating a dataset for classification tasks.

    Parameters
    ----------
    project_name: str
            Project name located in the `projects` directory,
            where the preprocessing setup json file is located.

    preprocess_name: str
            Preprocess name located in the `preprocesses` directory,
            where the preprocessing setup json file is located.

    list_datasource_names: List[str]
            List of datasource names located in the `datasources` directory,
            where the datasource setup json file is located.

    class_list: List[str]
            List of classes that will be used to perform label encoding.

    task: str
            Task to perform, either `multiclass` or `multilabel`.

    others_class_list: Optional[List[str]] = None
            List of classes that will be used to perform label encoding
            for the `others` class.

    transform: Optional[Callable] = None
            Transform function to apply to the data.
    """

    def __init__(
        self,
        project_name: str,
        preprocess_name: str,
        list_datasource_names: list[str],
        class_list: list[str],
        task: str,
        others_class_list: Optional[list[str]] = None,
        transform: Optional[Callable] = None,
    ) -> None:
        super().__init__(
            project_name,
            preprocess_name,
            list_datasource_names,
            class_list,
            others_class_list,
            transform,
        )

        self.task = task

        self._build_encoder()

        if self.list_labels_paths:
            self._filter_by_labels()

    def _build_encoder(self):
        kwargs_encoder = {
            "class_list": self.class_list,
            "others_class_list": self.others_class_list,
        }
        self.encoder = (
            OneHotEncoder(**kwargs_encoder)
            if self.task == "multilabel"
            else LabelEncoder(**kwargs_encoder)
        )

    def _filter_by_labels(self) -> None:
        set_of_allowed_classes = set(self.class_list)
        if self.others_class_list:
            set_of_allowed_classes = set_of_allowed_classes.union(
                self.others_class_list
            )

        def condition(labels):
            if self.task == "multilabel":
                return set(labels).issubset(set_of_allowed_classes)
            else:
                return set(labels).issubset(set_of_allowed_classes) and len(labels) == 1

        self.original_dataset_size = len(self.list_labels_paths)
        new_list_images_paths, new_list_labels_paths = [], []
        for img_path, label_path in zip(self.list_images_paths, self.list_labels_paths):
            labels = json_load(label_path)["labels"]

            if condition(labels):
                new_list_images_paths.append(img_path)
                new_list_labels_paths.append(label_path)

        self.list_images_paths = new_list_images_paths
        self.list_labels_paths = new_list_labels_paths

    def dataset_info(self):
        dataset_info = dict.fromkeys(self.class_list, 0)
        if self.others_class_list:
            dataset_info["others"] = 0

        for label_path in self.list_labels_paths:
            labels = json_load(label_path)["labels"]
            for label in labels:
                if dataset_info.get(label) is not None:
                    dataset_info[label] += 1
                else:
                    dataset_info["others"] += 1

        dataset_info["total_instances"] = sum(dataset_info.values())
        dataset_info["total_samples"] = len(self.list_labels_paths)
        if self.list_labels_paths:
            dataset_info["original_dataset_size"] = self.original_dataset_size
        return dataset_info

    def _load_label(self, label_path: str) -> torch.Tensor:
        """Load label from json file.

        Parameters
        ----------
        label_path: str
                Path to the json file containing the labels.

        Returns
        -------
        torch.Tensor
                Tensor containing the encoded label.
        """
        labels = json_load(label_path)["labels"]

        if self.task == "multilabel":
            return self.encoder(labels)
        else:
            return self.encoder(labels[0])
