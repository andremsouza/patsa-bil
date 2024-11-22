"""Implementation of a dataset for SAM fine-tuning."""

# %% [markdown]
# ## Imports

# %%
# Change directory to the root of the project
if __name__ == "__main__":
    import os
    import sys

    import dotenv
    import matplotlib.pyplot as plt

    os.chdir(os.getcwd().split("imagelog_ai")[0])
    print(f"cwd: {os.getcwd()}")
    dotenv.load_dotenv()
    PACKAGEPATH: str = os.getenv("PACKAGEPATH", "")
    sys.path.append(PACKAGEPATH)

# %%
from typing import Callable, Optional

import numpy as np
import pandas as pd
import skimage
import torch
import torchvision
from torchvision import tv_tensors
import torchvision.transforms.v2 as T

from imagelog_ai.base.base_dataset import BaseDataset
from imagelog_ai.utils.io_functions import json_load, tensor_load
from imagelog_ai.utils.mask_functions import get_bounding_box
from imagelog_ai.utils.encoders import LabelEncoder
from imagelog_ai.features.methodologies.sam.utils.labelstudio import (
    get_masks_from_annotation,
)

# %% [markdown]
# ## Constants

# %%

# %% [markdown]
# ## Classes

# %%


class SamDataset(BaseDataset):
    """A dataset for SAM fine-tuning.

    Args:
        project_name (str): The name of the project.
        preprocess_name (str): The name of the preprocessing method.
        list_datasource_names (list[str]): A list of datasource names.
        class_list (list[str]): A list of class names.
        others_class_list (Optional[list[str]], optional): A list of other class names.
            Defaults to None.
        transform (Optional[Callable], optional): A transformation function to apply to the images.
            Defaults to None.
        target_transform (Optional[Callable], optional): A label transformation function to apply.
            Defaults to None.
        target_labels (bool, optional): Whether to include labels in the target.
            Defaults to False.
        target_masks (bool, optional): Whether to include masks in the target.
            Defaults to False.
        target_boxes (bool, optional): Whether to include bounding boxes in the target.
            Defaults to False.
        masks_location (str, optional): The location of the masks. Can be "labels" or "images".
            Defaults to "labels".
        boxes_location (str, optional): The location the bounding boxes. "labels" or "masks".
            Defaults to "labels".
    """

    def __init__(
        self,
        project_name: str,
        preprocess_name: str,
        list_datasource_names: list[str],
        class_list: list[str],
        others_class_list: Optional[list[str]] = None,
        background_class: bool = False,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        target_labels: bool = False,
        target_masks: bool = False,
        target_boxes: bool = False,
        masks_location: str = "labels",
        boxes_location: str = "labels",
    ) -> None:
        """
        Initializes a SamDataset object.

        Args:
            project_name (str): The name of the project.
            preprocess_name (str): The name of the preprocessing method.
            list_datasource_names (list[str]): A list of datasource names.
            class_list (list[str]): A list of class names.
            others_class_list (Optional[list[str]], optional): A list of other class names.
                Defaults to None.
            transform (Optional[Callable], optional): An image transformation function to apply.
                Defaults to None.
            target_transform (Optional[Callable], optional): A label transformation function.
                Defaults to None.
            target_labels (bool, optional): Whether to include labels in the target.
                Defaults to False.
            target_masks (bool, optional): Whether to include masks in the target.
                Defaults to False.
            target_boxes (bool, optional): Whether to include bounding boxes in the target.
                Defaults to False.
            masks_location (str, optional): The location of the masks. Can be "labels" or "images".
                Defaults to "labels".
            boxes_location (str, optional)0: Location of bounding boxes. "labels" or "masks".
                Defaults to "labels".
        """
        super().__init__(
            project_name,
            preprocess_name,
            list_datasource_names,
            class_list,
            others_class_list,
            transform,
        )
        self.background_class = background_class
        self.target_transform: Optional[Callable] = target_transform
        self.target_labels: bool = target_labels
        self.target_masks: bool = target_masks
        self.target_boxes: bool = target_boxes
        self.mask_location: str = masks_location
        self.boxes_location: str = boxes_location
        self._build_encoder()
        self.dataframe = self._build_dataset()
        self._filter_by_labels()

    def _validate_arguments(self) -> None:
        """Validates the arguments passed to the constructor."""
        if not self.target_boxes and not self.target_labels and not self.target_masks:
            raise ValueError("At least one target must be selected.")
        if self.target_masks and self.mask_location not in ["labels", "images"]:
            raise ValueError("The mask location must be either 'labels' or 'images'.")

    def _build_encoder(self) -> None:
        """Builds the encoder for the labels."""
        kwargs_encoder = {
            "class_list": self.class_list,
            "others_class_list": self.others_class_list,
            "background": self.background_class,
        }
        self.encoder = LabelEncoder(**kwargs_encoder)

    def _build_dataset(self) -> pd.DataFrame:
        """Builds the dataset from the given data sources.

        Each row in the dataset contains the image path, the label path, and the label index.

        Returns:
            pd.DataFrame: The dataset.
        """
        # For each file, get the mask and count connected components per label
        components = []
        for json_file in self.list_labels_paths:
            data = json_load(json_file)
            # get masks
            masks = get_masks_from_annotation(data)
            # Count connected components
            for label, mask in masks.items():
                _, num = skimage.measure.label(mask, background=0, return_num=True)
                components.append(
                    {"file": json_file, "label": label, "num_components": num}
                )
        component_nums_df = pd.DataFrame(components).sort_values(by=["file", "label"])
        # For each row in component_nums_df, create num_components rows in the dataset
        components = []
        for _, row in component_nums_df.iterrows():
            for i in range(row["num_components"]):
                components.append(
                    {
                        "image_file": row["file"]
                        .replace("labels", "images")
                        .replace("json", "png"),
                        "label_file": row["file"],
                        "label": row["label"],
                        "component": i,
                    }
                )
        components_df = pd.DataFrame(components)
        return components_df

    def _filter_by_labels(self) -> None:
        """Filters the dataset by labels."""
        self.original_dataset_size = len(self.dataframe)

        for idx, row in self.dataframe.iterrows():
            label = row["label"]
            if label not in self.class_list + self.others_class_list:
                self.dataframe.drop(idx, inplace=True)
        self.dataframe.reset_index(drop=True, inplace=True)
        self.list_images_paths = list(set(self.dataframe["image_file"].tolist()))
        self.list_labels_paths = list(set(self.dataframe["label_file"].tolist()))

    def _load_label(self, label_path: str, label: str, component_num: int) -> dict:
        """Loads the label from the given path.

        Args:
            label_path (str): The path to the label.
            label (str): The label.
            component_num (int): The component number.

        Returns:
            dict: The label data.
        """
        data = json_load(label_path)
        # get masks
        masks = get_masks_from_annotation(data)
        mask = masks[label]
        mask_labels, num = skimage.measure.label(mask, background=0, return_num=True)
        assert (
            component_num <= num
        ), f"Component number {component_num} is out of range. There are {num} components."
        mask = mask_labels == component_num
        mask_tensor = torch.Tensor(mask).to(dtype=torch.uint8).unsqueeze(0)

        target: dict[str, Optional[torch.Tensor]] = {
            "labels": None,
            "masks": None,
            "boxes": None,
        }
        if self.target_labels:
            target["labels"] = torch.Tensor([self.encoder(label)]).to(dtype=torch.int64)
        if self.target_masks:
            if self.mask_location == "labels":
                target["masks"] = mask_tensor
        if self.target_boxes:
            if self.boxes_location == "labels":
                target["boxes"] = torch.Tensor(data["boxes"])

        return target

    def dataset_info(self) -> dict:
        """Returns information about the dataset.

        Returns:
            dict: The dataset information.
        """
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
        dataset_info["original_dataset_size"] = self.original_dataset_size
        return dataset_info

    def __getitem__(self, idx: int):
        """Returns the item at the given index.

        Args:
            idx (int): The index of the item.

        Returns:
            tuple or Any: The image and label (if available) at the given index.
        """
        img = torchvision.io.read_image(self.dataframe.iloc[idx]["image_file"])
        if self.mask_location == "images":
            img = img[0, :, :, :]
        if self.transform:
            img = self.transform(img)

        if self.list_labels_paths:
            labels = self._load_label(
                self.dataframe.iloc[idx]["label_file"],
                self.dataframe.iloc[idx]["label"],
                self.dataframe.iloc[idx]["component"],
            )
            if labels["masks"] is None and self.mask_location == "images":
                labels["masks"] = tensor_load(self.list_images_paths[idx])[1, :, :, :]
            if labels["boxes"] is None and self.boxes_location == "masks":
                labels["boxes"] = torch.tensor(
                    np.stack(
                        [
                            get_bounding_box(
                                labels["masks"][i].numpy(),
                                background_value=0,
                                perturbation=5,
                            )
                            for i in range(labels["masks"].shape[0])
                        ],
                        axis=0,
                    )
                )
            labels["boxes"] = tv_tensors.BoundingBoxes(
                labels["boxes"], format="xyxy", canvas_size=(img.shape[1], img.shape[2])
            )
            labels["masks"] = tv_tensors.Mask(labels["masks"])
            if self.target_transform:
                labels = self.target_transform(labels)

            return img, labels
        else:
            return img

    def __len__(self):
        """Returns the length of the dataset.

        Returns:
            int: The length of the dataset.
        """
        return len(self.dataframe)


# %% [markdown]
# ## Main (for testing)


# %%


if __name__ == "__main__":

    def transform_func(x):
        """
        Transforms the input tensor by selecting the first three channels.

        Args:
            x (torch.Tensor): Input tensor of shape (C, H, W).

        Returns:
            torch.Tensor: Transformed tensor with the first three channels selected.
        """
        # Select channels
        x = x[:3, :, :]
        return x

    def target_transform_func(x):
        """
        Applies transformations to the target data.

        Args:
            x (dict): A dictionary containing the target data.

        Returns:
            dict: The transformed target data.
        """
        # # Select channel
        # x["masks"] = x["masks"][2, :, :]
        # x["boxes"] = x["boxes"][2]
        # # Binarize masks
        # x["masks"] = x["masks"] > 0
        # # Repeat channels
        # transform = RepeatChannels(3)
        # x["masks"] = transform.forward(x["masks"])
        # x["boxes"] = x["boxes"].repeat(3, 1)
        return x

    # Load dataset with sample data
    project_name: str = "SAMFineTuning"
    preprocess_name: str = "lstudio"
    list_datasource_names: list[str] = [
        "WellD",
    ]
    class_list: list[str] = [
        "camada condutiva",
        "fratura condutiva",
        "fratura induzida",
        "fratura parcial",
        "vug",
    ]
    others_class_list: list[str] = ["outros"]
    transform = transform_func
    target_transform = target_transform_func

    dataset = SamDataset(
        project_name,
        preprocess_name,
        list_datasource_names,
        class_list,
        others_class_list,
        transform,
        target_transform,
        target_boxes=False,
        target_labels=True,
        target_masks=True,
        masks_location="labels",
        boxes_location="masks",
    )

# %%
if __name__ == "__main__":
    print(f"Dataset length: {len(dataset)}")
    img, labels = dataset[np.random.randint(0, len(dataset))]
    print(f"Sample image shape: {img.shape}")
    print(f"Sample mask shape: {labels['masks'].shape}")
    print(f"Samples boxes: {labels['boxes']}")

# %%
if __name__ == "__main__":
    # Plot sample image with box, and image with mask
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(T.ToPILImage()(img))
    axs[0].add_patch(
        plt.Rectangle(
            (labels["boxes"][0][0], labels["boxes"][0][1]),
            labels["boxes"][0][2] - labels["boxes"][0][0],
            labels["boxes"][0][3] - labels["boxes"][0][1],
            edgecolor="r",
            facecolor="none",
        )
    )
    axs[1].imshow(labels["masks"][0], cmap="Oranges")
    plt.show()

# %%
