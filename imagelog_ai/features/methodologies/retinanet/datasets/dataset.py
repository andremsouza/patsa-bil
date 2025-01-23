"""This module contains the MaskRCNNDataset to load images and labels for Mask R-CNN training."""

from typing import Callable, Optional

import numpy as np
import pandas as pd
import skimage
import torch
import torchvision
from torchvision import tv_tensors

from imagelog_ai.features.methodologies.sam.datasets.dataset import SamDataset
from imagelog_ai.features.methodologies.sam.utils.labelstudio import (
    get_masks_from_annotation,
)
from imagelog_ai.utils.io_functions import json_load, tensor_load
from imagelog_ai.utils.mask_functions import get_bounding_box


class RetinaNetDataset(SamDataset):
    """Class to load images and labels for Faster R-CNN training."""

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
        """Initialize MaskRCNNDataset.

        Args:
            project_name (str): Name of the project.
            preprocess_name (str): Name of the preprocess.
            list_datasource_names (list[str]): List of datasource names.
            class_list (list[str]): List of class names.
            others_class_list (Optional[list[str]], optional): List of other class names. Defaults to None.
            background_class (bool, optional): Whether to include a background class. Defaults to False.
            transform (Optional[Callable], optional): Transform to apply to the images. Defaults to None.
            target_transform (Optional[Callable], optional): Transform to apply to the targets. Defaults to None.
            target_labels (bool, optional): Whether to include labels in the targets. Defaults to False.
            target_masks (bool, optional): Whether to include masks in the targets. Defaults to False.
            target_boxes (bool, optional): Whether to include bounding boxes in the targets. Defaults to False.
            masks_location (str, optional): Location of the masks. Defaults to "labels".
            boxes_location (str, optional): Location of the bounding boxes. Defaults to "labels".
        """
        super().__init__(
            project_name,
            preprocess_name,
            list_datasource_names,
            class_list,
            others_class_list,
            background_class,
            transform,
            target_transform,
            target_labels,
            target_masks,
            target_boxes,
            masks_location,
            boxes_location,
        )
        self.dataframe_fasterrcnn = self._build_fasterrcnn_dataframe()

    def _build_fasterrcnn_dataframe(self) -> pd.DataFrame:
        # Create datsaet to iterate on image/label files
        dataframe_fasterrcnn = self.dataframe.copy()
        dataframe_fasterrcnn = dataframe_fasterrcnn.loc[:, ["image_file", "label_file"]]
        # Remove duplicates
        dataframe_fasterrcnn.drop_duplicates(inplace=True)

        return dataframe_fasterrcnn

    def _load_label_fasterrcnn(self, label_path: str) -> dict:
        data = json_load(label_path)
        # Get masks
        masks = get_masks_from_annotation(data)
        # Transform masks into one mask with label numbers
        mask_unified = np.zeros_like(masks[list(masks.keys())[0]], dtype=np.uint8)
        for mask_label, mask in masks.items():
            mask_unified[mask == 1] = self.encoder(mask_label)
        # Get labels for connected components
        mask_components, num = skimage.measure.label(
            mask_unified, background=0, return_num=True
        )
        # Get list of labels for each connected component
        labels = []
        for i in range(1, num + 1):
            # Get label value of connected component in unified mask
            label = mask_unified[mask_components == i][0]
            labels.append(label)

        # Transform mask labels into a list of masks for each connected component
        mask_targets = []
        for i in range(1, num + 1):
            mask = mask_components == i
            mask_targets.append(mask.astype(np.uint8))

        target: dict[str, Optional[torch.Tensor]] = {
            "labels": None,
            "masks": None,
            "masks_components": torch.tensor(mask_components, dtype=torch.int64),
            "boxes": None,
            "area": None,
            "iscrowd": torch.tensor([0] * len(labels), dtype=torch.uint8),
        }
        if self.target_labels:
            target["labels"] = torch.tensor(labels, dtype=torch.int64)
        if self.target_masks:
            target["masks"] = torch.tensor(np.array(mask_targets), dtype=torch.int64)
        if self.target_boxes and self.boxes_location == "labels":
            target["boxes"] = torch.Tensor(data["boxes"])
            # Calculate area
            if isinstance(target["boxes"], torch.Tensor):
                target["area"] = torch.tensor(
                    [
                        (target["boxes"][i][2] - target["boxes"][i][0])
                        * (target["boxes"][i][3] - target["boxes"][i][1])
                        for i in range(len(target["boxes"]))
                    ],
                    dtype=torch.float32,
                )
        return target

    def __getitem__(self, idx: int):
        img = torchvision.io.read_image(
            self.dataframe_fasterrcnn.iloc[idx]["image_file"]
        )
        if self.mask_location == "images":
            img = img[0, :, :, :]
        if self.transform:
            img = self.transform(img)

        if self.list_labels_paths:
            labels = self._load_label_fasterrcnn(
                self.dataframe_fasterrcnn.iloc[idx]["label_file"]
            )
            if labels["masks"] is None and self.mask_location == "images":
                labels["masks"] = tensor_load(self.list_images_paths[idx])[1, :, :, :]
            if labels["boxes"] is None and self.boxes_location == "masks":
                # Get bounding boxes from masks
                labels["boxes"] = torch.tensor(
                    np.stack(
                        [
                            get_bounding_box(
                                (labels["masks_components"] == i).numpy(),
                                background_value=0,
                                perturbation=5,
                            )
                            for i in range(
                                1, np.max(labels["masks_components"].numpy()) + 1
                            )
                        ],
                        axis=0,
                    )
                )
                # Calculate area
                if isinstance(labels["boxes"], torch.Tensor):
                    labels["area"] = torch.tensor(
                        [
                            (labels["boxes"][i][2] - labels["boxes"][i][0])
                            * (labels["boxes"][i][3] - labels["boxes"][i][1])
                            for i in range(len(labels["boxes"]))
                        ],
                        dtype=torch.float32,
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

    def __len__(self) -> int:
        return len(self.dataframe_fasterrcnn)
