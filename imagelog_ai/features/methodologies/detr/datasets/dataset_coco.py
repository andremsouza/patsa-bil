# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
COCO dataset which returns image_id for evaluation.

Mostly copy-paste from
    https://github.com/pytorch/vision/blob/13b35ff/references/detection/coco_utils.py
"""
from pathlib import Path

import torch
import torch.utils.data
import torchvision
from pycocotools import mask as coco_mask

import imagelog_ai.features.methodologies.detr.datasets.transforms as T


class CocoDetection(torchvision.datasets.CocoDetection):
    """A custom dataset class for COCO detection that extends torchvision's CocoDetection.
    Args:
        img_folder (str): Path to the folder containing the images.
        ann_file (str): Path to the annotation file.
        transforms (callable, optional): A function/transform that takes in an image and its target
          and returns a transformed version.
        return_masks (bool): If True, converts COCO polygon annotations to masks.
    Methods:
        __getitem__(idx):
            Retrieves the image and target at the specified index.
            Args:
                idx (int): Index of the image and target to retrieve.
            Returns:
                tuple: A tuple containing the transformed image and target.
    """

    def __init__(self, img_folder, ann_file, transforms, return_masks):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms
        self.prepare = ConvertCocoPolysToMask(return_masks)

    def __getitem__(self, idx):
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        target = {"image_id": image_id, "annotations": target}
        img, target = self.prepare(img, target)
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def convert_coco_poly_to_mask(segmentations, height, width):
    """Convert COCO polygon annotations to binary masks.
    Args:
        segmentations (list): List of polygon annotations for each object.
            Each polygon is represented as a list of points.
        height (int): Height of the output mask.
        width (int): Width of the output mask.
    Returns:
        torch.Tensor: A tensor of shape (N, height, width) where N is the number of segmentations.
            Each element is a binary mask.
    """
    masks = []
    for polygons in segmentations:
        rles = coco_mask.frPyObjects(polygons, height, width)
        mask = coco_mask.decode(rles)
        if len(mask.shape) < 3:
            mask = mask[..., None]
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)
    return masks


class ConvertCocoPolysToMask(object):
    """A class to convert COCO polygon annotations to masks.
    Attributes:
    -----------
    return_masks : bool
        A flag to indicate whether to return masks or not.
    Methods:
    --------
    __call__(image, target):
        Converts the COCO polygon annotations to masks and processes the target annotations.
        Parameters:
        -----------
        image : PIL.Image or similar
            The input image.
        target : dict
            The target annotations in COCO format.
        Returns:
        --------
        image : PIL.Image or similar
            The input image (unchanged).
        target : dict
            The processed target annotations with the following keys:
                - "boxes": Tensor of shape (N, 4) containing the bounding boxes.
                - "labels": Tensor of shape (N,) containing the class labels.
                - "masks": Tensor of shape (N, H, W) containing the instance masks
                    (if return_masks is True).
                - "image_id": Tensor containing the image ID.
                - "keypoints": Tensor of shape (N, K, 3) containing the keypoints
                    (if present in annotations).
                - "area": Tensor of shape (N,) containing the area of the bounding boxes.
                - "iscrowd": Tensor of shape (N,) indicating if the instance is crowd.
                - "orig_size": Tensor containing the original size of the image.
                - "size": Tensor containing the size of the image.
    """

    def __init__(self, return_masks=False):
        self.return_masks = return_masks

    def __call__(self, image, target):
        w, h = image.size

        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        anno = target["annotations"]

        anno = [obj for obj in anno if "iscrowd" not in obj or obj["iscrowd"] == 0]

        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        if self.return_masks:
            segmentations = [obj["segmentation"] for obj in anno]
            masks = convert_coco_poly_to_mask(segmentations, h, w)

        keypoints = None
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                keypoints = keypoints.view(num_keypoints, -1, 3)

        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        if self.return_masks:
            masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        if self.return_masks:
            target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # for conversion to coco api
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor(
            [obj["iscrowd"] if "iscrowd" in obj else 0 for obj in anno]
        )
        target["area"] = area[keep]
        target["iscrowd"] = iscrowd[keep]

        target["orig_size"] = torch.as_tensor([int(h), int(w)])
        target["size"] = torch.as_tensor([int(h), int(w)])

        return image, target


def make_coco_transforms(image_set):
    """Create a set of transformations for COCO dataset images.

    Args:
        image_set (str): The dataset split for which to create the transformations.
                         Should be either 'train' or 'val'.

    Returns:
        torchvision.transforms.Compose: A composition of transformations to be applied to images.

    Raises:
        ValueError: If the provided image_set is not 'train' or 'val'.

    The transformations include:
    - For 'train': Random horizontal flip, random selection between random resize and a combination
        of random resize, random size crop, and another random resize, followed by normalization.
    - For 'val': Random resize followed by normalization.
    """
    normalize = T.Compose(
        [T.ToTensor(), T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]
    )

    scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

    if image_set == "train":
        return T.Compose(
            [
                T.RandomHorizontalFlip(),
                T.RandomSelect(
                    T.RandomResize(scales, max_size=800),
                    T.Compose(
                        [
                            T.RandomResize([400, 500, 600]),
                            T.RandomSizeCrop(384, 600),
                            T.RandomResize(scales, max_size=800),
                        ]
                    ),
                ),
                normalize,
            ]
        )

    if image_set == "val":
        return T.Compose(
            [
                T.RandomResize([800], max_size=800),
                normalize,
            ]
        )

    raise ValueError(f"unknown {image_set}")


def build(image_set, args):
    """Build a COCO dataset for the given image set.

    Args:
        image_set (str): The image set to use, either 'train' or 'val'.
        args (Namespace): Arguments containing the COCO path and other configurations.

    Returns:
        CocoDetection: The COCO dataset for the specified image set.

    Raises:
        AssertionError: If the provided COCO path does not exist.
    """
    root = Path(args.coco_path)
    assert root.exists(), f"provided COCO path {root} does not exist"
    mode = "instances"
    PATHS = {
        "train": (root / "train2017", root / "annotations" / f"{mode}_train2017.json"),
        "val": (root / "val2017", root / "annotations" / f"{mode}_val2017.json"),
    }

    img_folder, ann_file = PATHS[image_set]
    dataset = CocoDetection(
        img_folder,
        ann_file,
        transforms=make_coco_transforms(image_set),
        return_masks=args.masks,
    )
    return dataset
