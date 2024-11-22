"""This module contains the dictionary of data processors used in the data preprocessing module."""

from typing import Dict, Type

from imagelog_ai.data.data_preprocessing.preprocess_images import (
    ObjectDetectorImagesProcessor,
    ClassifierImagesProcessor,
    SegmentationImagesProcessor,
)

processor_dict: Dict[str, Type] = {
    "ObjectDetection": ObjectDetectorImagesProcessor,
    "Classification": ClassifierImagesProcessor,
    "Segmentation": SegmentationImagesProcessor,
}
