from typing import Optional

from imagelog_ai.base.base_preprocess import BaseImagesProcessor
from imagelog_ai.utils.io_functions import json_load, json_save


class ClassificationImagesProcessor(BaseImagesProcessor):
    """Class for preprocessing image data and labels for classification.


    Parameters
    ----------
    project_name: str
            Project name located in the `projects` directory,
            where the preprocessing setup json file is located.

    label_classes: List[str]
            List of classes that will be used to perform label encoding.

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
        super().__init__(
            project_name, preprocess_name, image_format, input_pil_mode, override
        )

    def _process_label_data(self, src_file_path: str, **kwargs_process_label) -> None:
        image_labels = json_load(json_path=src_file_path)

        # image_labels["labels"] = [
        #     self.label_encoder(label) for label in image_labels["labels"]
        # ]

        dst_file_path = self._built_dst_file_path(src_file_path)
        json_save(image_labels, f"{dst_file_path}.json")
