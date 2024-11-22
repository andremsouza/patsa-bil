"""This module contains the endpoint for the data preprocessing."""

import json
from typing import Any, Dict, Type

from imagelog_ai.data.data_preprocessing.preprocess_images import (
    UnlabeledImagesProcessor,
    BaseImagesProcessor,
)
from imagelog_ai.data.data_preprocessing.utils.processor_dict import processor_dict


def _load_settings(project_name: str) -> Dict[str, Dict[str, Dict[str, Any]]]:
    json_path = f"projects/{project_name}/settings.json"
    try:
        with open(json_path, "r", encoding="utf-8") as specs_file:
            loaded_params: Dict[str, Dict[str, Dict[str, Any]]] = json.load(specs_file)
    except FileNotFoundError as exc:
        raise ValueError(
            f"Must create the file settings.json in projects/{project_name}/.\n"
            f"Path {json_path} not found"
        ) from exc

    return loaded_params


def run_image_preprocessing(project_name: str, **kwargs: Any) -> None:
    """Call the prediction using transfer learning model."""
    project_settings = _load_settings(project_name)

    unlabeled_processor = UnlabeledImagesProcessor(project_name=project_name, **kwargs)
    for data_name in project_settings["unlabeled_data"]:
        unlabeled_processor.run(data_name, **kwargs)

    labeled_processor_class: Type[BaseImagesProcessor] = processor_dict[
        project_settings["project_type"]
    ]
    labeled_processor: BaseImagesProcessor = labeled_processor_class(
        project_name=project_name, **kwargs
    )
    for data_name in project_settings["labeled_data"]:
        labeled_processor.run(data_name)
