"""This module contains the dictionary of transforms used in the data preprocessing module."""

import json
import time
from typing import Any, Dict, Final

from torchvision.transforms import ToPILImage

from imagelog_ai.data.data_preprocessing.utils.transform_dict import Transforms
from imagelog_ai.utils import argcheck


class ComposedTransforms:
    """Compose the transforms for the data preprocessing."""

    def __init__(
        self,
        project_name: str,
        configuration_name: str,
        to_image: bool,
        verbose: bool = False,
    ) -> None:
        """Initialize the ComposedTransforms object.

        Args:
            project_name (str): The name of the project.
            configuration_name (str): The name of the configuration.
            to_image (bool): Whether to convert the output to an image.
            verbose (bool, optional): Whether to print the time taken for each transform. Defaults to False.
        """
        argcheck.is_instance(bool, to_image=to_image, verbose=verbose)
        argcheck.is_instance(str, project_name=project_name)

        self.to_image = to_image
        self.verbose = verbose

        self.json_path: Final = f"projects/{project_name}/settings.json"

        with open(self.json_path, "r", encoding="utf-8") as specs_file:
            self.compose_specs: Dict[str, Dict[str, Dict[str, Any]]] = json.load(
                specs_file
            )["transforms"][configuration_name]

        self._compose()

    def _compose(self) -> None:
        """Compose the transforms."""
        self.composition = []
        for key, arg_values in self.compose_specs.items():
            self.composition.append(Transforms[key](**arg_values))

        if self.to_image:
            self.composition.append(ToPILImage())

    def __call__(self, input_data) -> Any:
        """Apply the transforms to the input data.

        Args:
            input_data (Any): The input data.

        Returns:
            Any: The transformed input data.
        """
        for transform in self.composition:
            if self.verbose:
                start = time.process_time()
                input_data = transform(input_data)
                print(f"{transform}: {time.process_time() - start}")
            else:
                input_data = transform(input_data)

        return input_data
