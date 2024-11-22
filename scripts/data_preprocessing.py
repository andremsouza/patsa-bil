"""This module is used to preprocess the image data for the project."""

from argparse import ArgumentParser
import os
import sys

import dotenv
import torch

dotenv.load_dotenv()
sys.path.append(os.getenv("PACKAGEPATH", ""))

from endpoints.data_preprocessing.endpoints import run_image_preprocessing
from imagelog_ai.utils.argparse import ParserTree


def _add_group_for_image_preprocessing(parser: ArgumentParser) -> None:
    """Add the flags related to the image preprocessing

    Parameters
    ----------
    parser : ArgumentParser
            Parser instance that will be used to parse the arguments received in the command line.
    """
    group = parser.add_argument_group(
        "Transfer Learning Data Module",
        description="Arguments for image preprocessing",
    )
    group.add_argument(
        "--project_name",
        metavar="NAME",
        type=str,
        required=True,
        help="Define the name of the project  [REQUIRED]",
    )
    group.add_argument(
        "--save_as_image",
        action="store_true",
        help="Choose to save preprocessed data as images",
    )
    group.add_argument(
        "--replace_existing_data",
        action="store_true",
        help="Choose to erase any existing preprocessed data",
    )
    group.add_argument(
        "--verbose",
        action="store_true",
        help="Set flag to display progress of data generation",
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    with ParserTree() as root:
        with ParserTree(
            "preprocess_images",
            parent=root,
            endpoint=run_image_preprocessing,
            help="Load and process project image data",
        ) as command:
            _add_group_for_image_preprocessing(command.parser)

        root.call_endpoint()
