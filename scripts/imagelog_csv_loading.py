"""This module is used to preprocess the image data for the project."""

from argparse import ArgumentParser
import os
import sys

import dotenv
import torch

dotenv.load_dotenv()
sys.path.append(os.getenv("PACKAGEPATH", ""))

from endpoints.data_loading.endpoints import csv_to_images
from imagelog_ai.utils.argparse import ParserTree


def _add_group_for_csv_to_image(parser: ArgumentParser) -> None:
    """Add the flags related to the lightning modules

    Parameters
    ----------
    parser : ArgumentParser
            Parser instance that will be used to parse the arguments received in the command line.
    """
    group = parser.add_argument_group(
        "Synthetic Imagelog Data Generation",
        description="Arguments for Synthetic Imagelog Data Generator",
    )

    group.add_argument(
        "--input_data_folder",
        metavar="NAME",
        type=str,
        required=True,
        help="Name of the directory where the CSVs are stored  [REQUIRED]",
    )
    group.add_argument(
        "--output_data_folder",
        metavar="NAME",
        type=str,
        required=True,
        help="Name of the directory where the images will be stored  [REQUIRED]",
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    with ParserTree() as root:
        with ParserTree(
            "csv_to_images",
            parent=root,
            endpoint=csv_to_images,
            help="Load and process CSVs of imagelogs",
        ) as command:
            _add_group_for_csv_to_image(command.parser)

        root.call_endpoint()
