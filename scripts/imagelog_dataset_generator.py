"""This module is used to generate synthetic imagelog data for the training of the models."""

from argparse import ArgumentParser
import os
import sys

import dotenv
import torch

dotenv.load_dotenv()
sys.path.append(os.getenv("PACKAGEPATH", ""))

from endpoints.imagelog_generator.endpoints import (
    synthetic_imagelog_generation,
    test_brechado,
    test_fracture,
    test_laminado,
    test_macico,
)
from imagelog_ai.utils.argparse import ParserTree


def _add_group_for_imagelog_generation(parser: ArgumentParser) -> None:
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
        "--generated_tile_shape",
        metavar="SHAPE",
        nargs="+",
        type=int,
        default=[480, 480],
        help="Define the shape of the imagelog tile when generated",
    )
    group.add_argument(
        "--output_tile_shape",
        metavar="SHAPE",
        nargs="+",
        type=int,
        default=[224, 224],
        help="Define the shape of the image that will be saved",
    )
    group.add_argument(
        "--batch_size",
        metavar="SIZE",
        type=int,
        required=True,
        help="How many tiles will be generated at each iteration, affects VRAM usage",
    )
    group.add_argument(
        "--number_of_tiles",
        metavar="SIZE",
        type=int,
        required=True,
        help="Number of tiles of each pattern to be generated",
    )
    group.add_argument(
        "--apply_colormap",
        action="store_true",
        help="Create data with an Orange/Brown colormap applied",
    )
    group.add_argument(
        "--no_blind_spots", action="store_true", help="Create data without blind spots"
    )
    group.add_argument(
        "--dataset_name",
        metavar="NAME",
        type=str,
        required=True,
        help="Name of the directory where the data is stored  [REQUIRED]",
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    with ParserTree() as root:
        with ParserTree(
            "generate_data",
            parent=root,
            endpoint=synthetic_imagelog_generation,
            help="Generate synthetic imagelog data",
        ) as command:
            _add_group_for_imagelog_generation(command.parser)

        with ParserTree(
            "test_brechado",
            parent=root,
            endpoint=test_brechado,
            help="Generate synthetic imagelog data",
        ) as command:
            _add_group_for_imagelog_generation(command.parser)
        with ParserTree(
            "test_fracture",
            parent=root,
            endpoint=test_fracture,
            help="Generate synthetic imagelog data",
        ) as command:
            _add_group_for_imagelog_generation(command.parser)
        with ParserTree(
            "test_laminado",
            parent=root,
            endpoint=test_laminado,
            help="Generate synthetic imagelog data",
        ) as command:
            _add_group_for_imagelog_generation(command.parser)
        with ParserTree(
            "test_macico",
            parent=root,
            endpoint=test_macico,
            help="Generate synthetic imagelog data",
        ) as command:
            _add_group_for_imagelog_generation(command.parser)

        root.call_endpoint()
