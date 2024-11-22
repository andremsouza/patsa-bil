"""This module contains the endpoints for the classification API."""

from argparse import ArgumentParser
import os
import sys

import dotenv
import torch

dotenv.load_dotenv()
sys.path.append(os.getenv("PACKAGEPATH", ""))

from endpoints.classification.endpoints import (
    fit,
    test,
    fit_test,
    predict,
    kfold,
    preprocess,
)
from imagelog_ai.utils.argparse import ParserTree


def _add_group_for_image_preprocessing(parser: ArgumentParser) -> None:
    """Add the flags related to the image preprocessing

    Parameters
    ----------
    parser : ArgumentParser
            Parser instance that will be used to parse the arguments received in the command line.
    """
    group = parser.add_argument_group(
        "Classification Module",
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
        "--preprocess_name",
        metavar="NAME",
        type=str,
        required=True,
        help="Define the name of the preprocess  [REQUIRED]",
    )

    group.add_argument(
        "--override_preprocess",
        action="store_true",
        help="Override previous preprocessing",
    )


def _add_group_for_project(parser: ArgumentParser) -> None:
    """Add the flags related to the lightning data modules

    Parameters
    ----------
    parser : ArgumentParser
            Parser instance that will be used to parse the arguments received in the command line.
    """
    group = parser.add_argument_group(
        "Transfer Learning Data Module", description="Arguments for project"
    )
    group.add_argument(
        "--project_name",
        metavar="NAME",
        type=str,
        required=True,
        help="Define the name of the project  [REQUIRED]",
    )

    group.add_argument(
        "--experiment_name",
        metavar="NAME",
        type=str,
        required=True,
        help="Define the name of the experiment  [REQUIRED]",
    )

    group.add_argument(
        "--override_experiment",
        action="store_true",
        help="Override previous run of the experiment",
    )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("medium")

    with ParserTree() as root:
        with ParserTree(
            "preprocess",
            parent=root,
            endpoint=preprocess,
            help="Preprocess data from data/interim",
        ) as command:
            _add_group_for_image_preprocessing(command.parser)

        with ParserTree(
            "fit",
            parent=root,
            endpoint=fit,
            help="Perform model training with memory mapped data",
        ) as command:
            _add_group_for_project(command.parser)

        with ParserTree(
            "test",
            parent=root,
            endpoint=test,
            help="Perform model testing with memory mapped data",
        ) as command:
            _add_group_for_project(command.parser)

        with ParserTree(
            "fit_test",
            parent=root,
            endpoint=fit_test,
            help="Perform model training and testing with memory mapped data",
        ) as command:
            _add_group_for_project(command.parser)

        with ParserTree(
            "predict",
            parent=root,
            endpoint=predict,
            help="Perform model predict with memory mapped data",
        ) as command:
            _add_group_for_project(command.parser)

        with ParserTree(
            "kfold",
            parent=root,
            endpoint=kfold,
            help="Perform KFold experiment",
        ) as command:
            _add_group_for_project(command.parser)

        root.call_endpoint()
