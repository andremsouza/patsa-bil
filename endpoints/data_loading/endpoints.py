"""This module contains the endpoint for the data loading process."""

from typing import Any

from imagelog_ai.data.data_loading.imagelog.imagelog_csv_loading import (
    CsvToImageConverter,
)


def csv_to_images(**kwargs: Any) -> None:
    """Call the prediction using transfer learning model."""
    converter = CsvToImageConverter(**kwargs)

    converter._generate_data()
