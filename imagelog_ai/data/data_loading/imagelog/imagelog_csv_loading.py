"""This module contains CsvToImageConverter, responsible for converting CSV files to images."""

import os
from typing import Any, Dict, Final

import numpy as np
import pandas
from PIL import Image
from tqdm import tqdm

from imagelog_ai.data.data_loading.utils.array_functions import (
    matrix_2D_gray_3channels,
    array_to_tiles,
)
from imagelog_ai.utils.os_functions import sort_listdir
from imagelog_ai.utils import argcheck

ImageLogData = Dict[str, np.ndarray]

# Compute mask and crop useful subimage.
INVALID_VALUE: Final = -9999.0  # Constant value indicading invalid value.
MAXIMUM_VALUE: Final = 40000  # Constant value indicading maximum value.


def load_image_log_csv(csv_path: str) -> ImageLogData:
    """Load an image log from a CSV file.

    Args:
        csv_path (str): The path to the CSV file.

    Returns:
        ImageLogData: A dictionary containing the signal, depths, angles, and mask.
    """
    argcheck.is_instance(str, csv_path=csv_path)

    # Load raw acoustic image log table.
    table = pandas.read_csv(
        csv_path, sep=";", decimal=",", dtype=np.float32
    ).to_numpy()  # Lê o arquivo csv e converte para um array numpy

    # Keep signal, depth, and angles.
    depths = table[:, 0]  # Extrair a amplitude do poço.
    # Obter o sinal ignorando a primeira coluna (amplitude do poço) e a última coluna se todos
    # os valores na última coluna são NaN.
    # Resolução do sinal e da máscara que o acompanha.
    signal = table[:, 1:-1] if np.all(np.isnan(table[:, -1])) else table[:, 1:]
    (
        signal_height,
        signal_width,
    ) = signal.shape
    # Calcular ângulos associados à colunas do sinal.
    angles = np.linspace(0.0, 2 * np.pi, signal_width, endpoint=False, dtype=np.float32)
    # Máscara booleana que indica com True quais pixes do sinal são inválidos
    # e com False quais são válidos.
    mask = np.logical_or(np.isnan(signal), np.isinf(signal))
    signal[mask] = INVALID_VALUE
    signal[signal < INVALID_VALUE] = INVALID_VALUE
    mask = signal == INVALID_VALUE

    valid_row = np.any(
        np.logical_not(mask), axis=1
    )  # Vetor booleano com True se pelo menos um dos valores da linha (eixo) for válido.
    end = signal_height - np.argmax(
        valid_row[::-1]
    )  # Índice da linha seguinte à última linha com algum valor válido.
    begin = np.argmax(valid_row)  # Índice da primeira linha com algum valor válido.

    depths = depths[begin:end]  # Profundidades sem a parte inválida inicial e final.
    signal = signal[begin:end, ...]  # Sinal sem a parte inválida inicial e final.
    mask = mask[begin:end, ...]  # Máscara sem a parte inválida inicial e final.

    # Move invalid values.
    signal[signal == INVALID_VALUE] = np.unique(signal)[1] - 10
    # Erase big outiliers.
    signal[signal > MAXIMUM_VALUE] = MAXIMUM_VALUE

    loaded_data: ImageLogData = {
        "signal": signal,
        "depths": depths,
        "angles": angles,
        "mask": mask,
    }

    # Return the image log.
    return loaded_data


class CsvToImageConverter:
    """Converts CSV files to images.

    Args:
        input_data_folder (str): The name of the folder containing the CSV files.
        output_data_folder (str): The name of the folder to save the images.
    """

    def __init__(
        self, input_data_folder: str, output_data_folder: str, **kwargs: Any
    ) -> None:
        """Initialize the CsvToImageConverter.

        Args:
            input_data_folder (str): The name of the folder containing the CSV files.
            output_data_folder (str): The name of the folder to save the images.
        """
        super().__init__(**kwargs)
        argcheck.is_instance(
            str,
            input_data_folder=input_data_folder,
            output_data_folder=output_data_folder,
        )

        input_folder_path = f"data/raw/{input_data_folder}"
        self.csv_paths = []
        for filename in sort_listdir(input_folder_path):
            if filename.lower().endswith(".csv"):
                self.csv_paths.append(os.path.join(input_folder_path, filename))

        self.output_folder_path = f"data/interim/{output_data_folder}"

        self.signal_processing = matrix_2D_gray_3channels

    def _generate_data(self) -> None:
        """Generate the image data from the CSV files."""
        os.makedirs(self.output_folder_path, exist_ok=True)

        image_counter = 0
        for csv_path in self.csv_paths:
            loaded_data = load_image_log_csv(csv_path)

            loaded_data["signal"] = self.signal_processing(loaded_data["signal"])

            tiled_data = array_to_tiles(
                loaded_data["signal"], loaded_data["signal"].shape[-2]
            )
            print(tiled_data.shape)

            for image in tqdm(tiled_data, desc="Generating image data"):
                # image *= 255
                # image = image.astype(np.uint8)
                pil_image = Image.fromarray(image[:, :, 0])

                pil_image.save(f"{self.output_folder_path}/{image_counter:06d}.tif")
                image_counter += 1
