from collections import OrderedDict
import json
from typing import Optional

from PIL import Image
import torch
from torchvision.transforms import ToTensor, ToPILImage, ConvertImageDtype


TO_TENSOR = ToTensor()
TO_FLOAT = ConvertImageDtype(torch.float)


def pil_load(
    pil_image_path: str, pil_mode: Optional[str] = "RGB", load_as_tensor: bool = True
) -> torch.Tensor:
    """Returns a tensor from the given image file (loaded as an PIL image).

    Parameters
    ----------
    pil_image_path: str
            Path to the image.

    pil_mode: str
            Mode which defines the type and depth of a pixel in the PIL image (Default is RGB).

    load_as_tensor: bool
            Load as a tensor (default) or a PIL image.
    """

    with open(pil_image_path, "rb") as f:
        img = Image.open(f)

        if pil_mode:
            img = img.convert(pil_mode)

        if load_as_tensor:
            img = TO_TENSOR(img)

            if img.dtype != torch.float:
                img = TO_FLOAT(img)

        return img


def pil_save(tensor_image: torch.Tensor, dst_file_path: str) -> None:
    """Saves a tensor as a PIL image at a given destination path.

    Parameters
    ----------
    tensor_image: torch.Tensor
            Tensor that will be saved as a PIL image.

    dst_file_path: str
            Path where the PIL image will be saved.
    """
    to_pil = ToPILImage()
    pil_image = to_pil(tensor_image)
    pil_image.save(dst_file_path)


def tensor_load(tensor_path: str) -> torch.Tensor:
    """Returns a tensor from the given tensor file (loaded as an .pt tensor).

    Parameters
    ----------
    tensor_path: str
            Path to the tensor.
    """
    return torch.load(tensor_path, weights_only=False)


def tensor_save(tensor: torch.Tensor, dst_file_path: str) -> None:
    """Saves a tensor as .pt tensor at a given destination path.

    Parameters
    ----------
    tensor: torch.Tensor
            Tensor that will be saved as a .pt tensor.

    dst_file_path: str
            Path where the .pt tensor will be saved.
    """
    torch.save(tensor, dst_file_path)


def json_load(json_path: str) -> dict:
    """Returns a dictonary from the given json file.

    Parameters
    ----------
    json_path: str
            Path to the json.
    """
    with open(json_path, "r", encoding="utf-8") as specs_file:
        loaded_params: dict = json.load(specs_file, object_pairs_hook=OrderedDict)

        return loaded_params


def json_save(dict_to_save: dict, dst_file_path: str) -> None:
    """Saves a dict as .json file at a given destination path.

    Parameters
    ----------
    dict_to_save: dict
            dict that will be saved as a .json file.

    dst_file_path: str
            Path where the .pt tensor will be saved.
    """
    with open(dst_file_path, "w", encoding="utf-8") as f:
        json.dump(dict_to_save, f, indent=4)
