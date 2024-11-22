"""This module contains classes for preprocessing images for different types of models."""

import json
import os
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Final

import torch
import torchvision
from PIL import Image
from torchvision.transforms import Resize
from tqdm import tqdm

from imagelog_ai.data.data_preprocessing.setup_transforms import ComposedTransforms
from imagelog_ai.utils import argcheck
from imagelog_ai.utils.os_functions import sort_listdir


class BaseImagesProcessor(ABC):
    """Base class for image processors."""

    def __init__(
        self,
        replace_existing_data: bool,
        save_as_image: bool,
        project_name: str,
        **kwargs,
    ) -> None:
        argcheck.is_instance(
            bool,
            replace_existing_data=replace_existing_data,
            save_as_image=save_as_image,
        )
        argcheck.is_instance(
            str,
            project_name=project_name,
        )

        self.replace_existing_data: Final = replace_existing_data
        self.to_tensor: Final = torchvision.transforms.ToTensor()
        self.to_pil: Final = torchvision.transforms.ToPILImage()
        self.save_as_image: Final = save_as_image
        self.project_name: Final = project_name
        self.composition: ComposedTransforms

        if self.save_as_image:
            self.save_func = self._save_pil_img
        else:
            self.save_func = self._save_torch_tensor

        self._make_compose()
        assert self.composition is not None, "Composition must be initialized"

    def _make_compose(self) -> None:
        self.composition = ComposedTransforms(
            project_name=self.project_name,
            to_image=self.save_as_image,
            configuration_name="process",
        )

    def _configure_data_paths(self, data_name):
        self.output_path = os.path.join("data/processed/", self.project_name, data_name)
        self.input_path = os.path.join("data/interim/", data_name)

    def _check_skip_processing(self):
        if os.path.exists(self.output_path):
            if self.replace_existing_data:
                shutil.rmtree(self.output_path)
            else:
                return True

        os.makedirs(self.output_path, exist_ok=True)
        return False

    def _save_torch_tensor(self, input_data: torch.Tensor, path) -> None:
        torch.save(input_data, path.with_suffix(".pt"))

    def _save_pil_img(self, input_data, path) -> None:
        input_data.save(path.with_suffix(".png"))

    def _process_images(self, image_names_list: str, data_name: str):
        image_iterable = tqdm(image_names_list, desc=f"Preprocessing {data_name}")

        for image_name in image_iterable:
            with Image.open(f"{self.input_path}/{image_name}") as image:
                image_tensor = self.to_tensor(image)

                if image_tensor.dtype == torch.int16:
                    image_tensor = image_tensor.float()  # [-2^15, 2^15]
                    image_tensor /= 2**16  # [-0.5, 0.5]
                    image_tensor += 0.5  # [0, 1]

                if image_tensor.size()[0] == 1:
                    image_tensor = image_tensor.repeat(3, 1, 1)

                image_tensor = self.composition(image_tensor)

                self.save_func(image_tensor, Path(f"{self.output_path}/{image_name}"))

    @abstractmethod
    def run(self, data_name, **kwargs) -> Any:
        """Run the image processor.

        Parameters
        ----------
        data_name : str
            The name of the data.

        Returns
        -------
        Any
            The output of the image processor.
        """
        raise NotImplementedError


class UnlabeledImagesProcessor(BaseImagesProcessor):
    """Class for preprocessing images for unlabeled data."""

    def __init__(
        self,
        replace_existing_data: bool,
        save_as_image: bool,
        project_name: str,
        **kwargs,
    ) -> None:
        """Initialize the UnlabeledImagesProcessor class.

        Parameters
        ----------
        replace_existing_data : bool
            Whether to replace existing data.
        save_as_image : bool
            Whether to save the data as images.
        project_name : str
            The name of the project.
        """
        super().__init__(replace_existing_data, save_as_image, project_name, **kwargs)

    def run(self, data_name, **kwargs) -> None:
        self._configure_data_paths(data_name)
        if self._check_skip_processing():
            return

        self._process_images(sort_listdir(self.input_path), data_name)


class ClassifierImagesProcessor(BaseImagesProcessor):
    """Class for preprocessing images for classification models."""

    def __init__(
        self,
        replace_existing_data: bool,
        save_as_image: bool,
        project_name: str,
        **kwargs,
    ) -> None:
        """Initialize the ClassifierImagesProcessor class.

        Parameters
        ----------
        replace_existing_data : bool
            Whether to replace existing data.
        save_as_image : bool
            Whether to save the data as images.
        project_name : str
            The name of the project.
        """
        super().__init__(replace_existing_data, save_as_image, project_name, **kwargs)

    def run(self, data_name, **kwargs) -> None:
        self._configure_data_paths(data_name)
        if self._check_skip_processing():
            return

        image_names_list = []
        image_class_list = []
        class_dictionary = {}
        class_index = 0

        for folder_name in sort_listdir(self.input_path):
            os.makedirs(f"{self.output_path}/{folder_name}", exist_ok=True)
            file_name_list = sort_listdir(f"{self.input_path}/{folder_name}")
            image_names_list += [folder_name + "/" + s for s in file_name_list]
            image_class_list += [class_index] * len(file_name_list)
            class_dictionary[class_index] = folder_name
            class_index += 1

        if not self.save_as_image:
            specs = {"class_to_index": {v: k for k, v in class_dictionary.items()}}
            with open(f"{self.output_path}/specs.json", "w") as outfile:
                json.dump(specs, outfile, indent=2)

        self._process_images(image_names_list, data_name)


class ObjectDetectorImagesProcessor(BaseImagesProcessor):
    """Class for preprocessing images for object detection models."""

    def __init__(
        self,
        replace_existing_data: bool,
        save_as_image: bool,
        project_name: str,
        **kwargs,
    ) -> None:
        """Initialize the ObjectDetectorImagesProcessor class.

        Parameters
        ----------
        replace_existing_data : bool
            Whether to replace existing data.
        save_as_image : bool
            Whether to save the data as images.
        project_name : str
            The name of the project.
        """
        super().__init__(replace_existing_data, save_as_image, project_name, **kwargs)

    def run(self, data_name, **kwargs) -> None:
        self._configure_data_paths(data_name)
        if self._check_skip_processing():
            return

        os.makedirs(os.path.join(self.output_path, "annotations"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images"), exist_ok=True)

        annotation_folder_path = os.path.join(self.input_path, "annotations")
        image_folder_path = os.path.join(self.input_path, "images")

        image_file_list = sort_listdir(image_folder_path)

        images_paths_list = [f"images/{file}" for file in image_file_list]

        for image_name in image_file_list:
            json_path = os.path.join(annotation_folder_path, f"{image_name[:-4]}.json")
            if os.path.isfile(json_path):
                shutil.copyfile(
                    json_path,
                    os.path.join(
                        self.output_path, "annotations", f"{image_name[:-4]}.json"
                    ),
                )
            else:
                with open(json_path, "w", encoding="utf-8") as specs_file:
                    json.dump([], specs_file)

        self._process_images(images_paths_list, data_name)


class SegmentationImagesProcessor(BaseImagesProcessor):
    """Class for preprocessing images for segmentation models."""

    def __init__(
        self,
        replace_existing_data: bool,
        save_as_image: bool,
        project_name: str,
        **kwargs,
    ) -> None:
        """Initialize the SegmentationImagesProcessor class.

        Parameters
        ----------
        replace_existing_data : bool
            Whether to replace existing data.
        save_as_image : bool
            Whether to save the data as images.
        project_name : str
            The name of the project.
        """
        super().__init__(replace_existing_data, save_as_image, project_name, **kwargs)

    def pil_loader(self, path: str) -> torch.Tensor:
        """Load an image using PIL.

        Parameters
        ----------
        path : str
            The path to the image.

        Returns
        -------
        torch.Tensor
            The image as a tensor.
        """
        with open(path, "rb") as f:
            img = Image.open(f)
            return self.to_tensor(img.convert("RGB"))

    def tensor_loader(self, path: str) -> torch.Tensor:
        """Load an image using torch.

        Parameters
        ----------
        path : str
            The path to the image.

        Returns
        -------
        torch.Tensor
            The image as a tensor.
        """
        return torch.load(path)

    def run(self, data_name, **kwargs) -> None:
        self._configure_data_paths(data_name)
        if self._check_skip_processing():
            return

        self.image_folder_path = os.path.join(self.input_path, "images")
        self.masks_folder_path = os.path.join(self.input_path, "masks")

        self.image_file_list = sort_listdir(self.image_folder_path)

        self.images_paths_list = [f"images/{file}" for file in self.image_file_list]
        self.masks_paths_list = [f"masks/{file}" for file in self.image_file_list]

        os.makedirs(os.path.join(self.output_path, "masks"), exist_ok=True)
        os.makedirs(os.path.join(self.output_path, "images"), exist_ok=True)

        self._process_images(self.images_paths_list, data_name)
        self._process_mask(self.image_file_list, data_name)

    def _process_mask(self, mask_names_list: str, data_name: str):

        if self.save_as_image:
            corraletd_image_loader = self.pil_loader
            corraletd_image_names_list = mask_names_list
        else:
            corraletd_image_loader = self.tensor_loader
            corraletd_image_names_list = [
                name.split(".")[0] + ".pt" for name in mask_names_list
            ]

        for mask_name, corraletd_image_name in tqdm(
            zip(mask_names_list, corraletd_image_names_list),
            desc=f"Preprocessing Masks {data_name}",
            total=len(mask_names_list),
        ):

            mask_path = os.path.join(self.input_path, "masks", mask_name)
            mask_tensor = self.pil_loader(mask_path)

            image_path = os.path.join(self.output_path, "images", corraletd_image_name)
            corraletd_image = corraletd_image_loader(image_path)
            corraletd_image_shape = corraletd_image.shape[1:]
            resize_transformation = Resize(corraletd_image_shape, antialias=1)
            mask_tensor = resize_transformation(mask_tensor)

            mask_tensor = (mask_tensor > 0.05).float()

            if self.save_as_image:
                mask_tensor = self.to_pil(mask_tensor)
            self.save_func(
                mask_tensor, Path(os.path.join(self.output_path, "masks", mask_name))
            )
