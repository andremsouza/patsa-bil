"""This module contains the base class for PyTorch Lightning DataModules in the imagelog project.
"""

from abc import ABC, abstractmethod
from typing import Optional

from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader, random_split

from imagelog_ai.base.base_dataset import BaseDataset


class LitBaseDataModule(ABC, LightningDataModule):
    """
    Base class for PyTorch Lightning DataModules in the imagelog project.

    Args:
        project_name (str): The name of the project.
        preprocess_name (str): The name of the preprocessing method.
        lists_datasource_names (dict[str, list[str]]): A dictionary mapping data source names to
        a list of data file names.
        validation_size (float, optional): The proportion of the training dataset to use for
        validation.
            Defaults to None.
        batch_size (int, optional): The batch size for the dataloaders.
            Defaults to 1.
        num_workers (int, optional): The number of worker processes to use for data loading.
            Defaults to None.
        persistent_workers (bool, optional): Whether to keep the worker processes alive between
        data loading iterations.
            Defaults to True.
        drop_last (bool, optional): Whether to drop the last incomplete batch if the dataset size
        is not divisible by the batch size.
            Defaults to False.
    """

    def __init__(
        self,
        project_name: str,
        preprocess_name: str,
        lists_datasource_names: dict[str, list[str]],
        validation_size: Optional[float] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        persistent_workers: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__()

        self.project_name: str = project_name
        self.preprocess_name: str = preprocess_name
        self.lists_datasource_names: dict[str, list[str]] = lists_datasource_names
        self.validation_size: Optional[float] = validation_size
        self.batch_size: int = batch_size
        self.num_workers: Optional[int] = num_workers
        self.persistent_workers: bool = persistent_workers
        self.drop_last: bool = drop_last

        self.dataloader_info: dict = dict.fromkeys(["fit", "val", "test"], None)

    @abstractmethod
    def _build_dataset(self, stage: str) -> BaseDataset:
        """
        Abstract method to build the dataset for a given stage.

        Args:
            stage (str): The stage of the data module (fit, val, test, predict).

        Returns:
            BaseDataset: The dataset object.
        """
        raise NotImplementedError

    def _build_dataloader(self, dataset: BaseDataset, stage: str) -> DataLoader:
        """
        Builds the dataloader for a given dataset and stage.

        Args:
            dataset (BaseDataset): The dataset object.
            stage (str): The stage of the data module (fit, val, test, predict).

        Returns:
            DataLoader: The dataloader object.
        """
        return DataLoader(
            dataset=dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers if self.num_workers is not None else 0,
            persistent_workers=self.persistent_workers,
            drop_last=self.drop_last,
            pin_memory=False,
            shuffle=True if stage == "fit" else False,
        )

    def setup(self, stage: str) -> None:
        """
        Sets up the data module for a given stage.

        Args:
            stage (str): The stage of the data module (fit, val, test, predict).
        """
        if stage == "fit":
            self.dataset_train = self._build_dataset(stage="fit")
            self.dataloader_info["fit"] = self.dataset_train.dataset_info()

            if self.validation_size:
                self.dataset_train, self.dataset_val = random_split(
                    self.dataset_train, [1 - self.validation_size, self.validation_size]
                )
            else:
                self.dataset_val = self._build_dataset(stage="val")
                self.dataloader_info["val"] = self.dataset_val.dataset_info()

        elif stage == "test":
            self.dataset_test = self._build_dataset(stage="test")
            self.dataloader_info["test"] = self.dataset_test.dataset_info()

        elif stage == "predict":
            self.dataset_predict = self._build_dataset(stage="predict")

    def train_dataloader(self) -> DataLoader:
        """
        Returns the dataloader for the training dataset.

        Returns:
            DataLoader: The dataloader object.
        """
        return self._build_dataloader(self.dataset_train, stage="fit")

    def val_dataloader(self) -> DataLoader:
        """
        Returns the dataloader for the validation dataset.

        Returns:
            DataLoader: The dataloader object.
        """
        return self._build_dataloader(self.dataset_val, stage="val")

    def test_dataloader(self) -> DataLoader:
        """
        Returns the dataloader for the test dataset.

        Returns:
            DataLoader: The dataloader object.
        """
        return self._build_dataloader(self.dataset_test, stage="test")

    def predict_dataloader(self) -> DataLoader:
        """
        Returns the dataloader for the prediction dataset.

        Returns:
            DataLoader: The dataloader object.
        """
        return self._build_dataloader(self.dataset_predict, stage="predict")
