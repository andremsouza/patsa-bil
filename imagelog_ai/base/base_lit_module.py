"""This module contains the base class for PyTorch Lightning Modules in the imagelog project."""

from abc import ABC, abstractmethod
from typing import Any, Optional

from lightning import LightningModule
import torch


class BaseLitModule(ABC, LightningModule):
    """Base class for PyTorch Lightning Modules in the imagelog project."""

    def __init__(
        self,
        early_stopping_patience: Optional[int] = None,
        lr_scheduler_patience: Optional[int] = None,
        learning_rate: float = 1e-05,
        weight_decay: float = 0,
    ) -> None:
        super().__init__()

        self.save_hyperparameters()

    @abstractmethod
    def _build_network(self) -> Any:
        """Method to build the Network.

        Returns
        -------
        network: Sequential
            Returns a Sequential with the layers of the network that will be used for training, testing and predict.

        Raises
        ------
            NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def _build_metrics(self) -> tuple:
        """Method to build the train, val and test metrics.

        Returns
        -------
        (train_metrics, val_metrics, test_metrics): tuple
            Returns a tuple with train, val and test metrics.

        Raises
        ------
            NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, input: torch.Tensor) -> Any:
        """Method specifying the computation performed at every call.

        Parameters
        ----------
        input: Tensor
            Tensor of the inputs.

        Returns
        -------
        Tensor
            Output of the `network`.
        dict
            Loss of the `network`.

        Raises
        ------
            NotImplementedError
        """
        raise NotImplementedError

    @abstractmethod
    def shared_step(self, batch, prefix) -> torch.Tensor:
        """Common part of the steps for training, validation and test.

        Parameters
        ----------
        batch: Tensor
                Batch of data.
        prefix: str
                Name of the step that called this function.

        Returns
        -------
        LossTensor
                Tensor of calculated Loss.

        Raises
        ------
                NotImplementedError
        """
        raise NotImplementedError

    def _log_metrics(self, preds, labels, prefix: str) -> None:
        if prefix == "Train":
            self.train_metrics(preds, labels)
            self.log_dict(
                self.train_metrics,
                prog_bar=True,
                on_step=True,
                on_epoch=True,
            )

        if prefix == "Val":
            self.val_metrics(preds, labels)
            self.log_dict(
                self.val_metrics,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

        if prefix == "Test":
            self.test_metrics(preds, labels)
            self.log_dict(
                self.test_metrics,
                prog_bar=True,
                on_step=False,
                on_epoch=True,
            )

    def training_step(self, batch, batch_idx) -> torch.Tensor:
        """Function that is run by lightning when using the module.
                        Step used in training.

        Parameters
        ----------
        batch: Tensor
                Batch of data.
        batch_idx: int
                Index of the batch.

        Returns
        -------
        LossTensor
                Tensor of calculated Loss.
        """
        loss = self.shared_step(batch, "Train")
        assert loss is not None, "shared_step must return a Tensor"

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        """Function that is run by lightning when using the module.
                        Step used in validation.

        Parameters
        ----------
        batch: Tensor
                Batch of data.
        batch_idx: int
                Index of the batch.
        """
        _ = self.shared_step(batch, "Val")

    def test_step(self, batch, batch_idx) -> None:
        """Function that is run by lightning when using the module.
                        Step used in testing.

        Parameters
        ----------
        batch: Tensor
                Batch of data.
        batch_idx: int
                Index of the batch.
        """
        _ = self.shared_step(batch, "Test")

    @abstractmethod
    def predict_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        raise NotImplementedError
