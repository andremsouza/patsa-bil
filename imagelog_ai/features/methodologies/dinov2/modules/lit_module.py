import os
from typing import Optional, Union

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from imagelog_ai.features.methodologies.dinov2.neural_networks.dinov2 import (
    Dinov2ForSemanticSegmentation,
)


def tensor_to_numpy_transform(x: torch.Tensor) -> np.ndarray:
    """Transform a tensor to a numpy array.

    Parameters
    ----------
    x: torch.Tensor
        Tensor to be transformed.

    Returns
    -------
    np.ndarray
        Numpy array.
    """
    return x.numpy().transpose((1, 2, 0))


class LitDINOv2(pl.LightningModule):

    def __init__(
        self,
        pretrained_model_name_or_path: Optional[Union[str, os.PathLike]],
        id2label: dict[int, str],
        num_labels: Optional[int] = None,
        batch_size: int = 2,
        early_stopping_patience: Optional[int] = None,
        lr_scheduler_patience: Optional[int] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.pretrained_model_name_or_path: Optional[Union[str, os.PathLike]] = (
            pretrained_model_name_or_path
        )
        self.id2label: dict[int, str] = id2label
        self.num_labels: int = num_labels if num_labels is not None else len(id2label)
        self.batch_size: int = batch_size
        self.early_stopping_patience: Optional[int] = early_stopping_patience
        self.lr_scheduler_patience: Optional[int] = lr_scheduler_patience
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay

        self.model: Dinov2ForSemanticSegmentation = (
            Dinov2ForSemanticSegmentation.from_pretrained(
                pretrained_model_name_or_path,
                id2label=self.id2label,
                num_labels=self.num_labels,
            )
        )
        self.trainable_params: list[torch.nn.Parameter] = []
        self._setup_trainable_layers()
        self.train_metrics, self.val_metrics, self.test_metrics = self._build_metrics()
        self.save_hyperparameters()

    def forward(
        self,
        pixel_values: torch.Tensor,
        *args,
        output_hidden_states: bool = False,
        output_attentions: bool = False,
        labels: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        return self.model(pixel_values, output_hidden_states, output_attentions, labels)

    def _setup_trainable_layers(self) -> None:
        network_named_parameters = list(self.model.named_parameters())
        for _, p in network_named_parameters:
            p.requires_grad = True
            if p.requires_grad:
                self.trainable_params.append(p)

    def training_step(self, batch, batch_idx):
        pixel_values, labels = batch

        # forward pass
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        # evaluate
        predicted = outputs.logits.argmax(dim=1)

        # Log masks with MLFlow logger
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(pixel_values[0].cpu()),
            artifact_file=f"batch-{batch_idx}_train_image.png",
            run_id=self.logger.run_id,
        )
        # Log annotations with MLFlow logger
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(labels[0]["gt_mask"].cpu().float()),
            artifact_file=f"batch-{batch_idx}_train_union_annotations.png",
            run_id=self.logger.run_id,
        )
        # Log predicted masks
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(predicted[0].cpu().unsqueeze(0).float()),
            artifact_file=f"batch-{batch_idx}_train_predicted.png",
            run_id=self.logger.run_id,
        )
        # log metrics
        self.log(
            "train_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        map_preds = [
            {
                "masks": predicted[i].cpu().unsqueeze(0).byte(),
                "scores": torch.tensor([1.0]),
                "labels": torch.tensor([1]),
            }
            for i in range(predicted.shape[0])
        ]
        map_target = [
            {
                "masks": labels[i]["gt_mask"].cpu().byte(),
                "labels": torch.tensor([1]),
            }
            for i in range(predicted.shape[0])
        ]
        map_dict = self.val_metrics(map_preds, map_target)
        self.log(
            "Train_mAP50",
            map_dict["map_50"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "Train_mAP05-95",
            map_dict["map"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        pixel_values, labels = batch

        # forward pass
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        # evaluate
        predicted = outputs.logits.argmax(dim=1)

        # log metrics
        self.log(
            "val_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        # Log masks with MLFlow logger
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(pixel_values[0].cpu()),
            artifact_file=f"batch-{batch_idx}_val_image.png",
            run_id=self.logger.run_id,
        )
        # Log annotations with MLFlow logger
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(labels[0]["gt_mask"].cpu().float()),
            artifact_file=f"batch-{batch_idx}_val_union_annotations.png",
            run_id=self.logger.run_id,
        )
        # Log predicted masks
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(predicted[0].cpu().unsqueeze(0).float()),
            artifact_file=f"batch-{batch_idx}_val_predicted.png",
            run_id=self.logger.run_id,
        )
        # log metrics
        map_preds = [
            {
                "masks": predicted[i].cpu().unsqueeze(0).byte(),
                "scores": torch.tensor([1.0]),
                "labels": torch.tensor([1]),
            }
            for i in range(predicted.shape[0])
        ]
        map_target = [
            {
                "masks": labels[i]["gt_mask"].cpu().byte(),
                "labels": torch.tensor([1]),
            }
            for i in range(predicted.shape[0])
        ]
        map_dict = self.val_metrics(map_preds, map_target)
        self.log(
            "Val_mAP50",
            map_dict["map_50"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "Val_mAP05-95",
            map_dict["map"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        return loss

    def test_step(self, batch, batch_idx):
        pixel_values, labels = batch

        # forward pass
        outputs = self.model(pixel_values, labels=labels)
        loss = outputs.loss
        # evaluate
        predicted = outputs.logits.argmax(dim=1)

        # log metrics
        self.log(
            "test_loss",
            loss,
            prog_bar=True,
            on_step=True,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        # Log masks with MLFlow logger
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(pixel_values[0].cpu()),
            artifact_file=f"batch-{batch_idx}_test_image.png",
            run_id=self.logger.run_id,
        )
        # Log annotations with MLFlow logger
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(labels[0]["gt_mask"].cpu().float()),
            artifact_file=f"batch-{batch_idx}_test_union_annotations.png",
            run_id=self.logger.run_id,
        )
        # Log predicted masks
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(predicted[0].cpu().unsqueeze(0).float()),
            artifact_file=f"batch-{batch_idx}_test_predicted.png",
            run_id=self.logger.run_id,
        )
        # log metrics
        map_preds = [
            {
                "masks": predicted[i].cpu().numpy(),
                "scores": torch.tensor([1.0]),
                "labels": torch.tensor([1]),
            }
            for i in range(predicted.shape[0])
        ]
        map_target = [
            {
                "masks": labels[i]["gt_mask"].cpu().numpy(),
                "labels": torch.tensor([1]),
            }
            for i in range(predicted.shape[0])
        ]
        map_dict = self.test_metrics(map_preds, map_target)
        self.log(
            "Test_mAP50",
            map_dict["map_50"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "Test_mAP05-95",
            map_dict["map"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

        return loss

    def configure_callbacks(self) -> list:
        """Function that is run by lightning when using the module.
                        Configures the callbacks used by the Model.

        Returns
        -------
        List[Callback]
                List of Callbacks
        """
        callback_list = []
        if self.early_stopping_patience is not None:
            callback_list.append(
                EarlyStopping(
                    patience=self.early_stopping_patience,
                    monitor="val_loss",
                    mode="min",
                )
            )

        return callback_list

    def configure_optimizers(self):
        """Function that is run by lightning when using the module.
                        Configures the optimizer used by the Model.

        Returns
        -------
        dict[str, Any]
                Dictionary of optimizer configuration
        """
        optimizer = AdamW(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

        if self.lr_scheduler_patience is not None:
            scheduler = ReduceLROnPlateau(
                patience=self.lr_scheduler_patience,
                cooldown=0,
                optimizer=optimizer,
                factor=0.5,
                mode="min",
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val_loss",
                },
            }
        else:
            return {"optimizer": optimizer}

    def _build_metrics(
        self,
    ) -> tuple[MeanAveragePrecision, MeanAveragePrecision, MeanAveragePrecision]:
        train_metrics = MeanAveragePrecision(
            iou_type="segm",
            iou_thresholds=list(np.arange(0.05, 0.95, 0.05)),
            extended_summary=True,
        )
        val_metrics = MeanAveragePrecision(
            iou_type="segm",
            iou_thresholds=list(np.arange(0.05, 0.95, 0.05)),
            extended_summary=True,
        )
        test_metrics = MeanAveragePrecision(
            iou_type="segm",
            iou_thresholds=list(np.arange(0.05, 0.95, 0.05)),
            extended_summary=True,
        )
        return train_metrics, val_metrics, test_metrics
