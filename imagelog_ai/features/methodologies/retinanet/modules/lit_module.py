"""This module contains the LitFasterRCNN class, which is a PyTorch Lightning Module for Mask R-CNN.
"""

from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from torchvision.models.detection import (
    retinanet_resnet50_fpn_v2,
    RetinaNet_ResNet50_FPN_V2_Weights,
)
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor


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


class LitRetinaNet(pl.LightningModule):
    """PyTorch Lightning Module for Faster R-CNN.

    Attributes
    ----------
    MASK_THRESHOLD: float
        Threshold for the masks.
    num_classes: int
        Number of classes.
    batch_size: int
        Batch size.
    num_frozen_blocks: int
        Number of frozen blocks.
    early_stopping_patience: Optional[int]
        Early stopping patience.
    lr_scheduler_patience: Optional[int]
        Learning rate scheduler patience.
    learning_rate: float
        Learning rate.
    weight_decay: float
        Weight decay.
    network: Sequential
        Network.
    trainable_params: list
        Trainable parameters.
    train_metrics: MeanAveragePrecision
        Train metrics.
    val_metrics: MeanAveragePrecision
        Validation metrics.
    test_metrics: MeanAveragePrecision
        Test metrics.

    Methods
    -------
    _build_network()
        Build the Network.
    trainable_layers_setup()
        Freeze some layers of the network setting the parameters that will be trained.
    forward(images, targets=None)
        Forward pass.
    _check_degenerate_boxes(targets)
        Check degenerate bounding boxes (negative height or width).
    training_step(batch, batch_idx)
        Training step.
    validation_step(batch, batch_idx)
        Validation step.
    test_step(batch, batch_idx)
        Test step.
    configure_callbacks()
        Configure the callbacks used by the Model.
    configure_optimizers()
        Configure the optimizer used by the Model.
    """

    MASK_THRESHOLD: float = 0.5

    def __init__(
        self,
        num_classes: int,
        batch_size: int,
        num_frozen_blocks: int = 0,
        early_stopping_patience: Optional[int] = None,
        lr_scheduler_patience: Optional[int] = None,
        learning_rate: float = 1e-5,
        weight_decay: float = 0,
    ) -> None:
        """Constructor for the LitFasterRCNN class.

        Parameters
        ----------
        num_classes: int
            Number of classes.
        batch_size: int
            Batch size.
        num_frozen_blocks: int
            Number of frozen blocks.
        early_stopping_patience: Optional[int]
            Early stopping patience.
        lr_scheduler_patience: Optional[int]
            Learning rate scheduler patience.
        learning_rate: float
            Learning rate.
        weight_decay: float
            Weight decay.
        """
        super().__init__()

        self.num_classes: int = num_classes
        self.batch_size: int = batch_size
        self.num_frozen_blocks: int = num_frozen_blocks
        self.early_stopping_patience: Optional[int] = early_stopping_patience
        self.lr_scheduler_patience: Optional[int] = lr_scheduler_patience
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay

        self.network = self._build_network()

        self.trainable_params: list = []
        self.trainable_layers_setup()

        self.train_metrics = MeanAveragePrecision(
            iou_type="segm",
            iou_thresholds=list(np.arange(0.05, 0.95, 0.05)),
            extended_summary=True,
        )
        self.val_metrics = MeanAveragePrecision(
            iou_type="segm",
            iou_thresholds=list(np.arange(0.05, 0.95, 0.05)),
            extended_summary=True,
        )
        self.test_metrics = MeanAveragePrecision(
            iou_type="segm",
            iou_thresholds=list(np.arange(0.05, 0.95, 0.05)),
            extended_summary=True,
        )

        self.save_hyperparameters()

    def _build_network(self):
        """Method to build the Network.

        Returns
        -------
        network: Sequential
            Returns a Sequential with the layers of the network that will be used for training,
                testing and predict.
        """
        network = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            trainable_backbone_layers=5 - self.num_frozen_blocks,
        )

        return network

    def trainable_layers_setup(self):
        """Freeze some layers of the network setting the parameters that will be trained."""
        network_named_parameters = list(self.network.named_parameters())
        for _, p in network_named_parameters:
            if p.requires_grad:
                self.trainable_params.append(p)

    def forward(self, images, targets=None):
        return self.network(images, targets)

    def _check_degenerate_boxes(self, targets):
        """Check degenerate bounding boxes (negative height or width).

        Parameters
        ----------
        targets: list[dict]
            Tensor of the inputs.
        """
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    # print the first degenerate box
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: list[float] = boxes[bb_idx].tolist()
                    raise ValueError(
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}."
                    )

    def training_step(self, batch, batch_idx):
        # Batch
        x, y = batch  # tuple unpacking
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # Check if y is a list
        if not isinstance(y, list):
            y = [y]

        self._check_degenerate_boxes(y)

        loss_dict = self(x, y)

        loss = sum(loss for loss in loss_dict.values())
        loss_dict["total_loss"] = loss
        self.log_dict(loss_dict, prog_bar=True, batch_size=self.batch_size)
        return loss

    def validation_step(self, batch, batch_idx):
        # Batch
        x, y = batch  # tuple unpacking
        # Check number of dimensions for x
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # Check if y is a list
        if not isinstance(y, list):
            y = [y]
        # Convert masks to uint8
        for target in y:
            target["masks"] = target["masks"].to(torch.uint8)

        self._check_degenerate_boxes(y)

        preds = self(x, y)
        self.add_masks_to_predictions(preds, list(x.shape))

        # Log masks with MLFlow logger
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(x[0].cpu()),
            artifact_file=f"batch-{batch_idx}_val_image.png",
            run_id=self.logger.run_id,
        )
        # Log annotations with MLFlow logger
        union_annotations = torch.sum(y[0]["masks"], dim=0).float()
        union_annotations = torch.where(union_annotations > 1, 1, union_annotations)
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(union_annotations.cpu().unsqueeze(0)),
            artifact_file=f"batch-{batch_idx}_val_union_annotations.png",
            run_id=self.logger.run_id,
        )
        # Log mask with highest score
        highest_score_idx = torch.argmax(preds[0]["scores"])
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(preds[0]["masks"][highest_score_idx].cpu()),
            artifact_file=f"batch-{batch_idx}_val_mask.png",
            run_id=self.logger.run_id,
        )
        # Log union of masks
        union_mask = torch.sum(preds[0]["masks"], dim=0)
        union_mask = torch.where(union_mask > 1, 1, union_mask)
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(union_mask.cpu()),
            artifact_file=f"batch-{batch_idx}_val_union_mask.png",
            run_id=self.logger.run_id,
        )

        # Squeeze preds masks
        for pred in preds:
            pred["masks"] = torch.squeeze(
                pred["masks"] > LitRetinaNet.MASK_THRESHOLD, dim=1
            ).to(torch.uint8)

        map_dict = self.val_metrics(preds, y)
        self.log(
            "mAP50_Val",
            map_dict["map_50"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "mAP05-95_Val",
            map_dict["map"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

    def test_step(self, batch, batch_idx):
        # Batch
        x, y = batch  # tuple unpacking
        # Check number of dimensions for x
        if x.dim() == 3:
            x = x.unsqueeze(0)
        # Check if y is a list
        if not isinstance(y, list):
            y = [y]
        # Convert masks to uint8
        for target in y:
            target["masks"] = target["masks"].to(torch.uint8)

        self._check_degenerate_boxes(y)

        preds = self(x, y)
        self.add_masks_to_predictions(preds, list(x.shape))

        # Log masks with MLFlow logger
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(x[0].cpu()),
            artifact_file=f"batch-{batch_idx}_val_image.png",
            run_id=self.logger.run_id,
        )
        # Log mask with highest score
        highest_score_idx = torch.argmax(preds[0]["scores"])
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(preds[0]["masks"][highest_score_idx].cpu()),
            artifact_file=f"batch-{batch_idx}_val_mask.png",
            run_id=self.logger.run_id,
        )
        # Log union of masks
        union_mask = torch.sum(preds[0]["masks"], dim=0)
        union_mask = torch.where(union_mask > 1, 1, union_mask)
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(union_mask.cpu()),
            artifact_file=f"batch-{batch_idx}_val_union_mask.png",
            run_id=self.logger.run_id,
        )

        # Squeeze preds masks
        for pred in preds:
            pred["masks"] = torch.squeeze(
                pred["masks"] > LitRetinaNet.MASK_THRESHOLD, dim=1
            ).to(torch.uint8)

        map_dict = self.test_metrics(preds, y)
        self.log(
            "mAP50_Test",
            map_dict["map_50"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )
        self.log(
            "mAP05-95_Test",
            map_dict["map"],
            prog_bar=True,
            on_step=False,
            on_epoch=True,
            batch_size=self.batch_size,
        )

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
                    stopping_threshold=1e-04,
                    monitor="total_loss",
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
                verbose=True,
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "total_loss",
                },
            }
        else:
            return {"optimizer": optimizer}

    def add_masks_to_predictions(self, outputs: dict, inputs_shape: list):
        masks_shape = np.ones((4,), np.int32)
        masks_shape[-2:] = inputs_shape[-2:]

        for input_index in range(inputs_shape[0]):
            boxes = outputs[input_index]["boxes"]
            masks_shape[0] = boxes.shape[0]
            masks = np.zeros(masks_shape)

            for bbox_index, bbox in enumerate(boxes):
                x1 = int(bbox[0])
                y1 = int(bbox[1])
                x2 = int(bbox[2])
                y2 = int(bbox[3])

                masks[bbox_index, :, x1:x2, y1:y2] = 1

            outputs[input_index]["masks"] = torch.Tensor(masks)

        return outputs
