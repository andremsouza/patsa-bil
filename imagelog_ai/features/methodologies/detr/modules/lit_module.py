"""This module contains the LitDETR class to train the DETR model using PyTorch Lightning.
"""

from typing import Optional

import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping
import numpy as np
import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchmetrics.detection.mean_ap import MeanAveragePrecision

from imagelog_ai.features.methodologies.detr.neural_networks.detr import build


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


class LitDETR(pl.LightningModule):

    MASK_THRESHOLD: float = 0.5

    def __init__(
        self,
        dataset_file: Optional[str] = None,
        num_classes: int = 1,
        num_queries: int = 100,
        aux_loss: bool = True,
        masks: bool = True,
        frozen_weights: Optional[str] = None,
        bbox_loss_coef: float = 5.0,
        giou_loss_coef: float = 2.0,
        mask_loss_coef: float = 1.0,
        dice_loss_coef: float = 3.0,
        dec_layers: int = 6,
        eos_coef: float = 0.1,
        lr_backbone: float = 1e-5,
        backbone: str = "resnet50",
        dilation: bool = False,
        hidden_dim: int = 256,
        dropout: float = 0.1,
        nheads: int = 8,
        dim_feedforward: int = 2048,
        enc_layers: int = 6,
        pre_norm: bool = False,
        set_cost_class: float = 1.0,
        set_cost_bbox: float = 5.0,
        set_cost_giou: float = 2.0,
        position_embedding: str = "sine",
        batch_size: int = 2,
        early_stopping_patience: Optional[int] = None,
        lr_scheduler_patience: Optional[int] = None,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.args = self._build_args(
            dataset_file,
            num_classes,
            num_queries,
            aux_loss,
            masks,
            frozen_weights,
            bbox_loss_coef,
            giou_loss_coef,
            mask_loss_coef,
            dice_loss_coef,
            dec_layers,
            eos_coef,
            lr_backbone,
            backbone,
            dilation,
            hidden_dim,
            dropout,
            nheads,
            dim_feedforward,
            enc_layers,
            pre_norm,
            set_cost_class,
            set_cost_bbox,
            set_cost_giou,
            position_embedding,
        )
        self.batch_size: int = batch_size
        self.early_stopping_patience: Optional[int] = early_stopping_patience
        self.lr_scheduler_patience: Optional[int] = lr_scheduler_patience
        self.learning_rate: float = learning_rate
        self.weight_decay: float = weight_decay
        self.model, self.criterion, self.postprocessors = build(self.args)
        self.trainable_params: list = []
        self._setup_trainable_layers()
        self.train_metrics, self.val_metrics, self.test_metrics = self._build_metrics()
        self.save_hyperparameters()

    def _setup_trainable_layers(self) -> None:
        network_named_parameters = list(self.model.named_parameters())
        for _, p in network_named_parameters:
            if p.requires_grad:
                self.trainable_params.append(p)

    def forward(self, images: torch.Tensor, *args, **kwargs) -> torch.Tensor:
        return self.model(images)

    def training_step(self, batch, batch_idx):
        samples, targets = batch
        if samples.dim() == 3:
            samples = samples.unsqueeze(0)
        if not isinstance(targets, list):
            targets = [targets]

        self._check_degenerate_boxes(targets)

        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        loss_dict_unscaled = {f"train_{k}_unscaled": v for k, v in loss_dict.items()}
        loss_dict_scaled = {
            f"train_{k}": v * weight_dict[k]
            for k, v in loss_dict.items()
            if k in weight_dict
        }

        self.log(
            "total_train_loss",
            losses,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log_dict(
            loss_dict_unscaled,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.batch_size,
        )
        self.log_dict(
            loss_dict_scaled,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.batch_size,
        )

        # map_dict = self.train_metrics(outputs, targets)
        # self.log(
        #     "mAP50_Train",
        #     map_dict["map_50"],
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=self.batch_size,
        # )
        # self.log(
        #     "mAP05-95_Train",
        #     map_dict["map"],
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=self.batch_size,
        # )

        return losses

    def validation_step(self, batch, batch_idx):
        samples, targets = batch
        if samples.dim() == 3:
            samples = samples.unsqueeze(0)
        if not isinstance(targets, list):
            targets = [targets]

        self._check_degenerate_boxes(targets)

        outputs = self.model(samples)
        loss_dict = self.criterion(outputs, targets)
        weight_dict = self.criterion.weight_dict
        losses = sum(
            loss_dict[k] * weight_dict[k] for k in loss_dict.keys() if k in weight_dict
        )
        loss_dict_unscaled = {f"val_{k}_unscaled": v for k, v in loss_dict.items()}
        loss_dict_scaled = {
            f"val_{k}": v * weight_dict[k]
            for k, v in loss_dict.items()
            if k in weight_dict
        }

        self.log(
            "total_val_loss",
            losses,
            on_epoch=True,
            prog_bar=True,
            batch_size=self.batch_size,
        )
        self.log_dict(
            loss_dict_unscaled,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.batch_size,
        )
        self.log_dict(
            loss_dict_scaled,
            on_epoch=True,
            prog_bar=False,
            batch_size=self.batch_size,
        )

        # Log masks with MLFlow logger
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(samples[0].cpu()),
            artifact_file=f"batch-{batch_idx}_val_image.png",
            run_id=self.logger.run_id,
        )
        # Log annotations with MLFlow logger
        union_annotations = torch.sum(targets[0]["masks"], dim=0).float()
        union_annotations = torch.where(union_annotations > 1, 1, union_annotations)
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(union_annotations.cpu().unsqueeze(0)),
            artifact_file=f"batch-{batch_idx}_val_union_annotations.png",
            run_id=self.logger.run_id,
        )

        orig_target_sizes = torch.stack([t["orig_size"] for t in targets], dim=0)
        results = self.postprocessors["bbox"](outputs, orig_target_sizes)
        if "segm" in self.postprocessors.keys():
            target_sizes = torch.stack([t["size"] for t in targets], dim=0)
            results = self.postprocessors["segm"](
                results, outputs, orig_target_sizes, target_sizes
            )

        # # Log mask with highest score
        # highest_score_idx = torch.argmax(outputs[0]["scores"])
        # self.logger.experiment.log_image(
        #     image=tensor_to_numpy_transform(
        #         outputs[0]["masks"][highest_score_idx].cpu()
        #     ),
        #     artifact_file=f"batch-{batch_idx}_val_mask.png",
        #     run_id=self.logger.run_id,
        # )
        # # Log union of masks
        # union_mask = torch.sum(outputs[0]["masks"], dim=0)
        # union_mask = torch.where(union_mask > 1, 1, union_mask)
        # self.logger.experiment.log_image(
        #     image=tensor_to_numpy_transform(union_mask.cpu()),
        #     artifact_file=f"batch-{batch_idx}_val_union_mask.png",
        #     run_id=self.logger.run_id,
        # )

        # map_dict = self.val_metrics(outputs, targets)
        # self.log(
        #     "mAP50_Val",
        #     map_dict["map_50"],
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=self.batch_size,
        # )
        # self.log(
        #     "mAP05-95_Val",
        #     map_dict["map"],
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=self.batch_size,
        # )

        return outputs, targets

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

        # Log masks with MLFlow logger
        self.logger.experiment.log_image(
            image=tensor_to_numpy_transform(x[0].cpu()),
            artifact_file=f"batch-{batch_idx}_val_image.png",
            run_id=self.logger.run_id,
        )
        # # Log mask with highest score
        # highest_score_idx = torch.argmax(preds[0]["scores"])
        # self.logger.experiment.log_image(
        #     image=tensor_to_numpy_transform(preds[0]["masks"][highest_score_idx].cpu()),
        #     artifact_file=f"batch-{batch_idx}_val_mask.png",
        #     run_id=self.logger.run_id,
        # )
        # # Log union of masks
        # union_mask = torch.sum(preds[0]["masks"], dim=0)
        # union_mask = torch.where(union_mask > 1, 1, union_mask)
        # self.logger.experiment.log_image(
        #     image=tensor_to_numpy_transform(union_mask.cpu()),
        #     artifact_file=f"batch-{batch_idx}_val_union_mask.png",
        #     run_id=self.logger.run_id,
        # )

        # # Squeeze preds masks
        # for pred in preds:
        #     pred["masks"] = torch.squeeze(
        #         pred["masks"] > LitMaskRCNN.MASK_THRESHOLD, dim=1
        #     ).to(torch.uint8)

        # map_dict = self.test_metrics(preds, y)
        # self.log(
        #     "mAP50_Test",
        #     map_dict["map_50"],
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=self.batch_size,
        # )
        # self.log(
        #     "mAP05-95_Test",
        #     map_dict["map"],
        #     prog_bar=True,
        #     on_step=False,
        #     on_epoch=True,
        #     batch_size=self.batch_size,
        # )

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
                    monitor="total_val_loss",
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
                    "monitor": "total_val_loss",
                },
            }
        else:
            return {"optimizer": optimizer}

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

    def _build_args(
        self,
        dataset_file,
        num_classes,
        num_queries,
        aux_loss,
        masks,
        frozen_weights,
        bbox_loss_coef,
        giou_loss_coef,
        mask_loss_coef,
        dice_loss_coef,
        dec_layers,
        eos_coef,
        lr_backbone,
        backbone,
        dilation,
        hidden_dim,
        dropout,
        nheads,
        dim_feedforward,
        enc_layers,
        pre_norm,
        set_cost_class,
        set_cost_bbox,
        set_cost_giou,
        position_embedding,
    ):
        args = {
            "dataset_file": dataset_file,
            "num_classes": num_classes,
            "device": torch.device("cuda" if torch.cuda.is_available() else "cpu"),
            "num_queries": num_queries,
            "aux_loss": aux_loss,
            "masks": masks,
            "frozen_weights": frozen_weights,
            "bbox_loss_coef": bbox_loss_coef,
            "giou_loss_coef": giou_loss_coef,
            "mask_loss_coef": mask_loss_coef,
            "dice_loss_coef": dice_loss_coef,
            "dec_layers": dec_layers,
            "eos_coef": eos_coef,
            "lr_backbone": lr_backbone,
            "backbone": backbone,
            "dilation": dilation,
            "hidden_dim": hidden_dim,
            "dropout": dropout,
            "nheads": nheads,
            "dim_feedforward": dim_feedforward,
            "enc_layers": enc_layers,
            "pre_norm": pre_norm,
            "set_cost_class": set_cost_class,
            "set_cost_bbox": set_cost_bbox,
            "set_cost_giou": set_cost_giou,
            "position_embedding": position_embedding,
        }
        args = type("Args", (object,), args)()
        return args
