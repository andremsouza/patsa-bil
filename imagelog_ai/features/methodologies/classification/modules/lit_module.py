from abc import abstractmethod
import os
from typing import Any, Final, Optional

from icecream import ic
from lightning.pytorch.callbacks import EarlyStopping, Callback
import torch
from torch import Tensor
from torch.nn import Parameter, Linear
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchvision import models
from torchmetrics import (
    Accuracy,
    Precision,
    Recall,
    F1Score,
    JaccardIndex,
    ConfusionMatrix,
)
from torchmetrics import MetricCollection

from imagelog_ai.base.base_lit_module import BaseLitModule


class ClassificationModule(BaseLitModule):
    def __init__(
        self,
        n_classes: int,
        task: str,
        trainable_stages: Optional[int] = None,
        early_stopping_patience: int = None,
        lr_scheduler_patience: int = None,
        learning_rate: float = 0.00001,
        weight_decay: float = 0,
    ) -> None:
        super().__init__(
            early_stopping_patience,
            lr_scheduler_patience,
            learning_rate,
            weight_decay,
        )
        self.n_classes: Final = n_classes
        self.task: Final = task
        self.trainable_stages: Final = trainable_stages

        torch.hub.set_dir(
            os.path.join(os.getcwd(), "data/PretrainedModels/Classification")
        )

        self.activation_func: Final = (
            torch.nn.Softmax(dim=1) if self.task == "multiclass" else torch.nn.Sigmoid()
        )
        self.loss_func: Final = (
            torch.nn.CrossEntropyLoss()
            if self.task == "multiclass"
            else torch.nn.BCEWithLogitsLoss()
        )

        self.train_metrics, self.val_metrics, self.test_metrics = self._build_metrics()
        self.network = self._build_network()

    @abstractmethod
    def _build_backbone(self):
        raise NotImplementedError

    @abstractmethod
    def _get_backbone_blocks(self):
        raise NotImplementedError

    def _build_pretrained_backbone(self, model_name: str) -> torch.nn.Module:
        return models.get_model(model_name, weights="DEFAULT")

    def _build_output_layer(self) -> torch.nn.Module:
        last_module = list(self.backbone.modules())[-1]
        classifier_in_features = last_module.out_features
        classifier_out_features = 1 if self.n_classes == 2 else self.n_classes

        return Linear(classifier_in_features, classifier_out_features)

    def _freeze_stages(self):
        blocks = self._get_backbone_blocks()
        n_stages = len(blocks)
        if self.trainable_stages <= n_stages:
            if self.trainable_stages < 1:
                for param in self.backbone.parameters():
                    param.requires_grad = False
            else:
                freeze = n_stages - self.trainable_stages
                for i in range(freeze):
                    block = blocks[i]
                    for layer in block:
                        if isinstance(layer, Parameter):
                            layer.requires_grad = False
                        else:
                            for param in layer.parameters():
                                param.requires_grad = False

    def _build_network(self) -> Any:
        self.backbone = self._build_backbone()
        if self.trainable_stages is not None:
            self._freeze_stages()

        self.output_layer = self._build_output_layer()

        return torch.nn.Sequential(self.backbone, self.output_layer)

    def _build_metrics(self) -> tuple:
        metrics_kwargs = {"task": self.task}
        if self.task == "multilabel":
            metrics_kwargs["num_labels"] = self.n_classes
        else:
            metrics_kwargs["num_classes"] = self.n_classes

        self.confusion_matrix = ConfusionMatrix(**metrics_kwargs, normalize="none")

        metrics_dict = {
            "Acc": Accuracy,
            "Pre": Precision,
            "Rec": Recall,
            "F1": F1Score,
            "Jac": JaccardIndex,
        }

        metric_collection = {}
        for metric_name, metric in metrics_dict.items():
            if metric == Accuracy:
                metrics_kwargs["average"] = "micro"
            else:
                metrics_kwargs["average"] = "macro"

            metric_collection[metric_name] = metric(**metrics_kwargs)
        metric_collection = MetricCollection(metric_collection)

        train_metrics = metric_collection.clone(postfix="/Train")
        val_metrics = metric_collection.clone(postfix="/Val")
        test_metrics = metric_collection.clone(postfix="/Test")

        return train_metrics, val_metrics, test_metrics

    def forward(self, input: Tensor) -> Any:
        return self.network(input)

    def shared_step(self, batch, prefix) -> Optional[Tensor]:
        x, y = batch

        if self.task == "binary":
            y = y.unsqueeze(1).type_as(x)

        y_pred = self(x)

        if self.task == "multilabel":
            loss = self.loss_func(y_pred, y.float())
        else:
            loss = self.loss_func(y_pred, y)

        self.log_dict({f"Loss/{prefix}": loss.detach()}, prog_bar=True)
        self._log_metrics(y_pred, y, prefix)

        if prefix == "Test":
            self.confusion_matrix.update(y_pred, y)

        return loss

    def predict_step(self, batch: Tensor, batch_idx: int) -> Tensor:
        return self.activation_func(self(batch))

    def on_test_epoch_end(self):
        self.test_confusion_matrix = (
            self.confusion_matrix.compute().detach().cpu().numpy()
        )

    def configure_callbacks(self) -> list[Callback]:
        """Function that is run by lightning when using the module.
                        Configures the callbacks used by the Model.

        Returns
        -------
        List[Callback]
                List of Callbacks
        """
        callback_list = []
        if self.hparams.early_stopping_patience != None:
            callback_list.append(
                EarlyStopping(
                    patience=self.hparams.early_stopping_patience,
                    stopping_threshold=1e-04,
                    monitor="Loss/Val",
                    mode="min",
                )
            )

        return callback_list

    def configure_optimizers(self) -> dict[str, Any]:
        """Function that is run by lightning when using the module.
                        Configures the optimizer used by the Model.

        Returns
        -------
        dict[str, Any]
                Dictionary of optimizer configuration
        """
        optimizer = Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        if self.hparams.lr_scheduler_patience != None:
            scheduler = ReduceLROnPlateau(
                patience=self.hparams.lr_scheduler_patience,
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
                    "monitor": "Loss/Val",
                },
            }
        else:
            return {"optimizer": optimizer}
