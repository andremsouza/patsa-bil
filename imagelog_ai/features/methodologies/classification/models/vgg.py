from typing import Optional

from torch.nn import MaxPool2d

from imagelog_ai.features.methodologies.classification.modules.lit_module import (
    ClassificationModule,
)


class VGG(ClassificationModule):
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
            n_classes,
            task,
            trainable_stages,
            early_stopping_patience,
            lr_scheduler_patience,
            learning_rate,
            weight_decay,
        )

    def _get_backbone_blocks(self) -> list:
        """Function that freezes the VGG stages using 'trainable_stages'"""
        blocks, block = [], []

        for layer in self.backbone.features.children():
            block.append(layer)
            if isinstance(layer, MaxPool2d):
                blocks.append(block)
                block = []

        blocks.append([self.backbone.avgpool, self.backbone.classifier])

        return blocks


class VGG11(VGG):
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
            n_classes,
            task,
            trainable_stages,
            early_stopping_patience,
            lr_scheduler_patience,
            learning_rate,
            weight_decay,
        )

    def _build_backbone(self):
        return self._build_pretrained_backbone(model_name="VGG11")


class VGG16(VGG):
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
            n_classes,
            task,
            trainable_stages,
            early_stopping_patience,
            lr_scheduler_patience,
            learning_rate,
            weight_decay,
        )

    def _build_backbone(self):
        return self._build_pretrained_backbone(model_name="VGG16")


class VGG19(VGG):
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
            n_classes,
            task,
            trainable_stages,
            early_stopping_patience,
            lr_scheduler_patience,
            learning_rate,
            weight_decay,
        )

    def _build_backbone(self):
        return self._build_pretrained_backbone(model_name="VGG19")
