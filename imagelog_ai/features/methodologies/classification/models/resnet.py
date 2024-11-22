from typing import Optional

from torch.nn import Sequential

from imagelog_ai.features.methodologies.classification.modules.lit_module import (
    ClassificationModule,
)


class ResNet(ClassificationModule):
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
        """Function that freezes the ResNet stages using 'trainable_stages'"""
        blocks = [
            [
                self.backbone.conv1,
                self.backbone.bn1,
                self.backbone.relu,
                self.backbone.maxpool,
            ]
        ]
        for child in self.backbone.children():
            if isinstance(child, Sequential):
                for block in child:
                    blocks.append([block])

        blocks.append([self.backbone.avgpool, self.backbone.fc])

        return blocks


class ResNet18(ResNet):
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
        return self._build_pretrained_backbone(model_name="RESNET18")


class ResNet34(ResNet):
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
        return self._build_pretrained_backbone(model_name="RESNET34")


class ResNet50(ResNet):
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
        return self._build_pretrained_backbone(model_name="RESNET50")


class ResNet101(ResNet):
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
        return self._build_pretrained_backbone(model_name="RESNET101")


class ResNet152(ResNet):
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
        return self._build_pretrained_backbone(model_name="RESNET152")
