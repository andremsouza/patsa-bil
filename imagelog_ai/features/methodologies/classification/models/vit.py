from typing import Optional

from torch.nn import MaxPool2d

from imagelog_ai.features.methodologies.classification.modules.lit_module import (
    ClassificationModule,
)


class ViT(ClassificationModule):
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
        """Function that freezes the ViT stages using 'trainable_stages'"""
        blocks = [
            [
                self.backbone.class_token,
                self.backbone.conv_proj,
                self.backbone.encoder.pos_embedding,
            ]
        ]
        for child in self.backbone.encoder.layers.children():
            blocks.append([child])

        blocks.append([self.backbone.encoder.ln, self.backbone.heads])

        return blocks


class ViT_B_16(ViT):
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
        return self._build_pretrained_backbone(model_name="ViT_B_16")


class ViT_B_32(ViT):
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
        return self._build_pretrained_backbone(model_name="ViT_B_32")


class ViT_H_14(ViT):
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
        return self._build_pretrained_backbone(model_name="ViT_H_14")


class ViT_L_16(ViT):
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
        return self._build_pretrained_backbone(model_name="ViT_L_16")


class ViT_L_32(ViT):
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
        return self._build_pretrained_backbone(model_name="ViT_L_32")
