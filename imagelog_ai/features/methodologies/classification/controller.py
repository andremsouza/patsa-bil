from imagelog_ai.base.base_controller import BaseController
from imagelog_ai.features.methodologies.classification.datasets.preprocess import (
    ClassificationImagesProcessor,
)
from imagelog_ai.features.methodologies.classification.modules.lit_datamodule import (
    ClassificationDataModule,
)
from imagelog_ai.base.base_lit_module import BaseLitModule
from imagelog_ai.features.methodologies.classification.utils.augmentation_dict import (
    create_augmentations,
)
import os
import numpy as np
from imagelog_ai.utils.confusion_matrix import ConfusionMatrix
from imagelog_ai.utils.list_functions import list_non_abstract_classes

from icecream import ic

NETWORK_CLASSES = list_non_abstract_classes(
    "imagelog_ai.features.methodologies.classification.models", BaseLitModule
)


class ClassificationController(BaseController):
    def __init__(
        self, project_name: str
    ) -> None:
        super().__init__(
            project_name,
            ClassificationImagesProcessor,
            ClassificationDataModule,
        )

    def _process_inputs(self) -> dict:

        self.datamodule_kwargs["task"] = self.module_kwargs["task"]

        n_classes = len(self.datamodule_kwargs["class_list"])
        if self.datamodule_kwargs["others_class_list"]:
            n_classes += 1
        self.module_kwargs["n_classes"] = n_classes

        self.module_class = NETWORK_CLASSES[self.module_kwargs.pop("network_class")]

        if self.module_kwargs.get("augmentation"):
            self.module_kwargs["augmentation"] = create_augmentations(
                self.module_kwargs["augmentation"]
            )

    def on_test_end(self) -> None:
        if self.model.task != "multilabel":
            class_list = self.datamodule.class_list
            if self.datamodule.others_class_list:
                class_list.append("others")

            self.confusion_matrix = ConfusionMatrix(
                cm=self.model.test_confusion_matrix,
                class_list=class_list,
                weighted_average=False,
            )

            self.confusion_matrix.save(
                dst_file_path=os.path.join(self.path_results, "confusion_matrix.npy")
            )
            self.confusion_matrix.save_metrics_to_json(
                dst_file_path=os.path.join(
                    self.path_results, "confusion_matrix_metrics.json"
                )
            )
            self.confusion_matrix.plot(
                dst_file_path=os.path.join(self.path_results, "confusion_matrix.png")
            )
            self.confusion_matrix.plot_metrics(
                dst_file_path=os.path.join(
                    self.path_results, "confusion_matrix_metrics.png"
                )
            )
