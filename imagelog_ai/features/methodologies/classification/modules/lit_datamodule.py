from typing import Optional, Callable

from imagelog_ai.base.base_lit_datamodule import LitBaseDataModule
from imagelog_ai.base.base_dataset import BaseDataset
from imagelog_ai.features.methodologies.classification.datasets.dataset import (
    ClassificationDataset,
)


class ClassificationDataModule(LitBaseDataModule):
    def __init__(
        self,
        project_name: str,
        preprocess_name: str,
        lists_datasource_names: dict[str, list[str]],
        class_list: list[str],
        task: str,
        others_class_list: Optional[list[str]] = None,
        augmentation: Optional[Callable] = None,
        validation_size: Optional[float] = None,
        batch_size: int = 1,
        num_workers: Optional[int] = None,
        persistent_workers: bool = True,
        drop_last: bool = False,
    ) -> None:
        super().__init__(
            project_name,
            preprocess_name,
            lists_datasource_names,
            validation_size,
            batch_size,
            num_workers,
            persistent_workers,
            drop_last,
        )

        self.class_list = sorted(class_list)
        self.task = task
        self.others_class_list = others_class_list
        self.augmentation = augmentation

    def _build_dataset(self, stage: str) -> BaseDataset:
        return ClassificationDataset(
            project_name=self.project_name,
            preprocess_name=self.preprocess_name,
            list_datasource_names=self.lists_datasource_names[stage],
            class_list=self.class_list,
            others_class_list=self.others_class_list,
            task=self.task,
            transform=self.augmentation if stage == "fit" else None,
        )
