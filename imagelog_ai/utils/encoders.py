import torch
from typing import Any, List, Optional


class LabelEncoder:
    def __init__(
        self,
        class_list: List[str],
        others_class_list: Optional[List[str]] = None,
        background: bool = False,
    ) -> None:
        self.class_list = sorted(class_list)
        self.others_class_list = others_class_list
        self.background = background

        self.possible_input_classes = self.class_list
        if self.others_class_list:
            self.possible_input_classes = set(self.possible_input_classes).union(
                set(self.others_class_list)
            )
            self.possible_input_classes = sorted(list(self.possible_input_classes))

        self._build_transformations()

    def _check_label(self, label: str):
        if label in self.possible_input_classes:
            return True
        else:
            raise ValueError(f"The label `{label}` does not exist.")

    def _build_transformations(self):

        classes = self.class_list
        if self.background:
            classes = ["BACKGROUND"] + classes

        self.class_to_idx = {
            label: label_idx for label_idx, label in enumerate(classes)
        }

        self.idx_to_class = self.class_list

        self.number_of_classes = len(classes)

        if self.others_class_list:
            other_int_encoder = len(classes)
            for other_class in self.others_class_list:
                self.class_to_idx[other_class] = other_int_encoder

            self.idx_to_class.append("OTHERS")
            self.number_of_classes += 1

    def __call__(self, label: str) -> int:
        if label in self.possible_input_classes:
            return self.class_to_idx[label]
        else:
            raise ValueError(f"The label `{label}` does not exist.")

    def decoder(self, label_idx: int) -> str:
        try:
            return self.idx_to_class[label_idx]
        except:
            raise ValueError(f"There is no label with an `{label_idx}` encoding.")


class OneHotEncoder:
    def __init__(
        self,
        class_list: List[str],
        others_class_list: Optional[List[str]] = None,
        background: bool = False,
    ) -> None:
        self.class_list = sorted(class_list)
        self.others_class_list = others_class_list
        self.background = background

        self.label_encoder = LabelEncoder(
            class_list=self.class_list,
            others_class_list=self.others_class_list,
            background=self.background,
        )

    def __call__(self, list_of_labels: list[str]) -> list[float]:
        encoded_list = self.label_encoder.number_of_classes * [0]
        for label in list_of_labels:
            encoded_list[self.label_encoder(label)] = 1
        return torch.Tensor(encoded_list).to(dtype=torch.int)

    def decoder(self, list_of_scores: torch.Tensor, threshold: float = 0.5):
        decoded_list = []
        for label_idx, score in enumerate(list_of_scores):
            if score >= threshold:
                decoded_list.append(self.label_encoder.decoder(label_idx))
