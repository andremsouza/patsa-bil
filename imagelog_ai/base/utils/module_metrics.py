"""This module contains a metrics dictionary to be be used in the training/evaluation of models."""

from typing import Type, Dict

from torchmetrics import JaccardIndex, Accuracy, Precision, Recall, F1Score, AUROC
from torchmetrics.detection.mean_ap import MeanAveragePrecision

metric_dict: Dict[str, Type] = {
    "MeanAveragePrecision": MeanAveragePrecision,
    "JaccardIndex": JaccardIndex,
    "Accuracy": Accuracy,
    "Precision": Precision,
    "Recall": Recall,
    "F1Score": F1Score,
    "AUROC": AUROC,
}
