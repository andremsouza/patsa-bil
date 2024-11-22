from imagelog_ai.utils.io_functions import json_save
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


SMALL = 1e-15


class ConfusionMatrix:
    def __init__(
        self, cm: np.array, class_list: list[str], weighted_average: bool = False
    ) -> None:
        """
        Initialize the ConfusionMatrix with a confusion matrix numpy.

        Parameters
        ----------
        cm: numpy.ndarray
            Confusion matrix.

        class_list: list[str]
            List of classses names.

        weighted_average: bool = False
            Defines the reduction that is applied over labels.
            If True, calculates statistics for each label and computes weighted average using their support.
            If False, calculate statistics for each label and average them.
        """
        self.cm = cm
        self.class_list = class_list
        self.weighted_average = weighted_average
        self.metrics = self._calculate_metrics()

    def _smart_numpy_division(
        self, numerator: np.array, denominator: np.array
    ) -> np.array:
        return np.divide(
            numerator,
            denominator,
            out=np.zeros_like(numerator, dtype=float),
            where=denominator != 0,
        )

    def _calculate_metrics(self) -> dict:
        """
        Calculate accuracy, precision, recall, F1 score, and Jaccard index from the confusion matrix.
        """
        TP = np.diag(self.cm)
        FP = np.sum(self.cm, axis=0) - TP
        FN = np.sum(self.cm, axis=1) - TP
        TN = np.sum(self.cm) - (FP + FN + TP)

        accuracy = self._smart_numpy_division(TP + TN, TP + TN + FP + FN)
        precision = self._smart_numpy_division(TP, TP + FP)
        recall = self._smart_numpy_division(TP, TP + FN)
        f1_score = self._smart_numpy_division(
            2 * (precision * recall), precision + recall
        )
        jaccard = self._smart_numpy_division(TP, TP + FP + FN)

        metrics = {
            "Accuracy": {k: v for k, v in zip(self.class_list, accuracy.tolist())},
            "Precision": {k: v for k, v in zip(self.class_list, precision.tolist())},
            "Recall": {k: v for k, v in zip(self.class_list, recall.tolist())},
            "F1 Score": {k: v for k, v in zip(self.class_list, f1_score.tolist())},
            "Jaccard Index": {k: v for k, v in zip(self.class_list, jaccard.tolist())},
        }

        weights = np.sum(self.cm, axis=1) if self.weighted_average else None
        metrics["Accuracy"]["Global"] = np.sum(TP) / np.sum(self.cm)
        metrics["Precision"]["Global"] = np.average(precision, weights=weights)
        metrics["Recall"]["Global"] = np.average(recall, weights=weights)
        metrics["F1 Score"]["Global"] = np.average(f1_score, weights=weights)
        metrics["Jaccard Index"]["Global"] = np.average(jaccard, weights=weights)

        return metrics

    def save_metrics_to_json(self, dst_file_path: str) -> None:
        """
        Save the metrics dictionary to a JSON file.

        Parameters
        ----------
        filename (str): The name of the file where to save the metrics.
        """
        json_save(dict_to_save=self.metrics, dst_file_path=dst_file_path)

    def save(self, dst_file_path: str) -> None:
        """
        Save confusion matrix into a numpy array npy.

        Parameters
        ----------
        filename (str): The name of the file where to save the confusion matrix.
        """
        np.save(dst_file_path, self.cm)

    def plot(self, dst_file_path: str) -> None:
        """
        Plot confusion Matrix.

        Parameters
        ----------
        filename (str): The name of the file where to save the plot.
        """
        group_counts = ["{0:0.0f}\n".format(value) for value in self.cm.flatten()]
        cm_norm = 100 * self.cm.astype("float") / self.cm.sum(axis=1)[:, np.newaxis]
        group_percentages = [f"{value:.01f}%" for value in cm_norm.flatten()]
        box_labels = [
            f"{v1}{v2}".strip() for v1, v2 in zip(group_counts, group_percentages)
        ]
        box_labels = np.asarray(box_labels).reshape(cm_norm.shape[0], cm_norm.shape[1])
        figsize = plt.rcParams.get("figure.figsize")

        plt.figure(figsize=figsize)
        sns.heatmap(
            cm_norm,
            annot=box_labels,
            fmt="",
            cmap="Blues",
            cbar=True,
            xticklabels=self.class_list,
            yticklabels=self.class_list,
        )
        plt.ylabel("True label")
        plt.xlabel("Predicted label")
        plt.tight_layout()
        plt.savefig(dst_file_path)

    def plot_metrics(self, dst_file_path: str):
        """
        Plot confusion Matrix metrics.

        Parameters
        ----------
        dst_file_path (str): The name of the file where to save the plot.
        """

        df = pd.DataFrame(self.metrics)
        df.index.name = "class"
        df.reset_index(inplace=True)
        df = df.melt(id_vars="class")

        sns.set_theme()
        g = sns.catplot(
            data=df,
            x="class",
            y="value",
            hue="class",
            col="variable",
            kind="bar",
            errorbar="sd",
            height=6,
            # aspect=1.5
        )
        plt.ylim(0, 1.1)
        # iterate through axes
        for ax in g.axes.ravel():
            # add annotations
            for c in ax.containers:
                # add custom labels with the labels=labels parameter if needed
                labels = [f"{h:.02f}" if (h := v.get_height()) > 0 else "" for v in c]
                ax.bar_label(c, labels=labels, label_type="center")
            ax.margins(y=0.2)
        plt.savefig(dst_file_path, dpi=300)
