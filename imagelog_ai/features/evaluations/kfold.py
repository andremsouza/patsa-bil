"""This module contains the KFold class for performing the k-fold cross-validation."""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import kurtosis

from imagelog_ai.base.base_controller import BaseController
from imagelog_ai.utils.confusion_matrix import ConfusionMatrix
from imagelog_ai.utils.io_functions import json_load
from imagelog_ai.utils.list_functions import (
    flatten_list,
    k_fold_split,
    remove_from_list,
)
from imagelog_ai.utils.os_functions import sort_listdir


class KFold:
    def __init__(self, controller: BaseController, experiment_name: str) -> None:

        self.controller = controller
        self.experiment_name = experiment_name

        self.controller._build_experiment_configurations(
            experiment_name=self.experiment_name
        )
        self.controller._process_inputs()
        self.controller._reproducibility()

        configuration = self.controller.configuration["kfold"]

        self.n_folds = configuration["n_folds"]
        self.lists_datasource_names = configuration["lists_datasource_names"]
        self.val_filter = configuration["val_filter"]
        self.test_filter = configuration["test_filter"]

        self.path_results = os.path.join(
            "projects",
            self.controller.project_name,
            "results",
            self.controller.experiment_name or "",
        )

    def _posprocess_metrics_kfods(self):
        dfs = []
        folds = sort_listdir(self.path_results)
        for fold in folds:
            fold_path = os.path.join(self.path_results, fold)
            if os.path.isdir(fold_path):
                df = pd.read_json(
                    os.path.join(
                        self.path_results, fold, "confusion_matrix_metrics.json"
                    )
                )
                df.index.name = "class"
                df.reset_index(inplace=True)
                df = df.melt(id_vars="class")
                df.insert(0, "fold", fold)
                # df.dropna(axis=0, inplace=True)
                dfs.append(df)
        dfs = pd.concat(dfs)
        dfs.to_csv(os.path.join(self.path_results, "kfold_metrics.csv"), index=False)

        sns.set_theme()
        g = sns.catplot(
            data=dfs,
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
        plt.savefig(os.path.join(self.path_results, "kfold_metrics.png"), dpi=300)

        dfs_statistics = (
            dfs.drop(["fold"], axis=1)
            .groupby(["variable", "class"])
            .agg(["mean", "std", kurtosis])
        )
        dfs_statistics.to_csv(
            os.path.join(self.path_results, "kfold_metrics_stats.csv"), index=True
        )

    def _posprocess_datainfo_kfods(self):
        list_df = []
        folds = sort_listdir(self.path_results)
        for fold in folds:
            fold_path = os.path.join(self.path_results, fold)
            if os.path.isdir(fold_path):
                json_path = json_load(os.path.join(fold_path, "data.json"))
                df = pd.DataFrame(json_path)
                df.index = df.index.set_names(["data"])
                df.reset_index(inplace=True)
                df = df.melt(id_vars="data", var_name="stage", value_name="quantity")
                df.insert(0, "fold", fold)
                list_df.append(df)
        df = pd.concat(list_df)

        df.to_csv(os.path.join(self.path_results, "kfold_datainfo.csv"), index=False)

    def _build_aggregated_confusion_matrix(self):
        list_matrix = []
        folds = sort_listdir(self.path_results)
        for fold in folds:
            fold_path = os.path.join(self.path_results, fold)
            if os.path.isdir(fold_path):
                matrix_path = os.path.join(fold_path, "confusion_matrix.npy")
                confusion_matrix = np.load(matrix_path)

                list_matrix.append(confusion_matrix)
        list_matrix = np.array(list_matrix)

        summed_matrix = ConfusionMatrix(
            cm=np.sum(list_matrix, axis=0),
            class_list=self.controller.datamodule.class_list,
            weighted_average=False,
        )
        summed_matrix.save(
            dst_file_path=os.path.join(self.path_results, "agg_confusion_matrix.npy")
        )
        summed_matrix.save_metrics_to_json(
            dst_file_path=os.path.join(
                self.path_results, "agg_confusion_matrix_metrics.json"
            )
        )
        summed_matrix.plot(
            dst_file_path=os.path.join(self.path_results, "agg_confusion_matrix.png")
        )
        summed_matrix.plot_metrics(
            dst_file_path=os.path.join(
                self.path_results, "agg_confusion_matrix_metrics.png"
            )
        )

    def __call__(self) -> None:
        folds = k_fold_split(self.lists_datasource_names, n_folds=self.n_folds)

        kfold_count = 0
        lists_datasource_names = {}
        for test_folds in folds:
            if set(test_folds).isdisjoint(set(self.test_filter)):
                for val_folds in remove_from_list(folds, [test_folds]):
                    if set(val_folds).isdisjoint(set(self.val_filter)):
                        kfold_count += 1

                        train_folds = remove_from_list(folds, [test_folds, val_folds])

                        lists_datasource_names["fit"] = flatten_list(train_folds)
                        lists_datasource_names["val"] = flatten_list(val_folds)
                        lists_datasource_names["test"] = flatten_list(test_folds)

                        self.controller.datamodule_kwargs["lists_datasource_names"] = (
                            lists_datasource_names
                        )

                        self.controller._build_paths(fold_name=f"fold{kfold_count}")
                        self.controller.fit_test()

        self._posprocess_datainfo_kfods()
        self._posprocess_metrics_kfods()
        self._build_aggregated_confusion_matrix()
