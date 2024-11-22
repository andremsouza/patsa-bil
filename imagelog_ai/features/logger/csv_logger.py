"""This module contains the MetricsCSVLogger class for saving metrics in a CSV file."""

import csv
from typing import Dict, List

import pandas as pd
from lightning.pytorch.loggers.logger import Logger
from lightning.pytorch.utilities import rank_zero_only


class MetricsCSVLogger(Logger):
    """Custom Lightning logger to save data in a CSV file."""

    def __init__(self, dst_path: str):
        super().__init__()

        self.dst_path = dst_path

        self.logged_data: List[Dict[str, float]] = []
        self.last_epoch = 0

        # Creating an experiment is not necessary, but the Trainer class access it,
        #  so the variable needs to exist
        self.experiment = None

    @property
    def name(self):
        """Return the name of the Logger."""
        return "MetricsCSVLogger"

    @property
    def version(self):
        """Return the experiment version, int or str."""
        return "1"

    @rank_zero_only
    def log_hyperparams(self, params):
        pass

    @rank_zero_only
    def log_metrics(self, metrics, step):
        """Function that collects the metrics."""
        # Saves log at the start of a new epoch
        if self.last_epoch != metrics["epoch"]:
            self.last_epoch = metrics["epoch"]
            self.save()

        # Adds the step to the log
        metrics["step"] = step
        self.logged_data.append(metrics)

    @rank_zero_only
    def save(self):
        """Save recorded metrics into a CSV file."""
        if not self.logged_data:
            return

        metrics_df = (
            pd.DataFrame(self.logged_data)
            .groupby(["epoch", "step"])
            .agg("mean")
            .reset_index()
        )
        self.logged_data = metrics_df.to_dict("records")

        with open(self.dst_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=self.logged_data[0].keys())
            writer.writeheader()
            writer.writerows(self.logged_data)

    @rank_zero_only
    def finalize(self, status):
        """Executes at the end of a run, saves collected metrics."""
        self.save()
