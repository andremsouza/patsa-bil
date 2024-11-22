"""Base Controller Module.

This module defines the `BaseController` class, which serves as the base controller for imagelog
experiments and preprocess. It provides methods for setting up preprocessing,
running experiments, fitting models, testing models, and making predictions.
"""

from abc import ABC, abstractmethod
import os
from pathlib import Path
import sys
from typing import Optional

from lightning.pytorch.loggers import Logger
from lightning.pytorch.loggers.wandb import WandbLogger
from lightning import Trainer, seed_everything
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import torch
from torchvision.transforms import Compose
from torchvision.transforms import *

from imagelog_ai.base.base_lit_datamodule import LitBaseDataModule
from imagelog_ai.base.base_lit_module import BaseLitModule
from imagelog_ai.base.base_preprocess import BaseImagesProcessor
from imagelog_ai.features.logger.csv_logger import MetricsCSVLogger

from imagelog_ai.data.data_preprocessing.transforms import *
from imagelog_ai.utils.io_functions import json_load, json_save
from imagelog_ai.utils.os_functions import instantiate_obj_from_str
from imagelog_ai.utils.argcheck import experiment_was_run


class BaseController(ABC):
    """Base Controller Class.

    This class serves as the base controller for imagelog experiments and preprocess.
    It provides methods for setting up preprocessing, running experiments, fitting models,
    testing models, and making predictions.

    Attributes:
        project_name (str): The name of the project.
        preprocessor_class (BaseImagesProcessor): The preprocessor class.
        datamodule_class (LitBaseDataModule): The datamodule class.
        preprocessor : Optional[BaseImagesProcessor]: The preprocessor instance.
        datamodule : Optional[LitBaseDataModule]: The datamodule instance.
        model : Optional[BaseLitModule]: The model instance.
        trainer : Optional[Trainer]: The trainer instance.
        experiment_name : Optional[str]: The name of the experiment.
        preprocess_name : Optional[str]: The name of the preprocess.
        path_ckpt : Optional[str]: The path to the model checkpoint.
        path_datainfo : Optional[str]: The path to the data information.
        path_log : Optional[str]: The path to the logs.
        path_plot : Optional[str]: The path to the plots.
        path_results : Optional[str]: The path to the results.
        random_seed : Optional[int]: The random seed.
        configuration (dict): The experiment configurations.
        preprocess_kwargs (dict): The preprocess configurations.
        datamodule_kwargs (dict): The datamodule configurations.
        module_kwargs (dict): The model configurations.
        trainer_kwargs (dict): The trainer configurations.
        kfold_kwargs (dict): The k-fold configurations.
        skip_csv_logger : Optional[bool]: Whether to skip the CSV logger.
        skip_wandb : Optional[bool]: Whether to skip Weights & Biases.
        project_load : Optional[bool]: Whether to load the project.
    """

    def __init__(
        self,
        project_name: str,
        preprocessor_class: BaseImagesProcessor,
        datamodule_class: LitBaseDataModule,
    ) -> None:
        """Initialize the BaseController.

        Args:
            project_name (str): The name of the project.
            preprocessor_class (BaseImagesProcessor): The preprocessor class.
            datamodule_class (LitBaseDataModule): The datamodule class.
        """
        self.project_name: str = project_name
        self.preprocessor_class = preprocessor_class
        self.datamodule_class: LitBaseDataModule = datamodule_class
        self.preprocessor: Optional[BaseImagesProcessor] = None
        self.datamodule: Optional[LitBaseDataModule] = None
        self.model: Optional[BaseLitModule] = None
        self.trainer: Optional[Trainer] = None
        self.experiment_name: Optional[str] = None
        self.preprocess_name: Optional[str] = None
        self.path_ckpt: Optional[str] = None
        self.path_datainfo: Optional[str] = None
        self.path_log: Optional[str] = None
        self.path_plot: Optional[str] = None
        self.path_results: Optional[str] = None
        self.random_seed: Optional[int] = None
        self.configuration: dict = {}
        self.preprocess_kwargs: dict = {}
        self.datamodule_kwargs: dict = {}
        self.module_kwargs: dict = {}
        self.trainer_kwargs: dict = {}
        self.kfold_kwargs: dict = {}
        self.skip_csv_logger: Optional[bool] = None
        self.skip_wandb: Optional[bool] = None
        self.project_load: Optional[bool] = None

        torch.set_float32_matmul_precision("medium")

    def plot_metrics(self) -> None:
        """Plot the metrics.

        This method reads the fit metrics from a CSV file, plots the training and validation
        metrics, and saves the plots as images.
        """
        if self.path_log is None or self.path_plot is None:
            raise ValueError("path_log or path_plot is not set.")
        csv_path: str = os.path.join(self.path_log, "fit_metrics.csv")
        df: pd.DataFrame = pd.read_csv(csv_path, sep=",")
        df = df.rename(columns={"Loss/Train": "Loss/Train_epoch"})

        metric_names: list[str] = list(df.columns.values)
        metric_names.remove("epoch")
        metric_names.remove("step")
        metric_type_list = set()

        for metric in metric_names:
            if not metric.endswith("_step"):
                metric_type = metric[: metric.find("/")]
                metric_type_list.add(metric_type)

        for metric_type in metric_type_list:
            train_metric = f"{metric_type}/Train_epoch"
            val_metric = f"{metric_type}/Val"

            epoch_metrics_positions = ~df[val_metric].isna()

            d = {
                "epoch": df[epoch_metrics_positions]["epoch"],
                "train": df[epoch_metrics_positions][train_metric],
                "val": df[epoch_metrics_positions][val_metric],
            }
            pdnumsqr = pd.DataFrame(d)

            sns.set_theme(style="darkgrid")
            # Create a line chart
            plt.figure(figsize=(12, 12))
            sns.lineplot(x="epoch", y="train", data=pdnumsqr, label="Train")
            sns.lineplot(x="epoch", y="val", data=pdnumsqr, label="Validation")

            plt.xlabel("Epoch")
            plt.ylabel(metric_type)

            if metric_type == "Loss":
                plt.yscale("log")
                image_path = os.path.join(self.path_plot, f"{metric_type}_Log.png")
                plt.savefig(image_path, bbox_inches="tight")
                plt.yscale("linear")
            else:
                plt.ylim(0, 1)

            image_path = os.path.join(self.path_plot, f"{metric_type}.png")
            plt.savefig(image_path, bbox_inches="tight")

            plt.close("all")

    def setup_preprocess(
        self, preprocess_name: str, load_configuration: bool = True
    ) -> None:
        """Setup the preprocessing.

        This method sets up the preprocessing by loading the preprocess configurations
        and ensuring reproducibility.

        Args:
            preprocess_name (str): The name of the preprocess.
            load_configuration (bool, optional): Whether to load the preprocess configurations.
                Defaults to True.
        """
        self._build_preprocess_configurations(
            preprocess_name=preprocess_name,
            load_configuration=load_configuration,
        )
        self._reproducibility()

    def preprocess(self, override_preprocess: bool) -> None:
        """Preprocess the data.

        This method performs the data preprocessing using the specified preprocessor
        and preprocess configurations.

        Args:
            override_preprocess (bool): Whether to override the previous preprocessing results.
        """
        self.preprocess_kwargs["override"] = override_preprocess
        self.preprocessor = self._build_preprocessor()

        self.preprocessor(**self.preprocess_kwargs)

    def setup_experiment(
        self, experiment_name: str, fold_name: Optional[str] = None
    ) -> None:
        """Setup the experiment.

        This method sets up the experiment by building the experiment configurations,
        processing the inputs, building the paths, and ensuring reproducibility.

        Args:
            experiment_name (str): The name of the experiment.
            fold_name (str, optional): The name of the fold. Defaults to None.
        """
        self._build_experiment_configurations(experiment_name=experiment_name)
        self._process_inputs()
        self._build_paths(fold_name=fold_name)
        self._reproducibility()

    def fit(
        self,
        override_experiment: bool = False,
        list_external_loggers: Optional[list[Logger]] = None,
    ) -> None:
        """Fit the model.

        This method fits the model using the specified experiment configurations
        and external loggers.

        Args:
            override_experiment (bool, optional): Whether to override the previous
            experiment results.
                Defaults to False.
            list_external_loggers (list[Logger], optional): List of external loggers.
                Defaults to None.
        """
        experiment_was_run(
            project_name=self.project_name,
            experiment_name=(
                self.experiment_name if self.experiment_name is not None else ""
            ),
            result_path="model/model.ckpt",
            override_experiment=override_experiment,
        )

        self.datamodule = self._build_datamodule()
        self.model = self._build_module(project_load=bool(self.project_load))

        self.trainer = self._build_trainer(
            stage="fit", list_external_loggers=list_external_loggers
        )
        self.trainer.fit(model=self.model, datamodule=self.datamodule)
        if self.path_ckpt is not None:
            self.trainer.save_checkpoint(filepath=self.path_ckpt)
        else:
            raise ValueError("path_ckpt is not set.")
        self.plot_metrics()

        if self.path_datainfo is not None:
            json_save(self.datamodule.dataloader_info, self.path_datainfo)
        else:
            raise ValueError("path_datainfo is not set.")

    def test(self, list_external_loggers: Optional[list[Logger]] = None) -> None:
        """Test the model.

        This method tests the model using the specified external loggers.

        Args:
            list_external_loggers (list[Logger], optional): List of external loggers.
                Defaults to None.
        """
        self.datamodule = self._build_datamodule()
        self.model = self._build_module(project_load=True)

        self.trainer = self._build_trainer(
            stage="test", list_external_loggers=list_external_loggers
        )
        self.trainer.test(model=self.model, datamodule=self.datamodule)

        dataloader_info = json_load(self.path_datainfo)
        dataloader_info["test"] = self.datamodule.dataloader_info["test"]
        json_save(dataloader_info, self.path_datainfo)
        self.on_test_end()

    def fit_test(
        self,
        override_experiment: bool = False,
        list_external_loggers: Optional[list[Logger]] = None,
    ) -> None:
        """Fit and test the model.

        This method fits the model and then tests it using the specified experiment
        configurations and external loggers.

        Args:
            override_experiment (bool, optional): Whether to override the previous
            experiment results.
                Defaults to False.
            list_external_loggers (list[Logger], optional): List of external loggers.
                Defaults to None.
        """
        experiment_was_run(
            project_name=self.project_name,
            experiment_name=(
                self.experiment_name if self.experiment_name is not None else ""
            ),
            result_path="model/model.ckpt",
            override_experiment=override_experiment,
        )

        self.datamodule = self._build_datamodule()
        self.model = self._build_module(project_load=bool(self.project_load))

        self.trainer = self._build_trainer(
            stage="fit", list_external_loggers=list_external_loggers
        )
        self.trainer.fit(model=self.model, datamodule=self.datamodule)
        self.trainer.save_checkpoint(filepath=self.path_ckpt)
        self.plot_metrics()

        self.trainer = self._build_trainer(stage="test")
        self.trainer.test(model=self.model, datamodule=self.datamodule)
        json_save(self.datamodule.dataloader_info, self.path_datainfo)
        self.on_test_end()

    def on_test_end(self) -> None:
        """Perform actions after testing.

        This method can be overridden to perform additional actions after testing the model.
        """
        pass

    def predict(self) -> None:
        """Make predictions.

        This method makes predictions using the trained model and saves the predictions
        as a JSON file.
        """
        self.datamodule = self._build_datamodule()
        self.model = self._build_module(project_load=True)

        self.trainer = self._build_trainer(stage="predict")
        prediction = self.trainer.predict(model=self.model, datamodule=self.datamodule)
        prediction = torch.vstack(prediction)

        encoder = self.datamodule.dataset_predict.encoder

        prediction = {
            "decoder": encoder.class_to_idx,
            "scores": prediction.tolist(),
            "prediction": [
                encoder.decoder(x) for x in prediction.argmax(dim=1).tolist()
            ],
        }

        json_save(prediction, os.path.join(self.path_results, "prediction.json"))

    def _load_preprocess_configurations(self, preprocess_name: str) -> None:
        """
        Load the preprocess configurations from a JSON file.

        Args:
            preprocess_name (str): The name of the preprocess.

        Returns:
            None
        """
        self.preprocess_name = preprocess_name
        self.preprocess_kwargs = json_load(
            os.path.join(
                "projects",
                self.project_name,
                "preprocesses",
                f"{self.preprocess_name}.json",
            )
        )["configurations"]
        self.preprocess_kwargs["project_name"] = self.project_name
        self.preprocess_kwargs["preprocess_name"] = self.preprocess_name

    def _build_preprocess_configurations(
        self, preprocess_name: str, load_configuration: bool = True
    ) -> None:
        """
        Build the preprocess configurations.

        Args:
            preprocess_name (str): The name of the preprocess.
            load_configuration (bool, optional): Whether to load the preprocess configurations.
                Defaults to True.

        Returns:
            None
        """
        if load_configuration:
            self._load_preprocess_configurations(preprocess_name=preprocess_name)

        if self.preprocess_kwargs["input_transform"]:
            if self.preprocess_kwargs["input_transform"] != "hard_copy":
                try:
                    module = sys.modules[__name__]
                    input_transform = [
                        instantiate_obj_from_str(cls_name, cls_kwargs, module=module)
                        for cls_name, cls_kwargs in self.preprocess_kwargs[
                            "input_transform"
                        ].items()
                    ]
                    self.preprocess_kwargs["input_transform"] = Compose(input_transform)
                except Exception as exc:
                    raise TypeError(
                        f"input_transform: {self.preprocess_kwargs['input_transform']} "
                        "is not valid."
                    ) from exc

    def _build_preprocessor(self) -> BaseImagesProcessor:
        """
        Builds and returns an instance of the preprocessor class.

        Returns:
            An instance of the preprocessor class.

        Raises:
            KeyError: If any required keyword argument is missing.
        """
        return self.preprocessor_class(
            project_name=self.preprocess_kwargs.pop("project_name"),
            preprocess_name=self.preprocess_kwargs.pop("preprocess_name"),
            image_format=self.preprocess_kwargs.pop("image_format"),
            input_pil_mode=self.preprocess_kwargs.pop("input_pil_mode"),
            override=self.preprocess_kwargs.pop("override"),
        )

    def _load_experiment_configurations(self, experiment_name: str) -> None:
        """
        Load experiment configurations from a JSON file.

        Args:
            experiment_name (str): The name of the experiment.

        Returns:
            None
        """
        self.experiment_name = experiment_name

        self.configuration = json_load(
            os.path.join(
                "projects",
                self.project_name,
                "experiments",
                f"{self.experiment_name}.json",
            )
        )

        self.preprocess_name = self.configuration["preprocess"]
        self.datamodule_kwargst = self.configuration["datamodule"]
        self.module_kwargs = self.configuration["model"]
        self.trainer_kwargs = self.configuration["trainer"]
        self.kfold_kwargs = self.configuration["kfold"]

    def _build_experiment_configurations(
        self, experiment_name: str, load_configuration: bool = True
    ) -> None:
        """
        Builds the experiment configurations.

        Args:
            experiment_name (str): The name of the experiment.
            load_configuration (bool, optional): Whether to load the configurations.
                Defaults to True.
        """
        if load_configuration:
            self._load_experiment_configurations(experiment_name=experiment_name)
        self.skip_csv_logger = self.trainer_kwargs.pop("skip_csv_logger")
        self.skip_wandb = self.trainer_kwargs.pop("skip_wandb")
        self.random_seed = self.trainer_kwargs.pop("random_seed")
        self.project_load = self.module_kwargs.pop("project_load")

    def _reproducibility(self) -> None:
        """
        Set the random seed for reproducibility.

        If `random_seed` is not None, the method uses the `seed_everything` function
        to set the random seed.
        """
        if self.random_seed is not None:
            seed_everything(self.random_seed, workers=True)

    def _build_paths(self, fold_name: Optional[str] = None) -> None:
        """
        Builds the necessary paths for storing results, models, logs, plots, and data
        information.

        Args:
            fold_name : Optional[str]: The name of the fold. Defaults to None.

        Returns:
            None
        """

        if fold_name:
            self.path_results = os.path.join(
                "projects",
                self.project_name,
                "results",
                self.experiment_name,
                fold_name,
            )
        else:
            self.path_results = os.path.join(
                "projects", self.project_name, "results", self.experiment_name
            )
        Path(self.path_results).mkdir(parents=True, exist_ok=True)

        model_path: str = os.path.join(self.path_results, "model")
        Path(model_path).mkdir(parents=True, exist_ok=True)
        self.path_ckpt = os.path.join(model_path, "model.ckpt")

        self.path_log = os.path.join(self.path_results, "logs")
        Path(self.path_log).mkdir(parents=True, exist_ok=True)

        self.path_plot = os.path.join(self.path_results, "plots")
        Path(self.path_plot).mkdir(parents=True, exist_ok=True)

        self.path_datainfo = os.path.join(self.path_results, "data.json")

    @abstractmethod
    def _process_inputs(self) -> None:
        """
        Process the inputs for the controller.

        This method is responsible for processing the inputs required for the controller's logic.
        It should be implemented by the derived classes.

        Raises:
            NotImplementedError: If the method is not implemented by the derived class.
        """
        raise NotImplementedError

    def _build_datamodule(self) -> LitBaseDataModule:
        """
        Builds and returns an instance of the LitBaseDataModule class.

        Returns:
            LitBaseDataModule: An instance of the LitBaseDataModule class.
        """
        return self.datamodule_class(
            project_name=self.project_name,
            preprocess_name=self.preprocess_name,
            **self.datamodule_kwargs,
        )

    def _build_module(self, project_load: bool) -> BaseLitModule:
        """
        Builds and returns an instance of the BaseLitModule class.

        Args:
            project_load (bool): A flag indicating whether to load the module from a checkpoint.

        Returns:
            BaseLitModule: An instance of the BaseLitModule class.

        """
        if project_load:
            return self.module_class.load_from_checkpoint(self.path_ckpt)
        else:
            return self.module_class(**self.module_kwargs)

    def _build_loggers(
        self, stage: str, list_external_loggers: Optional[list[Logger]] = None
    ) -> list:
        """
        Builds a list of loggers for the specified stage.

        Args:
            stage (str): The stage for which the loggers are being built.
            list_external_loggers (list[Logger] | None, optional): A list of external loggers
            to include.
                Defaults to None.

        Returns:
            list: A list of loggers for the specified stage, including any external loggers.

        """
        loggers: list = []

        if not self.skip_csv_logger:
            csv_logger = MetricsCSVLogger(
                dst_path=os.path.join(self.path_log, f"{stage}_metrics.csv")
            )
            loggers.append(csv_logger)

        if not self.skip_wandb:
            logger = WandbLogger(
                name=self.experiment_name,
                project=self.project_name,
                save_dir=self.path_log,
            )
            loggers.append(logger)

        if list_external_loggers:
            loggers.extend(list_external_loggers)

        return loggers if loggers else None

    def _build_trainer(
        self, stage: str, list_external_loggers: Optional[list[Logger]] = None
    ) -> Trainer:
        """
        Builds and returns a Trainer object.

        Args:
            stage (str): The stage of the training process.
            list_external_loggers (list[Logger] | None, optional): A list of external loggers.
                Defaults to None.

        Returns:
            Trainer: The built Trainer object.
        """
        loggers = (
            self._build_loggers(
                stage=stage, list_external_loggers=list_external_loggers
            )
            if stage != "predict"
            else None
        )
        return Trainer(logger=loggers, **self.trainer_kwargs)
