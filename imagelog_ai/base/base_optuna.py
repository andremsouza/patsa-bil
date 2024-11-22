"""Base class for implementing Optuna optimization."""

from abc import ABC, abstractmethod
from datetime import datetime
import os
from typing import Any, Callable, Optional

import lightning.pytorch as pl
import optuna
from optuna.integration import PyTorchLightningPruningCallback

from imagelog_ai.base.base_controller import BaseController
from imagelog_ai.utils.io_functions import json_load, json_save


# Optuna is operating with an old import of the Callback class, from pytorch_lightning
# An error occurs when sending it to the Trainer, which expects the Callback class from
# lightning.pytorch
# This hack fix adds the correct import from lightning.pytorch as a parent
# Hopefuly Optuna updates it so that this is no longer necessary
class OptunaPruningCallback(PyTorchLightningPruningCallback, pl.Callback):
    """Callback that allows Optuna to properly stop an active Lightning run when pruning."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class BaseOptuna(ABC):
    """Base class for implementing Optuna optimization.

    Args:
        controller (BaseController): The controller object.
        mode (str): The mode of optimization, either "preprocess" or "experiment".
        experiment_name (str, optional): The name of the experiment. Defaults to None.
    """

    def __init__(
        self,
        controller: BaseController,
        mode: str,
        experiment_preprocess_name: Optional[str] = None,
    ) -> None:
        """Base class for implementing Optuna optimization.

        Args:
            controller (BaseController): The controller object.
            mode (str): The mode of optimization, either "preprocess" or "experiment".
            experiment_name (str, optional): The name of the experiment. Defaults to None.
        """
        super().__init__()
        self.controller: BaseController = controller
        self.mode: str = mode
        self.experiment_name: Optional[str] = None
        self.preprocess_name: Optional[str] = None
        if self.mode == "experiment":
            self.experiment_name = experiment_preprocess_name
        elif self.mode == "preprocess":
            self.preprocess_name = experiment_preprocess_name
        else:
            raise ValueError(f"Mode {self.mode} not recognized")
        self.optuna_kwargs: dict = {}

        self.trial_idx = 0
        self._build_optuna_configurations()
        self._build_study()

    @abstractmethod
    def objective_fun(self) -> Any:
        """Abstract method to define the objective function for optimization.

        Should be implemented in the child class. The objective function should return a value to
        be optimized. The result value is computed after running preprocessing or experiment.

        Returns:
            Any: The result of the objective function.
        """
        raise NotImplementedError

    def _objective(self, trial: optuna.trial.Trial):
        """Helper function for trial. Builds current trial and calls the objective function.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object.

        Returns:
            Any: The result of the objective function.
        """
        self.trial_idx += 1
        self._build_trial(trial=trial)

        return self.objective_fun()

    def _build_trial_preprocess_configurations(
        self, trial_function_dictionary: dict[str, Callable]
    ):
        """Build preprocess configurations for an Optuna trial.

        Args:
            trial_function_dictionary (dict[str, Callable]): Dictionary mapping trial parameters
                to their suggested values.
        """
        # Load controller preprocess configurations
        self.controller._load_preprocess_configurations(
            preprocess_name=self.preprocess_name
        )  # pylint: disable=protected-access
        # Modify preprocess configurations for optuna trial
        new_configuration = self.controller.preprocess_kwargs["input_transform"].copy()
        for configuration_name, configuration_values in new_configuration.items():
            for key, value in configuration_values.items():
                if isinstance(value, list):
                    if "optuna" in value:
                        new_configuration[configuration_name][key] = (
                            trial_function_dictionary[value[1]](key, **value[2])
                        )
        # Build new configuration
        self.controller.preprocess_kwargs["input_transform"] = new_configuration
        self.controller.setup_preprocess(  # pylint: disable=protected-access
            preprocess_name=self.preprocess_name,
            load_configuration=False,
        )
        self.controller.preprocess(override_preprocess=True)

    def _build_trial_experiment_configurations(
        self, trial: optuna.trial.Trial, trial_function_dictionary: dict[str, Callable]
    ):
        """
        Build experiment configurations for an Optuna trial.

        Args:
            trial_function_dictionary (dict[str, Callable]): Dictionary mapping trial parameters
                to their suggested values.
        """
        self.controller._load_experiment_configurations(  # pylint: disable=protected-access
            experiment_name=self.experiment_name
        )

        for (
            configuration_name,
            configuration_values,
        ) in self.controller.configuration.items():
            for key, value in configuration_values.items():
                if isinstance(value, list):
                    if "optuna" in value:
                        self.controller.configuration[configuration_name][key] = (
                            trial_function_dictionary[value[1]](key, **value[2])
                        )

        self.controller._build_experiment_configurations(  # pylint: disable=protected-access
            experiment_name=self.experiment_name, load_configuration=False
        )
        self.controller._process_inputs()  # pylint: disable=protected-access
        self.controller._reproducibility()  # pylint: disable=protected-access
        # fold_name=f"trial{self.trial_idx}"
        self.controller._build_paths()  # pylint: disable=protected-access
        self.controller.fit_test(
            override_experiment=True,
            # list_external_loggers=[
            #     OptunaPruningCallback(monitor="Jac/Val", trial=trial)
            # ],
        )

    def _build_trial(self, trial: optuna.trial.Trial):
        """
        Build configurations for an Optuna trial based on the mode.

        Args:
            trial (optuna.trial.Trial): The Optuna trial object.
        """
        trial_function_dictionary = {
            "discrete_uniform": trial.suggest_discrete_uniform,
            "categorical": trial.suggest_categorical,
            "loguniform": trial.suggest_loguniform,
            "uniform": trial.suggest_uniform,
            "float": trial.suggest_float,
            "int": trial.suggest_int,
        }

        if self.mode == "preprocess":
            self._build_trial_preprocess_configurations(
                trial_function_dictionary=trial_function_dictionary
            )
        elif self.mode == "experiment":
            self._build_trial_experiment_configurations(
                trial=trial,
                trial_function_dictionary=trial_function_dictionary,
            )
        else:
            raise ValueError(f"Mode {self.mode} not recognized")

    def _build_optuna_configurations(self):
        """Build Optuna configurations based on the selected mode."""
        # Load controller configurations
        if self.mode == "preprocess":
            json_path = os.path.join(
                "projects",
                self.controller.project_name,
                "preprocesses",
                f"{self.preprocess_name}.json",
            )

        elif self.mode == "experiment":
            json_path = os.path.join(
                "projects",
                self.controller.project_name,
                "experiments",
                f"{self.experiment_name}.json",
            )

        else:
            raise ValueError(f"Mode {self.mode} not recognized")

        self.optuna_kwargs = json_load(json_path)["optuna"]

    def _build_study(self):
        """Build the Optuna study object."""
        self.study = optuna.create_study(
            storage=self.optuna_kwargs["create_study"].pop("storage", None),
            sampler=self.optuna_kwargs["create_study"].pop("sampler", None),
            pruner=optuna.pruners.PatientPruner(
                optuna.pruners.MedianPruner(n_min_trials=3), patience=3
            ),
            study_name=self.optuna_kwargs["create_study"].pop(
                "study_name",
                f"{self.controller.project_name}"
                f"_{self.mode}"
                f"_{datetime.now().strftime('%Y%m%d%H%M%S')}",
            ),
            direction=self.optuna_kwargs["create_study"].pop("direction", None),
            load_if_exists=self.optuna_kwargs["create_study"].pop(
                "load_if_exists", False
            ),
            directions=self.optuna_kwargs["create_study"].pop("directions", None),
        )

    def __call__(self) -> Any:
        """Run the Optuna optimization.

        Returns:
            Any: The result of the optimization.
        """
        self.study.optimize(
            self._objective,
            callbacks=[
                optuna.study.MaxTrialsCallback(
                    n_trials=self.optuna_kwargs["optimize"]["n_trials"], states=[1]
                )
            ],
            n_trials=int(1.5 * self.optuna_kwargs["optimize"].pop("n_trials")),
            **self.optuna_kwargs["optimize"],
        )

        results = {
            "fun_opt": self.study.best_trial.values,
            "x": self.study.best_trial.params,
        }
        json_save(
            results,
            os.path.join(self.controller.path_results, "optimization_results.json"),
        )
