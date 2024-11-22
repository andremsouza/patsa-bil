import os
from typing import Any, Callable

import optuna

from imagelog_ai.base.base_controller import BaseController
from imagelog_ai.base.base_optuna import BaseOptuna
from imagelog_ai.utils.io_functions import json_load, tensor_load


class ModelOptunaOptimizer(BaseOptuna):
    def __init__(self, controller: BaseController, experiment_name: str):
        """Optuna class for preprocessing optimization.

        Parameters
        ----------
        controller: BaseController
            Controller for the project.
        """
        super().__init__(controller=controller, mode="experiment", experiment_name=experiment_name)
        self.trial_function_dictionary: dict = {}  # Initiated in _build_trial

    def objective_fun(self) -> Any:
        return self.controller.confusion_matrix.metrics["Jaccard Index"]["Global"]
