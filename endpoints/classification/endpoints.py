"""Endpoints for classification tasks."""

from imagelog_ai.features.evaluations.kfold import KFold
from imagelog_ai.features.methodologies.classification.controller import (
    ClassificationController,
)


def _build_controller(project_name: str):
    """
    Build a ClassificationController object.

    Args:
        project_name (str): The name of the project.

    Returns:
        ClassificationController: The ClassificationController object.
    """
    return ClassificationController(
        project_name=project_name,
    )


def preprocess(project_name: str, preprocess_name: str, override_preprocess: bool):
    """
    Perform preprocessing for the given project.

    Args:
        project_name (str): The name of the project.
        preprocess_name (str): The name of the preprocessing method.
        override_preprocess (bool): Whether to override existing preprocessing.

    Returns:
        None
    """
    controller = _build_controller(project_name=project_name)
    controller.setup_preprocess(preprocess_name=preprocess_name)
    controller.preprocess(override_preprocess=override_preprocess)


def fit(project_name: str, experiment_name: str, override_experiment: bool):
    """
    Fit the model for the given project.

    Args:
        project_name (str): The name of the project.
        experiment_name (str): The name of the experiment.
        override_experiment (bool): Whether to override existing experiment.

    Returns:
        None
    """
    controller = _build_controller(project_name=project_name)
    controller.setup_experiment(experiment_name=experiment_name)
    controller.fit(override_experiment=override_experiment)


def test(project_name: str, experiment_name: str, override_experiment: bool):
    """
    Test the model for the given project.

    Args:
        project_name (str): The name of the project.
        experiment_name (str): The name of the experiment.
        override_experiment (bool): Whether to override existing experiment.

    Returns:
        None
    """
    controller = _build_controller(project_name=project_name)
    controller.setup_experiment(experiment_name=experiment_name)
    controller.test()


def fit_test(project_name: str, experiment_name: str, override_experiment: bool):
    """
    Fit and test the model for the given project.

    Args:
        project_name (str): The name of the project.
        experiment_name (str): The name of the experiment.
        override_experiment (bool): Whether to override existing experiment.

    Returns:
        None
    """
    controller = _build_controller(project_name=project_name)
    controller.setup_experiment(experiment_name=experiment_name, fold_name=None)
    controller.fit_test(override_experiment=override_experiment)


def predict(project_name: str, experiment_name: str, override_experiment: bool):
    """
    Make predictions using the model for the given project.

    Args:
        project_name (str): The name of the project.
        experiment_name (str): The name of the experiment.
        override_experiment (bool): Whether to override existing experiment.

    Returns:
        None
    """
    controller = _build_controller(project_name=project_name)
    controller.setup_experiment(experiment_name=experiment_name)
    controller.predict()


def kfold(project_name: str, experiment_name: str, override_experiment: bool):
    """
    Perform k-fold cross-validation for the given project.

    Args:
        project_name (str): The name of the project.
        experiment_name (str): The name of the experiment.
        override_experiment (bool): Whether to override existing experiment.

    Returns:
        None
    """
    controller = _build_controller(project_name=project_name)
    kfold_controller = KFold(controller=controller, experiment_name=experiment_name)
    kfold_controller()
