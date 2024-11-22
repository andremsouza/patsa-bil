"""This module contains a dictionary with the callbacks that can be used in the optuna study."""

from typing import Type, Dict

from optuna.study import (
    MaxTrialsCallback,
)

callback_dict: Dict[str, Type] = {
    "MaxTrials": MaxTrialsCallback,
}
