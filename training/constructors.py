from typing import Callable, List

from training.lib.values_list_training_set import ValuesListTrainingSet as _ValuesListTrainingSet
from training.lib.callable_training_set import CallableTrainingSet as _CallableTrainingSet

from training.training_set import TrainingSet


def create_training_set_from_callable(function: Callable[[int], int],
                                      domain_size: int) -> TrainingSet:
    return _CallableTrainingSet(function, domain_size)


def create_training_set_from_list(return_values: List[int]) -> TrainingSet:
    return _ValuesListTrainingSet(return_values)

