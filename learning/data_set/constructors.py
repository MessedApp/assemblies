from typing import Callable, List, Tuple

from learning.data_set.lib.basic_types.values_list_data_set import ValuesListDataSet as _ValuesListDataSet
from learning.data_set.lib.basic_types.callable_data_set import CallableDataSet as _CallableDataSet

from learning.data_set.data_set import DataSet, DataSets
from learning.data_set.lib.masks.lazy_mask import LazyMask as _LazyMask
from learning.data_set.lib.masks.explicit_mask import ExplicitMask as _ExplicitMask
from learning.data_set.lib.testing_set import TestingSet
from learning.data_set.lib.training_set import TrainingSet
from learning.data_set.mask import Mask


def create_data_set_from_callable(
        function: Callable[[int], int],
        domain_size: int,
        noise_probability: float = 0.) -> DataSet:
    return _CallableDataSet(function, domain_size, noise_probability)


def create_data_set_from_list(
        return_values: List[int],
        noise_probability: float = 0.) -> DataSet:
    return _ValuesListDataSet(return_values, noise_probability)


def create_lazy_mask(percentage: float, seed: int = None) -> Mask:
    return _LazyMask(percentage, seed)


def create_explicit_mask(mask_values: List[int]) -> Mask:
    return _ExplicitMask(mask_values)


def create_training_and_testing_sets_from_callable(
        data_set_function: Callable[[int], int],
        domain_size: int,
        mask: Mask,
        training_set_length: int,
        noise_probability: float = 0.) -> Tuple[DataSet, DataSet]:
    base_data_set = create_data_set_from_callable(data_set_function, domain_size, noise_probability)
    return DataSets(training_set=TrainingSet(base_data_set, mask, training_set_length, noise_probability),
                    testing_set=TestingSet(base_data_set, mask))


def create_training_and_testing_sets_from_list(
        data_set_return_values: List[int],
        mask: Mask,
        training_set_length: int,
        noise_probability: float = 0.) -> DataSets:
    base_data_set = create_data_set_from_list(data_set_return_values, noise_probability)
    return DataSets(training_set=TrainingSet(base_data_set, mask, training_set_length, noise_probability),
                    testing_set=TestingSet(base_data_set, mask))

