from typing import Callable, List

from learning.data_set.lib.basic_types.values_list_data_set import ValuesListDataSet as _ValuesListDataSet
from learning.data_set.lib.basic_types.callable_data_set import CallableDataSet as _CallableDataSet

from learning.data_set.data_set import DataSet


def create_data_set_from_callable(
        function: Callable[[int], int],
        domain_size: int,
        noise_probability: float = 0.) -> DataSet:
    return _CallableDataSet(function, domain_size, noise_probability)


def create_data_set_from_list(
        return_values: List[int],
        noise_probability: float = 0.) -> DataSet:
    return _ValuesListDataSet(return_values, noise_probability)

