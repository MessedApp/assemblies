from abc import ABCMeta

import numpy as np

from learning.data_set.data_point import DataPoint
from learning.data_set.errors import DataSetValueError
from learning.data_set.data_set import DataSet


class DataSetBase(DataSet, metaclass=ABCMeta):
    """
    An abstract class for an iterator defining the data_set set for a brain,
    based on a binary function, such as f(x) =  x (identity) or f(x, y) = (x + y) % 2.
    This class contains any and all shared logic between different types of DataSets.
    """
    def __init__(self, noise_probability=0.) -> None:
        super().__init__()
        self._noise_probability = noise_probability

    def __iter__(self):
        return self

    def __getitem__(self, item) -> DataPoint:
        if not 0 <= item < 2 ** self.domain_size:
            raise IndexError(f"Item of index {item} is out of range (choose an"
                             f"index between 0 and {2 ** self.domain_size})")

        return self._process_output_values(item, self._get_item(item))

    def __next__(self) -> DataPoint:
        output_value = self._next()
        return self._process_output_values(self.current_input_value, output_value)

    def _process_output_values(self, input_value, output_value):
        if output_value not in (0, 1):
            raise DataSetValueError(input_value, output_value)

        noise = np.random.binomial(1, self._noise_probability)
        return output_value ^ noise

    def set_noise_probability(self, noise_probability):
        self._noise_probability = noise_probability
