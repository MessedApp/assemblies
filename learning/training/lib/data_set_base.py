from abc import ABCMeta

import numpy as np

from learning.training.errors import DataSetValueError
from learning.training.data_set import DataSet


class DataSetBase(DataSet, metaclass=ABCMeta):
    """
    An abstract class for an iterator defining the training set for a brain,
    based on a binary function, such as f(x) =  x (identity) or f(x, y) = (x + y) % 2.
    This class contains any and all shared logic between different types of DataSets.
    """
    def __init__(self, noise_probability=0.) -> None:
        super().__init__()
        self._noise_probability = noise_probability

    def __iter__(self):
        return self

    def __next__(self):
        output_value = self._next()
        if output_value not in (0, 1):
            raise DataSetValueError(self.current_input_value, output_value)

        noise = np.random.binomial(1, self._noise_probability)
        return output_value ^ noise
