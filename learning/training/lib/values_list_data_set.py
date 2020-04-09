from math import log

from learning.training.errors import DataSetSizeError
from learning.training.lib.data_set_base import DataSetBase


class ValuesListDataSet(DataSetBase):
    """
    An iterator defining the training set for a brain, based on a list of output
    values of binary function. For example, given binary function such as
    f(x) =  x (identity) or f(x, y) = (x + y) % 2, the list of values should be
    [0, 1] and [0, 1, 1, 0] (respectively).
    """
    def __init__(self, return_values, noise_probability=0.) -> None:
        super().__init__(noise_probability=noise_probability)
        self._return_values = return_values
        self._domain_size = self._get_domain_size(return_values)
        self._value = -1

    @staticmethod
    def _get_domain_size(return_values):
        domain_size = log(len(return_values), 2)
        if not domain_size.is_integer():
            raise DataSetSizeError(len(return_values))

        return int(domain_size)

    @property
    def domain_size(self):
        return self._domain_size

    @property
    def current_input_value(self):
        return self._value

    def _next(self):
        if self._value == 2 ** self._domain_size - 1:
            raise StopIteration()

        self._value += 1
        return self._return_values[self._value]
