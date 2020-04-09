from math import log

from training.lib.training_set_base import TrainingSetBase


class ValuesListTrainingSet(TrainingSetBase):
    def __init__(self, return_values) -> None:
        super().__init__()
        self._return_values = return_values
        self._domain_size = self._get_domain_size(return_values)
        self._value = -1

    @staticmethod
    def _get_domain_size(return_values):
        domain_size = log(len(return_values), 2)
        if not domain_size.is_integer():
            raise ValueError("Return values list must be of proper length (a power of 2).")

        return domain_size

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
