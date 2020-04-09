from abc import ABCMeta

from training.errors import DataSetValueError
from training.training_set import TrainingSet


class TrainingSetBase(TrainingSet, metaclass=ABCMeta):
    def __iter__(self):
        return self

    def __next__(self):
        output_value = self._next()
        if output_value not in (0, 1):
            raise DataSetValueError(self.current_input_value, output_value)

        return output_value
