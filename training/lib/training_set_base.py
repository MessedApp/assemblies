from abc import ABCMeta

import numpy as np

from training.errors import DataSetValueError
from training.training_set import TrainingSet


class TrainingSetBase(TrainingSet, metaclass=ABCMeta):
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
