from abc import ABCMeta, abstractmethod

from training.errors import DataSetValueError


class TrainingSet(metaclass=ABCMeta):
    """
    An iterator defining the training set for a brain, based on a binary
    function, such as f(x) =  x (identity) or f(x, y) = (x + y) % 2.
    """
    @abstractmethod
    def _next(self):
        pass

    def __iter__(self):
        return self

    def __next__(self):
        output_value = self._next()
        if output_value not in (0, 1):
            raise DataSetValueError(self.current_input_value, output_value)

        return output_value

    @property
    @abstractmethod
    def domain_size(self):
        pass

    @property
    @abstractmethod
    def current_input_value(self):
        pass
