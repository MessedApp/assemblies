from abc import ABCMeta, abstractmethod


class TrainingSet(metaclass=ABCMeta):
    """
    An iterator defining the training set for a brain, based on a binary
    function, such as f(x) =  x (identity) or f(x, y) = (x + y) % 2.
    """
    @abstractmethod
    def _next(self):
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    @property
    @abstractmethod
    def domain_size(self):
        pass

    @property
    @abstractmethod
    def current_input_value(self):
        pass
