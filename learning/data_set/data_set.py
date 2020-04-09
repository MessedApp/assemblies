from abc import ABCMeta, abstractmethod

from learning.data_set.data_point import DataPoint


class DataSet(metaclass=ABCMeta):
    """
    An abstract class for an iterator defining the data set for a brain,
    based on a binary function, such as f(x) =  x (identity) or f(x, y) = (x + y) % 2.
    """
    @abstractmethod
    def _next(self) -> DataPoint:
        """
        Get the next element in the data set, if any.
        :return: int
        """
        pass

    @abstractmethod
    def _get_item(self, item) -> DataPoint:
        """
        Get the element in the <item> index in the data_set.
        :return: int
        """
        pass

    @abstractmethod
    def set_noise_probability(self, noise_probability):
        """
        Set the noise probability of the data set to a new value.
        """
        pass

    @abstractmethod
    def __getitem__(self, item) -> DataPoint:
        pass

    @abstractmethod
    def __iter__(self):
        pass

    @abstractmethod
    def __next__(self) -> DataPoint:
        pass

    @property
    @abstractmethod
    def domain_size(self):
        """
        Get the domain size (i.e., the number of bits required to represent
        an item from the domain of the data set's function's).
        :return: int
        """
        pass

    @property
    @abstractmethod
    def current_input_value(self):
        """
        Get the current input value (i.e., the 10-based index of the element
        that has been issued last). Prior to calling next(), the function should
        return -1; after calling next() once, the function should return 0, and
        after the last valid call to next(), the function should return 2^domain_size-1
        :return: int
        """
        pass
