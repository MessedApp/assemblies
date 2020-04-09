from abc import ABCMeta, abstractmethod


class TrainingSet(metaclass=ABCMeta):
    """
    An abstract class for an iterator defining the training set for a brain,
    based on a binary function, such as f(x) =  x (identity) or f(x, y) = (x + y) % 2.
    """
    @abstractmethod
    def _next(self):
        """
        Get the next element in the training set, if any.
        :return: int
        """
        pass

    def __iter__(self):
        pass

    def __next__(self):
        pass

    @property
    @abstractmethod
    def domain_size(self):
        """
        Get the domain size (i.e., the number of bits required to represent
        an item from the domain of the training set's function's).
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
