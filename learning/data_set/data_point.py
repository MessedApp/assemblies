from abc import abstractmethod, ABCMeta


class DataPoint(metaclass=ABCMeta):
    @abstractmethod
    @property
    def input(self):
        """
        :return: The data point's input value
        """
        pass

    @abstractmethod
    @property
    def output(self):
        """
        :return: The data point's output value
        """
        pass

