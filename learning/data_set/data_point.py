from abc import abstractmethod, ABCMeta


class DataPoint(metaclass=ABCMeta):
    @property
    @abstractmethod
    def input(self):
        """
        :return: The data point's input value
        """
        pass

    @property
    @abstractmethod
    def output(self):
        """
        :return: The data point's output value
        """
        pass

