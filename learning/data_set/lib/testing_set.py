from learning.data_set.data_point import DataPoint
from learning.data_set.data_set import DataSet
from learning.data_set.lib.basic_types.partial_data_set import PartialDataSet
from learning.data_set.mask import Mask


class TestingSet(PartialDataSet):
    """
    TestingSet is the partial data set representing the set used for the
    testing phase.
    A testing set does not contain noise (by definition), and is ordered.
    Iterating over the testing set will output data points from the portion of
    the data dedicated to testing.
    """
    def __init__(self, base_data_set: DataSet, mask: Mask) -> None:
        super().__init__(base_data_set, mask, noise_probability=0.)

    def _next(self) -> DataPoint:
        data_point = next(self._base_data_set)
        mask_value = self._mask.in_testing_set(self._value)

        while not mask_value:
            data_point = next(self._base_data_set)

        return data_point
