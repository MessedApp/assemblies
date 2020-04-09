import random

from learning.data_set.data_point import DataPoint
from learning.data_set.data_set import DataSet
from learning.data_set.lib.data_point import DataPointImpl
from learning.data_set.lib.mask import Mask
from learning.data_set.lib.partial_data_set import PartialDataSet


class TrainingSet(PartialDataSet):
    def __init__(self, base_data_set: DataSet, mask: Mask, length: int = None,
                 noise_probability: float = 0.) -> None:
        super().__init__(base_data_set, mask, noise_probability)
        self._count_left = length
        self._range_max = 2 ** self.domain_size - 1

    def _next(self) -> DataPoint:
        if self._count_left == 0:
            raise StopIteration()

        mask_value = 0
        index = 0
        while not mask_value:
            index = random.randint(0, self._range_max)
            mask_value = self._mask.in_training_set(index)

        self._count_left -= 1
        return DataPointImpl(index, self._base_data_set[index])

    def _get_item(self, item) -> DataPoint:
        raise NotImplemented()

