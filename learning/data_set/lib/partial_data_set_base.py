from abc import ABCMeta
from random import random

from learning.data_set.data_point import DataPoint
from learning.data_set.data_set import DataSet
from learning.data_set.lib.data_point import DataPointImpl
from learning.data_set.lib.data_set_base import DataSetBase
from learning.data_set.lib.mask import Mask


class PartialDataSetBase(DataSetBase, metaclass=ABCMeta):
    def __init__(self, base_data_set: DataSet, mask: Mask, length: int = None,
                 noise_probability: float = 0.) -> None:
        super().__init__(noise_probability)
        self._base_data_set = base_data_set
        self._mask = mask
        self._count_left = length
        self._range_max = 2 ** self.domain_size - 1

    def __iter__(self):
        return self

    def _next(self) -> DataPoint:
        if self._count_left == 0:
            raise StopIteration()

        mask_value = 0
        index = 0
        while not mask_value:
            index = random.randint(0, self._range_max)
            mask_value = self._mask.get_mask(index)

        self._count_left -= 1
        return DataPointImpl(index, self._base_data_set[index])
