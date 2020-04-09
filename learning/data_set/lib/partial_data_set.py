from abc import ABCMeta

from learning.data_set.data_set import DataSet
from learning.data_set.lib.data_set_base import DataSetBase
from learning.data_set.lib.mask import Mask


class PartialDataSet(DataSetBase, metaclass=ABCMeta):
    def __init__(self, base_data_set: DataSet, mask: Mask,
                 noise_probability: float = 0.) -> None:
        super().__init__(noise_probability)
        self._base_data_set = base_data_set
        self._mask = mask

    @property
    def domain_size(self):
        return self._base_data_set.domain_size
