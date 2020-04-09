from typing import List

from learning.data_set.errors import MaskValueError
from learning.data_set.mask import Mask


class ExplicitMask(Mask):
    def __init__(self, mask_values: List[int]) -> None:
        """
        Create a mask based entirely of the given mask values. Default values
        for indices out the given mask_values' range will be 0.
        """
        super().__init__()
        self._mask_values = mask_values

    @staticmethod
    def _validate_mask_value(index, mask_value):
        if mask_value not in (0, 1):
            raise MaskValueError(index, mask_value)

    def in_training_set(self, index) -> bool:
        """
        Get the value of the mask for the training set.
        :return: True if the index is included in the mask, or False if it isn't.
        """
        mask_value = self._mask_values[index]
        self._validate_mask_value(index, mask_value)
        return bool(mask_value)

    def in_testing_set(self, index) -> bool:
        """
        Get the value of the mask for the testing set.
        :return: True if the index is included in the mask, or False if it isn't.
        """
        return not self.in_training_set(index)
