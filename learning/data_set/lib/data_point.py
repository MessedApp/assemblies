from learning.data_set.data_point import DataPoint


class DataPointImpl(DataPoint):
    def __init__(self, input_value, output_value) -> None:
        super().__init__()
        self._input = input_value
        self._output = output_value

    @property
    def input(self):
        return self._input

    @property
    def output(self):
        return self._output

