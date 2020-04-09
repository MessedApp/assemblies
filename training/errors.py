class DataSetValueError(Exception):
    def __init__(self, input_value, output_value, *args: object) -> None:
        super().__init__(*args)
        self._input_value = input_value
        self._output_value = output_value

    def __str__(self) -> str:
        return f"Received an invalid value for a boolean function data set: " \
               f"input {self._input_value} returned value " \
               f"{self._output_value}, which is not a boolean value."


