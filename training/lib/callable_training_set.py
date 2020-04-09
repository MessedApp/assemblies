from training.training_set import TrainingSet


class CallableTrainingSet(TrainingSet):
    def __init__(self, function, domain_size) -> None:
        super().__init__()
        self._function = function
        self._domain_size = domain_size
        self._value = -1

    @property
    def domain_size(self):
        return self._domain_size

    @property
    def current_input_value(self):
        return self._value

    def _next(self):
        if self._value == 2 ** self._domain_size - 1:
            raise StopIteration()

        self._value += 1
        return self._function(self._value)
