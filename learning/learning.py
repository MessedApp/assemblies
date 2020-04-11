from brain import Brain
from learning.data_set.lib.basic_types.partial_data_set import PartialDataSet
from learning.data_set.lib.training_set import TrainingSet
from learning.errors import ItemNotInitialized
from learning.learning_architecture import LearningArchitecture
from learning.learning_model import LearningModel


class Learning:

    def __init__(self, brain: Brain, domain_size: int):
        self.brain = brain
        self.domain_size = domain_size

        self._architecture = None
        self._training_set = None
        self._test_set = None

    @property
    def architecture(self):
        if not self._architecture:
            raise ItemNotInitialized('Architecture')
        return self._architecture

    @architecture.setter
    def architecture(self, architecture: LearningArchitecture):
        self._architecture = architecture

    @property
    def training_set(self):
        if not self._training_set:
            raise ItemNotInitialized('Training set')
        return self._training_set

    @training_set.setter
    def training_set(self, training_set: TrainingSet):
        self._training_set = training_set

    @property
    def test_set(self):
        if not self._test_set:
            raise ItemNotInitialized('Test set')
        return self._test_set

    @test_set.setter
    def test_set(self, test_set: PartialDataSet):
        self._test_set = test_set

    def create_model(self) -> LearningModel:
        """
        This function creates a learning model according to the configured preferences, trains it and tests it
        :return: the learning model
        """
        learning_model = LearningModel(brain=self.brain,
                                       domain_size=self.domain_size,
                                       architecture=self.architecture)
        learning_model.train_model(training_set=self.training_set)
        learning_model.test_model(test_set=self.test_set)
        return learning_model
