from unittest import TestCase

from learning.learning_architecture import LearningArchitecture
from learning.learning_configurations import LearningConfigurations
from tests import TestBrainUtils


def modify_customizations(supervised, unsupervised):
    def decorator(function):
        def wrapper(*args, **kwargs):
            original_data = \
                LearningConfigurations.NUMBER_OF_SUPERVISED_CYCLES, LearningConfigurations.NUMBER_OF_UNSUPERVISED_CYCLES
            LearningConfigurations.NUMBER_OF_SUPERVISED_CYCLES, LearningConfigurations.NUMBER_OF_UNSUPERVISED_CYCLES = \
                supervised, unsupervised
            result = function(*args, **kwargs)
            LearningConfigurations.NUMBER_OF_SUPERVISED_CYCLES, LearningConfigurations.NUMBER_OF_UNSUPERVISED_CYCLES = \
                original_data
            return result
        return wrapper
    return decorator


class TestLearningBase(TestCase):

    def setUp(self) -> None:
        utils = TestBrainUtils(lazy=False)
        self.brain = utils.create_brain(number_of_areas=3, number_of_stimuli=4,
                                        area_size=100, winners_size=10)

        self.area_a = utils.area0
        self.area_b = utils.area1
        self.area_c = utils.area2

        self.stim_a = utils.stim0
        self.stim_b = utils.stim1
        self.stim_c = utils.stim2
        self.stim_d = utils.stim3

        self.architecture = LearningArchitecture(self.brain, intermediate_area=self.area_c.name)
        self.architecture.add_stimulus_to_area_iteration('A', 'A')
        self.architecture.add_stimulus_to_area_iteration('B', 'A')
        self.architecture.add_stimulus_to_area_iteration('C', 'B')
        self.architecture.add_stimulus_to_area_iteration('D', 'B')

        self.architecture.add_area_to_area_iteration('A', 'C')
        self.architecture.add_area_to_area_iteration('B', 'C')