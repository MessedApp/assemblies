from unittest import TestCase, skip

from learning.data_set.constructors import create_data_set_from_list
from learning.errors import DomainSizeMismatch
from learning.learning_architecture import LearningArchitecture
from learning.learning_configurations import LearningConfigurations
from learning.learning_model import LearningModel
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


class TestLearningModel(TestCase):

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
        self.architecture.add_stimulus_to_area_iteration(self.stim_a.name, self.area_a.name)
        self.architecture.add_stimulus_to_area_iteration(self.stim_b.name, self.area_a.name)
        self.architecture.add_stimulus_to_area_iteration(self.stim_a.name, self.area_b.name)
        self.architecture.add_stimulus_to_area_iteration(self.stim_b.name, self.area_b.name)

        self.architecture.add_area_to_area_iteration(self.area_a.name, self.area_c.name)
        self.architecture.add_area_to_area_iteration(self.area_b.name, self.area_c.name)

    @modify_customizations(1, 1)
    def test_run_model_sanity(self):
        model = LearningModel(brain=self.brain, domain_size=2, architecture=self.architecture)
        self.assertIn(model.run_model(0), [0, 1])
        self.assertIn(model.run_model(1), [0, 1])
        self.assertIn(model.run_model(2), [0, 1])
        self.assertIn(model.run_model(3), [0, 1])
        self.assertRaises(DomainSizeMismatch, model.run_model, 4)

    @modify_customizations(10, 10)
    def test_run_model_consistency(self):
        model = LearningModel(brain=self.brain, domain_size=2, architecture=self.architecture)

        result_00 = model.run_model(0)
        result_11 = model.run_model(3)

        result_00_2 = model.run_model(0)
        result_11_2 = model.run_model(3)

        self.assertEqual(result_11, result_11_2)
        self.assertEqual(result_00, result_00_2)

    @skip("Awaits 'training' flag in projection")
    @modify_customizations(10, 10)
    def test_train_model_sanity(self):
        model = LearningModel(brain=self.brain, domain_size=2, architecture=self.architecture)

        training_set = create_data_set_from_list([0, 1, 0, 1])

        model.train_model(training_set)
        model.test_model(training_set)
        self.assertEqual(1, model.accuracy)
