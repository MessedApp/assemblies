from unittest import TestCase

from parameterized import parameterized

from learning.errors import ArchitectureRunningNotInitialized
from learning.learning_architecture import LearningArchitecture
from tests import TestBrainUtils


class TestLearningArchitecture(TestCase):

    def setUp(self) -> None:
        self.utils = TestBrainUtils(lazy=False)
        self.brain = self.utils.create_brain(number_of_areas=5, number_of_stimuli=4)

    @parameterized.expand([
        ('one_cycle', 1),
        ('three_cycles', 3)
    ])
    def test_architecture_one_run_per_iteration(self, name, number_of_cycles):
        architecture = LearningArchitecture(self.brain, intermediate_area=self.utils.area4.name)
        architecture.add_stimulus_to_area_iteration(self.utils.stim0.name, self.utils.area0.name, 1)
        architecture.add_stimulus_to_area_iteration(self.utils.stim1.name, self.utils.area1.name, 1)
        architecture.add_area_to_area_iteration(self.utils.area0.name, self.utils.area2.name, 1)
        architecture.add_area_to_area_iteration(self.utils.area1.name, self.utils.area2.name, 1)

        expected_iterations = [
            (self.utils.stim0, self.utils.area0),
            (self.utils.stim1, self.utils.area1),
            (self.utils.area0, self.utils.area2),
            (self.utils.area1, self.utils.area2),
        ]
        expected_iterations = expected_iterations * number_of_cycles

        architecture.initialize_run(number_of_cycles=number_of_cycles)
        for idx, iteration in enumerate(architecture):
            self.assertEqual(expected_iterations[idx], iteration)

    def test_architecture_multiple_runs_per_iteration(self):
        architecture = LearningArchitecture(self.brain, intermediate_area=self.utils.area4.name)
        architecture.add_stimulus_to_area_iteration(self.utils.stim0.name, self.utils.area0.name, 2)
        architecture.add_stimulus_to_area_iteration(self.utils.stim1.name, self.utils.area1.name, 1)
        architecture.add_area_to_area_iteration(self.utils.area0.name, self.utils.area2.name, 3)
        architecture.add_area_to_area_iteration(self.utils.area1.name, self.utils.area2.name, 2)

        expected_iterations = [
            (self.utils.stim0, self.utils.area0),
            (self.utils.stim0, self.utils.area0),

            (self.utils.stim1, self.utils.area1),

            (self.utils.area0, self.utils.area2),
            (self.utils.area0, self.utils.area2),
            (self.utils.area0, self.utils.area2),

            (self.utils.area1, self.utils.area2),
            (self.utils.area1, self.utils.area2),
        ]

        architecture.initialize_run(number_of_cycles=1)
        for idx, iteration in enumerate(architecture):
            self.assertEqual(expected_iterations[idx], iteration)

    def test_architecture_not_initialized(self):
        architecture = LearningArchitecture(self.brain, intermediate_area=self.utils.area4.name)
        architecture.add_stimulus_to_area_iteration(self.utils.stim0.name, self.utils.area0.name, 3)

        # Iterating without initializing raises an error
        with self.assertRaises(ArchitectureRunningNotInitialized):
            for iteration in architecture:
                self.assertIsNotNone(iteration)

        # Initializing and starting to iterate
        architecture.initialize_run(number_of_cycles=1)
        for iteration in architecture:
            break

        # Iterating again without re-initializing raises an error
        with self.assertRaises(ArchitectureRunningNotInitialized):
            for iteration in architecture:
                self.assertIsNotNone(iteration)
