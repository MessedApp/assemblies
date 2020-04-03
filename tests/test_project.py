import unittest

from tests.test_brain_utils import TestBrainUtils
from utils import get_matrix_max, get_matrix_min


class TestProject(unittest.TestCase):

    def test_project_sanity(self):
        utils = TestBrainUtils()
        brain = utils.create_and_stimulate_brain(number_of_areas=1, number_of_stimulated_areas=1)

        area = utils.area0
        winners_before_projection = area.winners
        connectomes_before_projection = brain.connectomes[area.name][area.name]
        self.assertEqual(utils.winners_size, len(winners_before_projection))
        self.assertEqual(1, get_matrix_max(connectomes_before_projection))
        self.assertEqual(0, get_matrix_min(connectomes_before_projection))

        brain.project(area_to_area={area.name: area.name}, stim_to_area={})
        connectomes_after_projection = brain.connectomes[area.name][area.name]
        self.assertEqual(utils.winners_size, len(area.winners))
        self.assertNotEqual(connectomes_after_projection, area.winners)
        self.assertEqual(1 + utils.beta, get_matrix_max(connectomes_after_projection))
        self.assertEqual(0, get_matrix_min(connectomes_after_projection))
