from tests.test_non_lazy_brain import TestNonLazyBrain
from utils import get_matrix_max, get_matrix_min


class TestProject(TestNonLazyBrain):

    def test_project_from_area_to_itself(self):
        brain = self.utils.create_and_stimulate_brain(number_of_areas=1, number_of_stimulated_areas=1)

        area = self.utils.area0
        winners_before_projection = area.winners
        connectomes_before_projection = brain.connectomes[area.name][area.name]
        self.assertEqual(self.utils.winners_size, len(winners_before_projection))
        self.assertEqual(1, get_matrix_max(connectomes_before_projection))
        self.assertEqual(0, get_matrix_min(connectomes_before_projection))

        brain.project(area_to_area={area.name: [area.name]}, stim_to_area={})
        connectomes_after_projection = brain.connectomes[area.name][area.name]
        self.assertEqual(self.utils.winners_size, len(area.winners))
        self.assertNotEqual(connectomes_after_projection, area.winners)
        self.assertAlmostEqual((1 + self.utils.beta) * 1, get_matrix_max(connectomes_after_projection))
        self.assertEqual(0, get_matrix_min(connectomes_after_projection))

    def test_project_from_area_to_another_area(self):
        brain = self.utils.create_and_stimulate_brain(number_of_areas=2, number_of_stimulated_areas=1)

        source_area = self.utils.area0
        target_area = self.utils.area1

        self.assertEqual([], target_area.winners)

        brain.project(area_to_area={source_area.name: [target_area.name]}, stim_to_area={})
        connectomes_after_projection = brain.connectomes[source_area.name][target_area.name]
        self.assertEqual(source_area.k, len(source_area.winners))
        self.assertAlmostEqual((1 + target_area.beta) * 1, get_matrix_max(connectomes_after_projection))
        self.assertEqual(0, get_matrix_min(connectomes_after_projection))

    def test_project_from_area_to_output_area(self):
        brain = self.utils.create_and_stimulate_brain(number_of_areas=1, number_of_stimulated_areas=1,
                                                      add_output_area=True)

        origin_area = self.utils.area0
        output_area = self.utils.output_area

        self.assertEqual([], output_area.winners)

        brain.project(area_to_area={origin_area.name: [output_area.name]}, stim_to_area={})
        connectomes_after_projection = brain.output_connectomes[origin_area.name][output_area.name]
        self.assertEqual(origin_area.k, len(origin_area.winners))
        self.assertAlmostEqual((1 + output_area.beta) * 1, get_matrix_max(connectomes_after_projection))
        self.assertEqual(0, get_matrix_min(connectomes_after_projection))
