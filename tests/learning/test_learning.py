from learning.data_set.constructors import create_data_set_from_list
from learning.learning import Learning
from tests.learning import TestLearningBase, modify_configurations


class TestLearning(TestLearningBase):

    @modify_configurations(30, 30)
    def test_learning_sanity(self):
        learning = Learning(brain=self.brain, domain_size=2)

        data_set = create_data_set_from_list([0, 1, 0, 1])

        learning.architecture = self.architecture
        learning.training_set = data_set

        model = learning.create_model()
        test_results = model.test_model(data_set)

        self.assertEqual(1, test_results.accuracy) # TODO: There may be models in which this is not necessary
