from learning.data_set.constructors import create_data_set_from_list
from learning.learning import Learning
from tests.learning import TestLearningBase, modify_customizations


class TestLearning(TestLearningBase):

    @modify_customizations(30, 30)
    def test_learning_sanity(self):
        learning = Learning(brain=self.brain, domain_size=2)
        learning.architecture = self.architecture

        data_set = create_data_set_from_list([0, 1, 0, 1])
        learning.training_set = data_set
        learning.test_set = data_set

        model = learning.create_model()
        self.assertEqual(1, model.accuracy)
