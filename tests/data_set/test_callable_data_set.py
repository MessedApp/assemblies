from unittest import TestCase

from learning.data_set.constructors import create_data_set_from_callable
from learning.data_set.errors import DataSetValueError


class TestCallableDataSet(TestCase):
    def test_data_set_with_list_of_2_works(self):
        s = create_data_set_from_callable(lambda x: 1 - x, 1)
        self.assertEqual(1, s.domain_size)
        self.assertEqual(1, next(s))
        self.assertEqual(0, next(s))
        self.assertRaises(StopIteration, next, s)

    def test_data_set_with_non_boolean_values_fails(self):
        s = create_data_set_from_callable(lambda x: x + 1, 1)
        self.assertEqual(1, next(s))
        self.assertRaises(DataSetValueError, next, s)

    def test_data_set_with_list_of_16_works(self):
        expected = [
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        ]
        s = create_data_set_from_callable(lambda x: (1 - x) % 2, 4)
        self.assertEqual(4, s.domain_size)
        for expected_value in expected:
            self.assertEqual(expected_value, next(s))
        self.assertRaises(StopIteration, next, s)

    def test_data_set_iterable_works(self):
        expected = [
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        ]
        s = create_data_set_from_callable(lambda x: (1 - x) % 2, 4)
        for i, value in enumerate(s):
            self.assertEqual(expected[i], value)
        self.assertRaises(StopIteration, next, s)

    def test_data_set_with_full_noise_flips_all_results(self):
        expected_not_noisy = [
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        ]
        s = create_data_set_from_callable(lambda x: (1 - x) % 2, 4, noise_probability=1)
        for i, value in enumerate(s):
            self.assertEqual(expected_not_noisy[i] ^ 1, value)
        self.assertRaises(StopIteration, next, s)

    def test_data_set_with_noise_flips_some_results(self):
        expected_not_noisy = [
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0,
            1, 0, 1, 0
        ]
        s = create_data_set_from_callable(lambda x: (1 - x) % 2, 4, noise_probability=0.5)
        count_flipped = sum(expected_not_noisy[i] ^ value for i, value in enumerate(s))

        # Note: This test can fail with extremely low probability.
        #       If it does, run again to verify it was one of those extreme cases.
        self.assertLess(0, count_flipped)
        self.assertGreater(len(expected_not_noisy), count_flipped)
        self.assertRaises(StopIteration, next, s)
