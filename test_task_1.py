from assignment2 import assign
import unittest

class TestTask1(unittest.TestCase):
    def test_spec_example(self):
        preferences = [[2, 1], [2, 2], [1, 1], [2, 1], [0, 2]]
        places = [2, 3]
        expected_list = [
            [[0, 3], [1, 4, 2]]
        ]
        got = assign(preferences, places)
        self._check_example(got, expected_list)
    
    def test_my_example_1(self):
        preferences = [[2, 1], [2, 2], [1, 1], [1, 1], [0, 2]]
        places = [2, 3]
        self.assertIsNone(assign(preferences, places))
    
    def _check_example(self, got, expected_list):
        self.assertTrue(
            any(self._compare(got, expected) for expected in expected_list)
        )
        
    def _compare(self, got, expected) -> bool:
        for got_place, expected_place in zip(got, expected):
            if set(got_place) !=  set(expected_place):
                return False
            elif len(got_place) != len(expected_place):
                return False
        return True

if __name__ == '__main__':
    unittest.main()