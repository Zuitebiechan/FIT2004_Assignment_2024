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

    def test_none_examples(self):
        preferences = [[2, 2], [1, 1], [0, 1], [2, 2], [1, 2]]
        places = [3, 2]
        self.expected_none(preferences, places)

        preferences = [[2, 0], [2, 0], [2, 0], [2, 0], [2, 0]]
        places = [3, 2]
        self.expected_none(preferences, places)

        preferences = [[0, 2], [0, 2], [0, 2], [0, 2], [0, 2]]
        places = [3, 2]
        self.expected_none(preferences, places)
    
        preferences = [[1, 2], [1, 2], [1, 2], [1, 2], [1, 2]]
        places = [3, 2]
        self.expected_none(preferences, places)

        preferences = [[2, 1], [2, 1], [2, 1], [2, 1], [2, 1]]
        places = [3, 2]
        self.expected_none(preferences, places)

        preferences = [[2, 2], [1, 1], [1, 1], [2, 2], [1, 1]]
        places = [3, 2]
        self.expected_none(preferences, places)

        preferences = [[1, 1], [1, 1], [1, 1], [1, 1], [1, 1]]
        places = [3, 2]
        self.expected_none(preferences, places)

    def test_other_examples(self):
        preferences = [[2, 1], [2, 1], [2, 0], [1, 2], [1, 2]]
        places = [3, 2]
        expected = [[0, 1, 2], [3, 4]]
        msg = self.check_ans(preferences, places, expected)
        self.assertIsNone(msg, msg)

        preferences = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
        places = [2, 3]
        result = assign(preferences, places)
        self.assertEqual(len(result), 2, f'Expected 2 activities, but got {len(result)} activities')
        self.assertEqual(len(result[0]), 2, f'Expected A1 to have 2 participants, but got {len(result[0])} participants')
        self.assertEqual(len(result[1]), 3, f'Expected A2 to have 3 participants, but got {len(result[1])} participants')
        all_activities = set(result[0]) | set(result[1])  
        self.assertEqual(len(all_activities), 5, f'Expected 5 participants to join the activities, but got {len(all_activities)} participants')

        preferences = [[2, 2], [2, 2], [2, 2], [2, 2], [2, 2]]
        places = [3, 2]
        result = assign(preferences, places)
        self.assertEqual(len(result), 2, f'Expected 2 activities, but got {len(result)} activities')
        self.assertEqual(len(result[0]), 3, f'Expected A1 to have 3 participants, but got {len(result[0])} participants')
        self.assertEqual(len(result[1]), 2, f'Expected A2 to have 2 participants, but got {len(result[1])} participants')
        all_activities = set(result[0]) | set(result[1])  
        self.assertEqual(len(all_activities), 5, f'Expected 5 participants to join the activities, but got {len(all_activities)} participants')

        preferences = [[2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 2, 2]]
        places = [3, 3, 2]
        result = assign(preferences, places)
        self.assertEqual(len(result), 3, f'Expected 3 activities, but got {len(result)} activities')
        self.assertEqual(len(result[0]), 3, f'Expected A1 to have 3 participants, but got {len(result[0])} participants')
        self.assertEqual(len(result[1]), 3, f'Expected A2 to have 3 participants, but got {len(result[1])} participants')
        self.assertEqual(len(result[2]), 2, f'Expected A3 to have 2 participants, but got {len(result[2])} participants')
        all_activities = set(result[0]) | set(result[1]) | set(result[2])
        self.assertEqual(len(all_activities), 8, f'Expected 8 participants to join the activities, but got {len(all_activities)} participants')
    
   
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
    
    def expected_none(self, preferences, places):
        result = assign(preferences, places)
        error_message = f'Expected None but got {result}'
        self.assertIsNone(result, error_message)
    
    def check_ans(self, preferences, places, expected):
        result = assign(preferences, places)
        error_message = f'Expected {expected} but got {result}'
        if len(expected) != len(result):
            return error_message
        for i in range(len(expected)):
            if set(expected[i]) != set(result[i]):
                return error_message        
        return None

if __name__ == '__main__':
    unittest.main()
