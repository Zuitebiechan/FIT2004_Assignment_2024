import unittest
import os
from assignment2 import SpellChecker

# change working directory just in case
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

class TestTask2(unittest.TestCase):
    
    def test_spec_example_1(self):
        myChecker = SpellChecker("text_files/Messages.txt")
        self.assertEqual(myChecker.check('IDK'), [])
    
    def test_spec_example_2(self):
        myChecker = SpellChecker("text_files/Messages.txt")
        self.assertEqual(myChecker.check('zoo'), [])
    
    def test_spec_example_3(self):
        myChecker = SpellChecker("text_files/Messages.txt")
        self._check_example(myChecker, 'LOK', ['LOL', 'LMK'])
    
    def test_spec_example_4(self):
        myChecker = SpellChecker("text_files/Messages.txt")
        self._check_example(myChecker, 'IDP', ['IDK', 'IDC', 'I'])
    
    def test_spec_example_5(self):
        myChecker = SpellChecker("text_files/Messages.txt")
        self._check_example(myChecker, 'Ifc', ['If', 'I', 'IDK'])

    def _check_example(self, checker: SpellChecker, input_word: str, expected: list):
        got = checker.check(input_word)
        self.assertEqual(set(got), set(expected), f'Expected {expected}, got {got}')
        self.assertEqual(len(got), len(expected), f'There was a duplicate element in {got}')

if __name__ == '__main__':
    unittest.main()

