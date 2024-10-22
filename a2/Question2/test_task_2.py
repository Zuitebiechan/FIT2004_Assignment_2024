import unittest
import os
from q2_answer import SpellChecker

# change working directory just in case
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

class TestTask2(unittest.TestCase):
    
    # def test_spec_example_1(self):
    #     myChecker = SpellChecker("text_files/Messages.txt")
    #     self.assertEqual(myChecker.check('IDK'), [])
    
    # def test_spec_example_2(self):
    #     myChecker = SpellChecker("text_files/Messages.txt")
    #     self.assertEqual(myChecker.check('zoo'), [])
    
    # def test_spec_example_3(self):
    #     myChecker = SpellChecker("text_files/Messages.txt")
    #     self._check_example(myChecker, 'LOK', ['LOL', 'LMK'])
    
    # def test_spec_example_4(self):
    #     myChecker = SpellChecker("text_files/Messages.txt")
    #     self._check_example(myChecker, 'IDP', ['IDK', 'IDC', 'I'])
    
    # def test_spec_example_5(self):
    #     myChecker = SpellChecker("text_files/Messages.txt")
    #     self._check_example(myChecker, 'Ifc', ['If', 'I', 'IDK'])

    def test_example_6(self): # WARNING: Very large, will take a while
        myChecker = SpellChecker('text_files/VeryLarge.txt')
        self._check_example(myChecker, 'Hello', ['Hel', 'Helsxa', 'HelynaR'])
        self._check_example(myChecker, 'l', [])
        self._check_example(myChecker, 'abc', ['abcI9E', 'abcZq0cY9sl18UB', 'ab'])
        self._check_example(myChecker, '2137', ['213', '213T', '213lTyc'])

    def _check_example(self, checker: SpellChecker, input_word: str, expected: list):
        got = checker.check(input_word)
        self.assertEqual(set(got), set(expected), f'Expected {expected}, got {got}')
        self.assertEqual(len(got), len(expected), f'There was a duplicate element in {got}')

if __name__ == '__main__':
    unittest.main()

