import unittest
import os
from assignment2 import SpellChecker

# change 
current_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(current_dir)

class TestTask2(unittest.TestCase):

    # def test_example_1(self):
    #     myChecker = SpellChecker('text_files/Nonsense.txt')
    #     self._check_example(myChecker, 'z', ['zzz'])
    #     self._check_example(myChecker, '0', ['0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefhijklmnopqrstuvz'])
    
    # def test_example_2(self):
    #     myChecker = SpellChecker('text_files/Empty.txt')
    #     self._check_example(myChecker, 'z', [])
    
    # def test_example_3(self):
    #     myChecker = SpellChecker('text_files/Punctuation.txt')
    #     self._check_example(myChecker, 'abc', [])
    
    # def test_example_4(self):
    #     myChecker = SpellChecker('text_files/Small.txt')
    #     self._check_example(myChecker, 'hh', ['h'])

    # def test_example_5(self):
    #     myChecker = SpellChecker('text_files/A.txt')
    #     self._check_example(myChecker, 'aa', ['aaa'])
    
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