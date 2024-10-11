from assignment2 import SpellChecker
import unittest

class TestSpellChecker(unittest.TestCase):
    def check_ans(self, spell_checker, prompt, expected):
        result = spell_checker.check(prompt)
        error_message = f'Prompt "{prompt}" Expected {expected} but got {result}'
        self.assertEqual(len(expected), len(result), error_message)
        self.assertEqual(set(expected), set(result), error_message)

    def test_case_1(self):
        spell_checker = SpellChecker('text_files_q2/test_case_1.txt')

        prompt = "IDc"
        expected = ["IDA", "IDC", "IDJ"]
        self.check_ans(spell_checker, prompt, expected)

        prompt = "I"
        expected = []
        self.check_ans(spell_checker, prompt, expected)

        prompt = "Is"
        expected = ["I", "IDA", "IDC"]
        self.check_ans(spell_checker, prompt, expected)

        prompt = "meow"
        expected = ["me", "m"]
        self.check_ans(spell_checker, prompt, expected)

    def test_case_2(self):
        spell_checker = SpellChecker('text_files_q2/test_case_2.txt')
        
        prompt = "IDc"
        expected = ["IDK", "IDC", "IDA"]
        self.check_ans(spell_checker, prompt, expected)

        prompt = "I"
        expected = ["IDK", "IDC", "IDA"]
        self.check_ans(spell_checker, prompt, expected)

        prompt = "IDJM"
        expected = ["IDJ", "IDK", "IDC"]
        self.check_ans(spell_checker, prompt, expected)

        prompt = "IDAB"
        expected = ["IDA", "IDAA", "IDK"]
        self.check_ans(spell_checker, prompt, expected)

    def test_case_3(self):
        spell_checker = SpellChecker('text_files_q2/test_case_3.txt')
        
        prompt = "A"
        expected = ["Aespa"]
        self.check_ans(spell_checker, prompt, expected)
        
        prompt = "W"
        expected = ["Whiplash", "Woncho"]
        self.check_ans(spell_checker, prompt, expected)

        prompt = "O"
        expected = ["October21"]
        self.check_ans(spell_checker, prompt, expected)

        prompt = "Su"
        expected = ["Supernova", "Sageoneun"]
        self.check_ans(spell_checker, prompt, expected)

        prompt = "a"
        expected = []
        self.check_ans(spell_checker, prompt, expected)

    def test_case_4(self):
        spell_checker = SpellChecker('text_files_q2/test_case_4.txt')
        
        prompt = "ID"
        expected = ["IDA", "IDAA", "IDAAA"]
        self.check_ans(spell_checker, prompt, expected)


if __name__ == '__main__':
    unittest.main()