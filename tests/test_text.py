import unittest

from cryptochat.text import DefaultTextProcessor, SynonymError, SynonymMap, stem


class StemmerTests(unittest.TestCase):
    def test_returns_expected_base(self):
        cases = {"вернуть": "верн", "лицензию": "лицензи", "истекал": "истек", "счета": "счет",
                 "договор": "договор", "ngate": "ngate", "ipsec": "ipsec", "пин": "пин", "эдо": "эд"}
        for word, expected in cases.items():
            with self.subTest(word=word):
                self.assertEqual(expected, stem(word))

    def test_is_case_insensitive(self):
        self.assertEqual("лицензи", stem("ЛИЦЕНЗИЮ"))


def run(text, synonyms=None):
    return " ".join(DefaultTextProcessor(synonyms=synonyms).lemmatize(text))


class PreprocessorTests(unittest.TestCase):
    def test_lemmatize(self):
        cases = [
            ("как купить лицензию", "куп лицензи"),
            ("какие порты ipsec", "порт ipsec"),
            ("что такое кластер ngate", "кластер ngate"),
            ("я не вижу счет в эдо", "счет эд"),
            ("Пин-код рутокен", "пин код рутокен"),
            ("ЛИЦЕНЗИЯ истекла", "лицензи истекл"),
            ("версия 5.0.", "верси 5.0"),
            ("счёт", "счет"),
            ("как и что", ""),
            ("", ""),
            ("   ", ""),
        ]
        for text, expected in cases:
            with self.subTest(text=text):
                self.assertEqual(expected, run(text))

    def test_applies_synonyms(self):
        synonyms = SynonymMap([["вернуть", "возврат", "вернуть деньги"]])
        self.assertEqual("верн лицензи", run("как вернуть деньги за лицензию", synonyms))
        self.assertEqual("верн лицензи", run("возврат лицензии", synonyms))

    def test_group_with_one_word_raises(self):
        with self.assertRaises(SynonymError):
            SynonymMap([["вернуть"]])

    def test_word_in_two_groups_raises(self):
        with self.assertRaises(SynonymError):
            SynonymMap([["вернуть", "возврат"], ["отдать", "возврат"]])


if __name__ == "__main__":
    unittest.main()
