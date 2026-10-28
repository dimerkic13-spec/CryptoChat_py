import json
import tempfile
import unittest
from pathlib import Path

from cryptochat import knowledge
from cryptochat.knowledge import RawCategory, Severity, validate
from cryptochat.models import ContentError

from .support import CONTENT_DIR, valid


def codes(content):
    return [i.code for i in validate(content)]


class ContentValidatorTests(unittest.TestCase):
    def test_valid_content_has_no_issues(self):
        self.assertEqual([], validate(valid()))

    def test_v001_duplicate_question_id(self):
        c = valid(); c.questions[1].id = "1.1.1"
        self.assertIn("V001", codes(c))

    def test_v001_duplicate_category_id(self):
        c = valid(); c.categories[0].children.append(RawCategory(id="1.1", name="Копия", kind="subtopic"))
        self.assertIn("V001", codes(c))

    def test_v002_unknown_category(self):
        c = valid(); c.questions[0].category = "9.9"
        self.assertIn("V002", codes(c))

    def test_v002_category_is_not_leaf(self):
        c = valid(); c.questions[0].category = "1"; c.questions[0].id = "1.5"
        self.assertIn("V002", codes(c))

    def test_v003_id_does_not_match_category(self):
        c = valid(); c.questions[0].id = "9.9.9"
        self.assertIn("V003", codes(c))

    def test_v004_empty_answer(self):
        c = valid(); c.questions[0].answer = "   "
        self.assertIn("V004", codes(c))

    def test_v005_source_is_not_https(self):
        c = valid(); c.questions[0].source.url = "http://example.com"
        self.assertIn("V005", codes(c))

    def test_v005_source_is_missing(self):
        c = valid(); c.questions[0].source = None
        self.assertIn("V005", codes(c))

    def test_v006_too_few_keywords(self):
        c = valid(); del c.questions[0].keywords[2]
        self.assertIn("V006", codes(c))

    def test_v006_many_keywords_are_allowed(self):
        c = valid()
        c.questions[0].keywords += [knowledge.RawKeyword(word=w, weight=1) for w in
                                    ("один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять")]
        self.assertNotIn("V006", codes(c))

    def test_v007_weight_out_of_range(self):
        c = valid(); c.questions[0].keywords[0].weight = 0
        self.assertIn("V007", codes(c))

    def test_v008_duplicate_keyword_after_stemming(self):
        c = valid(); c.questions[0].keywords[2].word = "лицензию"
        self.assertIn("V008", codes(c))

    def test_v009_mixed_latin_and_cyrillic(self):
        c = valid(); c.questions[1].keywords[2].word = "ngаte"  # «а» здесь кириллическая
        self.assertIn("V009", codes(c))

    def test_v010_keyword_empty_after_stemming(self):
        c = valid(); c.questions[0].keywords[2].word = "и"
        self.assertIn("V010", codes(c))

    def test_v011_child_id_does_not_start_with_parent(self):
        c = valid(); c.categories[0].children[0].id = "3.1"
        self.assertIn("V011", codes(c))

    def test_v012_group_with_one_word(self):
        c = valid(); c.synonym_groups.append(["пин"])
        self.assertIn("V012", codes(c))

    def test_v012_word_in_two_groups(self):
        c = valid(); c.synonym_groups.append(["отдать", "возврат"])
        self.assertIn("V012", codes(c))

    def _only_warning(self, content, code):
        issues = validate(content)
        self.assertTrue(any(i.code == code and i.severity is Severity.WARNING for i in issues))
        self.assertFalse(any(i.severity is Severity.ERROR for i in issues))

    def test_v013_empty_leaf_is_only_warning(self):
        c = valid(); c.categories[0].children.append(RawCategory(id="1.2", name="Документы", kind="subtopic"))
        self._only_warning(c, "V013")

    def test_v014_stop_word_keyword_is_only_warning(self):
        c = valid(); c.questions[0].keywords[2].word = "как"
        self._only_warning(c, "V014")

    def test_v016_bad_revision_is_only_warning(self):
        c = valid(); c.questions[0].source.revision = "09.10.2026"
        self._only_warning(c, "V016")


TAXONOMY = {"categories": [{"id": "1", "name": "Коммерческие", "kind": "topic",
                            "children": [{"id": "1.1", "name": "Лицензии", "kind": "subtopic"}]}]}
GOOD_QUESTIONS = [{"id": "1.1.1", "category": "1.1", "title": "Как купить лицензию?", "answer": "Ответ.",
                   "source": {"url": "https://cryptopro.ru/faq/how-to-buy", "anchor": "a", "revision": "2026-10-09"},
                   "keywords": [{"word": "купить", "weight": 5}, {"word": "лицензия", "weight": 3},
                                {"word": "оплата", "weight": 1}]}]


class KnowledgeLoaderTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.root = Path(self._tmp.name)

    def tearDown(self):
        self._tmp.cleanup()

    def write(self, relative, data):
        path = self.root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(data if isinstance(data, str) else json.dumps(data, ensure_ascii=False), encoding="utf-8")

    def test_reads_and_stems_keywords(self):
        self.write("taxonomy.json", TAXONOMY)
        self.write("questions/a.json", GOOD_QUESTIONS)
        kb = knowledge.load(self.root)
        self.assertEqual(1, len(kb.roots))
        self.assertEqual(1, len(kb.questions))
        self.assertEqual(["куп", "лицензи", "оплат"], [k.word for k in kb.questions[0].keywords])
        self.assertEqual(1, len(kb.questions_under("1")))
        self.assertEqual([], kb.questions_under("2"))

    def test_applies_synonyms_to_keywords(self):
        self.write("taxonomy.json", TAXONOMY)
        self.write("synonyms.json", {"groups": [["вернуть", "возврат"]]})
        self.write("questions/a.json", [{"id": "1.1.1", "category": "1.1", "title": "Как вернуть лицензию?",
                                         "answer": "Ответ.", "source": {"url": "https://cryptopro.ru/faq/how-to-buy"},
                                         "keywords": [{"word": "возврат", "weight": 5}, {"word": "деньги", "weight": 3},
                                                      {"word": "лицензия", "weight": 1}]}])
        kb = knowledge.load(self.root)
        self.assertIn("верн", [k.word for k in kb.questions[0].keywords])

    def test_missing_directory_raises(self):
        with self.assertRaises(ContentError):
            knowledge.load(self.root / "no-such")

    def test_missing_taxonomy_raises(self):
        with self.assertRaises(ContentError) as ctx:
            knowledge.load(self.root)
        self.assertIn("taxonomy.json", str(ctx.exception))

    def test_broken_json_raises_with_file_name(self):
        self.write("taxonomy.json", "{ это не JSON")
        with self.assertRaises(ContentError) as ctx:
            knowledge.load(self.root)
        self.assertIn("taxonomy.json", str(ctx.exception))

    def test_invalid_content_raises_with_codes(self):
        self.write("taxonomy.json", TAXONOMY)
        broken = json.loads(json.dumps(GOOD_QUESTIONS)); broken[0]["answer"] = ""
        self.write("questions/a.json", broken)
        with self.assertRaises(ContentError) as ctx:
            knowledge.load(self.root)
        self.assertIn("V004", str(ctx.exception))


class RealContentTests(unittest.TestCase):
    def test_has_no_validation_errors(self):
        errors = [i for i in validate(knowledge.load_content(CONTENT_DIR)) if i.severity is Severity.ERROR]
        self.assertEqual([], errors, "\n".join(map(str, errors)))

    def test_loads_expected_number_of_questions(self):
        kb = knowledge.load(CONTENT_DIR)
        self.assertEqual(38, len(kb.questions))
        self.assertEqual(15, len(kb.questions_under("1.1")))
        self.assertEqual(17, len(kb.questions_under("1.2")))
        self.assertEqual(6, len(kb.questions_under("1.3")))
        self.assertEqual([], kb.questions_under("2"))

    def test_titles_are_unique(self):
        titles = [q.title for q in knowledge.load(CONTENT_DIR).questions]
        self.assertEqual(len(titles), len(set(titles)))


if __name__ == "__main__":
    unittest.main()
