"""Морфология: проверка на поддельном анализаторе (работает без pymorphy3)
и, если pymorphy3 установлен, на настоящем."""
import unittest

from cryptochat.cli import check_acceptance
from cryptochat.engine import ChatEngine
from cryptochat.knowledge import KnowledgeBase, Severity, validate
from cryptochat.models import AppConfig, ReplyKind
from cryptochat.text import (DefaultTextProcessor, MorphLemmatizer, MorphologyUnavailable, SynonymError,
                             SynonymMap, make_lemmatizer, stem)

from .support import CONTENT_DIR, kw, valid


class _Parse:
    def __init__(self, normal_form):
        self.normal_form = normal_form


class FakeAnalyzer:
    """Ведёт себя как pymorphy3.MorphAnalyzer: parse(слово)[0].normal_form."""
    FORMS = {
        "лицензию": "лицензия", "лицензии": "лицензия", "лицензия": "лицензия",
        "купить": "купить", "купил": "купить", "куплю": "купить", "купили": "купить",
        "оплата": "оплата", "оплату": "оплата", "кластер": "кластер", "кластера": "кластер",
        "узел": "узел", "вернуть": "вернуть", "возврат": "возврат", "счета": "счёт",
        "присоединились": "присоединиться", "присоединиться": "присоединиться",
    }

    def __init__(self):
        self.calls = 0

    def parse(self, word):
        self.calls += 1
        return [_Parse(self.FORMS.get(word, word))]


def fake_lemmatizer():
    return MorphLemmatizer(FakeAnalyzer())


class MorphLemmatizerTests(unittest.TestCase):
    def test_gives_normal_form(self):
        lem = fake_lemmatizer()
        self.assertEqual("лицензия", lem("лицензию"))
        self.assertEqual("присоединиться", lem("присоединились"))

    def test_replaces_yo_in_result(self):
        self.assertEqual("счет", fake_lemmatizer()("счёта"))

    def test_keeps_words_without_cyrillic(self):
        analyzer = FakeAnalyzer()
        lem = MorphLemmatizer(analyzer)
        self.assertEqual("ngate", lem("ngate"))
        self.assertEqual("5.0", lem("5.0"))
        self.assertEqual(0, analyzer.calls)

    def test_caches_results(self):
        analyzer = FakeAnalyzer()
        lem = MorphLemmatizer(analyzer)
        lem("лицензию"); lem("лицензию")
        self.assertEqual(1, analyzer.calls)

    def test_pipeline_uses_morphology(self):
        processor = DefaultTextProcessor(lemmatize=fake_lemmatizer())
        self.assertEqual(["купить", "лицензия"], processor.lemmatize("как купил лицензию"))


class MorphSynonymsAndValidationTests(unittest.TestCase):
    def test_same_lemma_inside_one_group_is_allowed(self):
        synonyms = SynonymMap([["купить", "купил", "куплю", "покупка"]], fake_lemmatizer())
        self.assertEqual("купить", synonyms.map_lemma("покупка"))

    def test_same_lemma_in_two_groups_raises(self):
        with self.assertRaises(SynonymError):
            SynonymMap([["купить", "купил"], ["приобрести", "куплю"]], fake_lemmatizer())

    def test_duplicate_keyword_forms_are_only_warning_and_merged(self):
        content = valid()
        content.questions[0].keywords = kw(("купить", 5), ("купил", 2), ("лицензия", 3), ("оплата", 1))
        lem = fake_lemmatizer()
        issues = validate(content, lemmatize=lem)
        self.assertTrue(any(i.code == "V008" and i.severity is Severity.WARNING for i in issues))
        self.assertFalse(any(i.severity is Severity.ERROR for i in issues))

        kb = KnowledgeBase.build(content, lem)
        words = {k.word: k.weight for k in kb.questions[0].keywords}
        self.assertEqual({"купить": 5, "лицензия": 3, "оплата": 1}, words)

    def test_duplicate_keyword_with_stemmer_is_still_error(self):
        content = valid()
        content.questions[0].keywords[2].word = "лицензию"
        issues = validate(content, lemmatize=stem)
        self.assertTrue(any(i.code == "V008" and i.severity is Severity.ERROR for i in issues))


class MorphEngineTests(unittest.TestCase):
    def test_engine_answers_with_morphology(self):
        kb = KnowledgeBase.build(valid(), fake_lemmatizer())
        engine = ChatEngine(kb, AppConfig())
        reply = engine.ask("купили лицензию")
        self.assertIs(ReplyKind.ANSWER, reply.kind)
        self.assertEqual("1.1.1", reply.question.id)
        self.assertEqual("pymorphy3", engine.morphology)


class ModeSelectionTests(unittest.TestCase):
    def test_stemmer_mode(self):
        lem, note = make_lemmatizer("stemmer")
        self.assertIs(stem, lem)
        self.assertIsNone(note)

    def test_unknown_mode_raises(self):
        with self.assertRaises(ValueError):
            make_lemmatizer("magic")

    def test_auto_falls_back_to_stemmer_without_library(self):
        try:
            import pymorphy3  # noqa: F401
        except ImportError:
            lem, note = make_lemmatizer("auto")
            self.assertIs(stem, lem)
            self.assertIn("не установлен", note)
            with self.assertRaises(MorphologyUnavailable):
                make_lemmatizer("pymorphy3")
        else:
            lem, _ = make_lemmatizer("auto")
            self.assertIsInstance(lem, MorphLemmatizer)


try:
    import pymorphy3  # noqa: F401
    HAVE_PYMORPHY = True
except ImportError:
    HAVE_PYMORPHY = False


@unittest.skipUnless(HAVE_PYMORPHY, "pymorphy3 не установлен")
class RealPymorphyTests(unittest.TestCase):
    def test_real_normal_forms(self):
        lem = MorphLemmatizer()
        self.assertEqual("лицензия", lem("лицензию"))
        self.assertEqual("купить", lem("купили"))

    def test_real_content_loads_with_pymorphy(self):
        engine = ChatEngine.create(CONTENT_DIR, "pymorphy3", semantic=False)
        self.assertEqual("pymorphy3", engine.morphology)
        self.assertEqual(38, len(engine.knowledge_base.questions))

    def test_acceptance_with_pymorphy(self):
        """Если этот тест падает, запустите  python run.py --compare  и пришлите вывод:
        по нему подбираются ключевые слова под морфологию."""
        failures, _, _ = check_acceptance(ChatEngine.create(CONTENT_DIR, "pymorphy3", semantic=False), CONTENT_DIR)
        self.assertEqual([], failures, "\n".join(failures))


if __name__ == "__main__":
    unittest.main()


class DayMoneyCollisionTests(unittest.TestCase):
    """С pymorphy3 «дней» → «день», а «день» — начало слова «деньги». Без группы синонимов
    «средства/деньги» запрос про пробный период уходил в вопрос о возврате денег."""

    def test_days_do_not_match_money(self):
        from unittest import mock
        from pathlib import Path
        import cryptochat.engine as engine_module
        import cryptochat.text as text
        from cryptochat.engine import ChatEngine
        forms = {"дней": "день", "дня": "день", "день": "день", "деньги": "деньги", "денег": "деньги",
                 "средства": "средство", "средств": "средство", "работает": "работать", "работать": "работать"}

        def lemmatize(word):
            return forms.get(word, text.stem(word))
        lemmatize.name = "имитация pymorphy3"
        original = text.make_lemmatizer
        with mock.patch.object(engine_module, "make_lemmatizer",
                               lambda mode="stemmer": (lemmatize, None) if mode == "test" else original(mode)):
            content = Path(__file__).resolve().parent.parent / "content"
            engine = ChatEngine.create(content, morphology="test")
            reply = engine.ask("сколько дней работает без лицензии")
        first = reply.question.id if reply.question else reply.options[0].question.id
        self.assertEqual("1.1.2.2", first)
