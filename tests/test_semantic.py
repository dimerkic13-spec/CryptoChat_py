"""Поиск по смыслу: проверка на поддельной модели (без PyTorch) и, по желанию, на настоящей."""
import io
import os
import unittest
from contextlib import redirect_stdout
from pathlib import Path

from cryptochat import cli, semantic as semantic_module
from cryptochat.engine import ChatEngine
from cryptochat.knowledge import KnowledgeBase
from cryptochat.models import AppConfig, ReplyKind, SemanticConfig
from cryptochat.semantic import (SemanticIndex, SemanticUnavailable, SentenceTransformerEncoder, make_semantic)

from .support import CONTENT_DIR, q, valid

# «Смыслы» для поддельной модели: фразы с любым словом из группы получают одно и то же направление.
CONCEPTS = [
    ("кабинет", "аккаунт", "зайти", "войти", "логин"),
    ("купить", "покупк", "приобрест", "лицензи"),
    ("кластер", "ngate", "узел", "отказоустойчив"),
    ("пароль",),
    ("сбой", "ошибк"),
]


class FakeEncoder:
    """Ведёт себя как модель: близкие по смыслу фразы дают близкие векторы."""
    name = "поддельная модель"

    def __init__(self, *args, **kwargs):
        self.calls = 0

    def encode(self, texts):
        self.calls += 1
        vectors = []
        for text in texts:
            low = text.lower()
            vector = [1.0 if any(w in low for w in group) else 0.0 for group in CONCEPTS]
            vector.append(0.05)  # чтобы вектор не был нулевым
            vectors.append(vector)
        return vectors


def content_with_cabinet():
    content = valid()
    content.questions.append(q("1.1.2", "1.1", "Забыл логин от личного кабинета",
                               ("кабинет", 6), ("логин", 4), ("забыл", 2)))
    return content


class SemanticIndexTests(unittest.TestCase):
    def setUp(self):
        self.kb = KnowledgeBase.build(content_with_cabinet())
        self.index = SemanticIndex(FakeEncoder(), self.kb.questions, low=0.5, high=0.9, weight=0.8)

    def test_score_mapping(self):
        self.assertEqual(0.0, self.index.to_score(0.5))
        self.assertAlmostEqual(0.4, self.index.to_score(0.7))
        self.assertAlmostEqual(0.8, self.index.to_score(0.95))

    def test_paraphrase_without_common_words_is_close(self):
        cos = self.index.similarities("не могу зайти в аккаунт")
        self.assertEqual("1.1.2", max(cos, key=cos.get))
        self.assertGreater(cos["1.1.2"], 0.9)

    def test_examples_are_indexed(self):
        content = content_with_cabinet()
        content.questions[1].examples = ["отказоустойчивость нескольких серверов"]
        index = SemanticIndex(FakeEncoder(), KnowledgeBase.build(content).questions)
        self.assertEqual(4, index.phrases)  # 3 заголовка + 1 пример

    def test_last_query_is_cached(self):
        encoder = FakeEncoder()
        index = SemanticIndex(encoder, self.kb.questions)
        index.similarities("зайти в аккаунт"); index.similarities("зайти в аккаунт")
        self.assertEqual(2, encoder.calls)  # 1 раз для вопросов + 1 раз для запроса


class SemanticEngineTests(unittest.TestCase):
    def engine(self, weight=0.8):
        kb = KnowledgeBase.build(content_with_cabinet())
        index = SemanticIndex(FakeEncoder(), kb.questions, low=0.5, high=0.9, weight=weight)
        return ChatEngine(kb, AppConfig(), semantic=index)

    def test_finds_question_with_no_common_words(self):
        plain = ChatEngine(KnowledgeBase.build(content_with_cabinet()), AppConfig())
        self.assertIs(ReplyKind.ESCALATE, plain.ask("не могу зайти в аккаунт").kind)

        reply = self.engine().ask("не могу зайти в аккаунт")
        self.assertIs(ReplyKind.ANSWER, reply.kind)
        self.assertEqual("1.1.2", reply.question.id)

    def test_low_weight_only_suggests(self):
        reply = self.engine(weight=0.5).ask("не могу зайти в аккаунт")
        self.assertIs(ReplyKind.CLARIFY, reply.kind)
        self.assertEqual("1.1.2", reply.options[0].question.id)

    def test_keywords_still_win_when_stronger(self):
        reply = self.engine().ask("кластер ngate")
        self.assertIs(ReplyKind.ANSWER, reply.kind)
        self.assertEqual("2.1.1", reply.question.id)

    def test_debug_shows_similarity(self):
        engine = self.engine()
        engine.debug_enabled = True
        debug = engine.ask("не могу зайти в аккаунт").debug
        self.assertIn("смысл: похожесть", debug)
        self.assertIn("смысл: поддельная модель", debug)


class MakeSemanticTests(unittest.TestCase):
    def test_disabled(self):
        self.assertEqual((None, None), make_semantic(SemanticConfig(enabled=False), []))

    def test_unavailable_gives_note(self):
        def broken(*args, **kwargs):
            raise SemanticUnavailable("нет библиотеки")
        index, note = make_semantic(SemanticConfig(enabled=True), [], encoder_factory=broken)
        self.assertIsNone(index)
        self.assertEqual("нет библиотеки", note)

    def test_cache_dir_is_relative_to_app_dir(self):
        seen = {}

        def factory(model, cache_dir=None, device="cpu", local_files_only=False):
            seen.update(model=model, cache_dir=cache_dir, device=device)
            return FakeEncoder()
        index, note = make_semantic(SemanticConfig(enabled=True), [], app_dir=Path("/app"), encoder_factory=factory)
        self.assertIsNotNone(index)
        self.assertEqual(str(Path("/app") / "models"), seen["cache_dir"])
        self.assertEqual("cointegrated/rubert-tiny2", seen["model"])

    def test_real_encoder_without_library_is_unavailable(self):
        try:
            import sentence_transformers  # noqa: F401
        except ImportError:
            with self.assertRaises(SemanticUnavailable):
                SentenceTransformerEncoder("cointegrated/rubert-tiny2")


class RealContentSemanticTests(unittest.TestCase):
    def test_config_has_semantic_block(self):
        config = AppConfig.load(CONTENT_DIR)
        self.assertTrue(config.semantic.enabled)
        self.assertEqual("cointegrated/rubert-tiny2", config.semantic.model)

    def test_bot_works_when_library_missing(self):
        def broken(*args, **kwargs):
            raise SemanticUnavailable("нет библиотеки")
        original = semantic_module.SentenceTransformerEncoder
        semantic_module.SentenceTransformerEncoder = broken
        try:
            engine = ChatEngine.create(CONTENT_DIR, "stemmer")
        finally:
            semantic_module.SentenceTransformerEncoder = original
        self.assertIsNone(engine.semantic)
        self.assertEqual("нет библиотеки", engine.semantic_note)
        self.assertIs(ReplyKind.ANSWER, engine.ask("забыл логин и пароль от кабинета").kind)

    def test_compare_semantic_runs_with_fake_model(self):
        original = semantic_module.SentenceTransformerEncoder
        semantic_module.SentenceTransformerEncoder = FakeEncoder
        try:
            out = io.StringIO()
            with redirect_stdout(out):
                code = cli.run_compare_semantic(CONTENT_DIR, "stemmer")
        finally:
            semantic_module.SentenceTransformerEncoder = original
        self.assertEqual(0, code)
        text = out.getvalue()
        self.assertIn("paraphrases.json", text)
        self.assertIn("acceptance.json", text)


@unittest.skipUnless(os.environ.get("CRYPTOCHAT_TEST_SEMANTIC") == "1",
                     "проверка настоящей модели: задайте CRYPTOCHAT_TEST_SEMANTIC=1")
class RealModelTests(unittest.TestCase):
    """Скачивает и загружает настоящую модель. Запуск: set CRYPTOCHAT_TEST_SEMANTIC=1 и python -m unittest."""

    def test_real_model_finds_paraphrase(self):
        engine = ChatEngine.create(CONTENT_DIR, semantic=True, app_dir=cli.app_directory())
        self.assertIsNotNone(engine.semantic, engine.semantic_note)
        cos = engine.semantic.similarities("не могу зайти в аккаунт")
        top = sorted(cos, key=cos.get, reverse=True)[:3]
        self.assertIn("1.3.3.2", top)


if __name__ == "__main__":
    unittest.main()


class RelativeModeTests(unittest.TestCase):
    """Относительная оценка: вопрос должен выделяться на фоне остальных."""

    class TableEncoder:
        """Модель с заранее заданными векторами: похожесть запроса с вопросом i — values[i]."""
        name = "таблица"

        def __init__(self, values):
            self.values = values

        def encode(self, texts):
            # Вектор вопроса i — единичный по оси i; вектор запроса — values по осям (нормализуется в индексе).
            if len(texts) == len(self.values):
                return [[1.0 if j == i else 0.0 for j in range(len(texts))] for i in range(len(texts))]
            return [list(self.values)]

    def index(self, values, **kwargs):
        content = valid()
        for n in range(len(values) - 2):
            content.questions.append(q(f"1.1.{n + 2}", "1.1", f"Вопрос {n}", ("слово", 1), ("ещё", 1), ("три", 1)))
        kb = KnowledgeBase.build(content)
        return SemanticIndex(self.TableEncoder(values), kb.questions, mode="relative", weight=0.5, **kwargs)

    def test_flat_similarities_give_nothing(self):
        # Все вопросы одинаково «похожи» — никто не выделяется, смысловой оценки нет.
        self.assertEqual({}, self.index([0.8] * 10).scores("запрос"))

    def test_standout_question_gets_score(self):
        values = [0.3] * 9 + [0.9]
        scores = self.index(values, z_low=1.0, z_high=2.0).scores("запрос")
        self.assertEqual(1, len(scores))
        self.assertAlmostEqual(0.5, list(scores.values())[0])

    def test_only_top_k_considered(self):
        values = [0.1] * 6 + [0.9, 0.9, 0.9, 0.9]
        scores = self.index(values, z_low=0.5, z_high=1.0, top_k=2).scores("запрос")
        self.assertEqual(2, len(scores))

    def test_prefixes_are_added(self):
        seen = []

        class Recorder(FakeEncoder):
            def encode(self, texts):
                seen.extend(texts)
                return super().encode(texts)
        kb = KnowledgeBase.build(valid())
        index = SemanticIndex(Recorder(), kb.questions, query_prefix="query: ", passage_prefix="passage: ")
        index.similarities("вопрос")
        self.assertTrue(all(t.startswith("passage: ") for t in seen[:-1]))
        self.assertEqual("query: вопрос", seen[-1])


class CombineTests(unittest.TestCase):
    def test_modes(self):
        from cryptochat.retrieval import combine_scores
        self.assertAlmostEqual(0.5, combine_scores(0.5, 0.3, "max"))
        self.assertAlmostEqual(0.65, combine_scores(0.5, 0.3, "or"))
        self.assertAlmostEqual(0.3, combine_scores(0.0, 0.3, "or"))

    def test_bad_mode_in_config_is_error(self):
        import json, tempfile
        from cryptochat.models import ContentError
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({"semantic": {"mode": "magic"}}), encoding="utf-8")
            with self.assertRaises(ContentError):
                AppConfig.load(tmp)


class SemanticDumpTests(unittest.TestCase):
    def test_dump_writes_file(self):
        import json, tempfile
        original = semantic_module.SentenceTransformerEncoder
        semantic_module.SentenceTransformerEncoder = FakeEncoder
        cwd = os.getcwd()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                os.chdir(tmp)
                out = io.StringIO()
                with redirect_stdout(out):
                    code = cli.run_semantic_dump(CONTENT_DIR, "stemmer")
                data = json.loads(Path(tmp, "semantic_dump.json").read_text(encoding="utf-8"))
                os.chdir(cwd)
        finally:
            os.chdir(cwd)
            semantic_module.SentenceTransformerEncoder = original
        self.assertEqual(0, code)
        self.assertEqual(38, len(data["cases"][0]["cosine"]))
        self.assertEqual(38, len(data["cases"][0]["keyword"]))
        self.assertIn("правильный вопрос первый по смыслу", out.getvalue())


class SemanticMenuFallbackTests(unittest.TestCase):
    """Ключи ничего не нашли — бот показывает похожие по смыслу вопросы вместо меню тем."""

    def engine(self, menu_fallback=True, menu_z=1.0):
        content = content_with_cabinet()
        # несколько «посторонних» вопросов, чтобы было на фоне чего выделяться
        for n, title in enumerate(["Как получить счёт", "Как вернуть деньги", "Нужен договор", "Где акт сверки"]):
            content.questions.append(q(f"1.1.{n + 3}", "1.1", title, ("счет", 1), ("договор", 1), ("акт", 1)))
        kb = KnowledgeBase.build(content)
        config = AppConfig()
        config.dialog.guided = True
        config.semantic.menu_fallback = menu_fallback
        config.semantic.menu_z = menu_z
        # weight маленький: сам по себе смысл не дотягивает до порога 0.3, работает только запасное меню
        index = SemanticIndex(FakeEncoder(), kb.questions, mode="relative", z_low=0.0, z_high=5.0, weight=0.1)
        return ChatEngine(kb, config, semantic=index)

    def test_menu_instead_of_topics(self):
        reply = self.engine().ask("не могу зайти в аккаунт")
        self.assertIs(ReplyKind.CLARIFY, reply.kind)
        self.assertEqual("1.1.2", reply.options[0].question.id)
        self.assertIn("Возможно, вы имели в виду", reply.text)

    def test_choice_from_menu_gives_answer(self):
        engine = self.engine()
        engine.ask("не могу зайти в аккаунт")
        answer = engine.ask("1")
        self.assertIs(ReplyKind.ANSWER, answer.kind)
        self.assertEqual("1.1.2", answer.question.id)

    def test_disabled_fallback_goes_to_topics(self):
        reply = self.engine(menu_fallback=False).ask("не могу зайти в аккаунт")
        self.assertIs(ReplyKind.CLARIFY_BRANCH, reply.kind)

    def test_nothing_stands_out_goes_to_topics(self):
        reply = self.engine(menu_z=10).ask("не могу зайти в аккаунт")
        self.assertIs(ReplyKind.CLARIFY_BRANCH, reply.kind)

    def test_menu_counts_as_clarification_step(self):
        engine = self.engine()
        for _ in range(4):
            self.assertIs(ReplyKind.CLARIFY, engine.ask("не могу зайти в аккаунт").kind)
        self.assertIs(ReplyKind.ESCALATE, engine.ask("не могу зайти в аккаунт").kind)


class CalibratedConfigTests(unittest.TestCase):
    def test_real_config_has_calibrated_values(self):
        s = AppConfig.load(CONTENT_DIR).semantic
        self.assertEqual(("relative", "or"), (s.mode, s.combine))
        self.assertAlmostEqual(0.3, s.weight)
        self.assertAlmostEqual(0.0, s.z_low)
        self.assertAlmostEqual(5.0, s.z_high)
        self.assertTrue(s.menu_fallback)
        self.assertAlmostEqual(1.5, s.menu_z)


class CalibrationRegressionTests(unittest.TestCase):
    """Похожести настоящей модели rubert-tiny2 (выгрузка с компьютера пользователя) + настройки из config.json.
    Если тест упал после правки базы или кода — поиск по смыслу стал работать хуже, чем при калибровке."""

    @classmethod
    def setUpClass(cls):
        import json
        data = json.loads((Path(__file__).parent / "data" / "semantic_calibration.json").read_text(encoding="utf-8"))
        cls.cosines = data["cosines"]
        base = ChatEngine.create(CONTENT_DIR, "stemmer", semantic=False)
        cfg = base.config.semantic
        cfg.enabled = True

        class Table:
            name = "rubert-tiny2 (записанные похожести)"

            def encode(self, texts):
                return [[1.0] for _ in texts]

        index = SemanticIndex(Table(), base.knowledge_base.questions, cfg.low, cfg.high, cfg.weight,
                              mode=cfg.mode, z_low=cfg.z_low, z_high=cfg.z_high, top_k=cfg.top_k)
        index.similarities = lambda query: cls.cosines.get((query or "").strip(), {})
        cls.base, cls.index = base, index

    def fresh(self):
        return ChatEngine(self.base.knowledge_base, self.base.config, semantic=self.index)

    def test_acceptance_still_passes_with_model(self):
        engine = self.fresh()
        failures, _, _ = cli.check_acceptance(engine, CONTENT_DIR)
        self.assertEqual([], failures, "\n".join(failures))

    def test_paraphrases_improve(self):
        import json
        cases = json.loads((CONTENT_DIR / "paraphrases.json").read_text(encoding="utf-8"))
        first = in_menu = 0
        for case in cases:
            reply = self.fresh().ask(case["q"])
            ids = [reply.question.id] if reply.kind is ReplyKind.ANSWER else [o.question.id for o in reply.options if o.question]
            first += bool(ids) and ids[0] == case["expect"]
            in_menu += case["expect"] in ids
        self.assertGreaterEqual(first, 6)
        self.assertGreaterEqual(in_menu, 8)
