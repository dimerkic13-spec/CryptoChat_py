"""Учёт опечаток (RapidFuzz или difflib) и база бесед SQLite."""
import sqlite3
import tempfile
from contextlib import closing
import unittest
from datetime import datetime
from pathlib import Path

from cryptochat.cli import Conversation, check_acceptance, database_path
from cryptochat.engine import ChatEngine
from cryptochat.fuzzy import FuzzyMatcher, make_fuzzy
from cryptochat.knowledge import KnowledgeBase
from cryptochat.models import AppConfig, ChatReply, FeedbackEntry, FuzzyConfig, Keyword, ReplyKind
from cryptochat.retrieval import evaluate, typo_matches
from cryptochat.storage import ChatDatabase, stats_report

from .support import CONTENT_DIR, valid


class FuzzyMatcherTests(unittest.TestCase):
    fuzzy = FuzzyMatcher(threshold=85, min_length=5, weight=0.8)

    def test_typos_are_similar(self):
        for typo, word in [("рутокин", "рутокен"), ("кабинед", "кабинет"), ("дубликт", "дубликат"),
                           ("лицензя", "лицензи")]:
            with self.subTest(typo=typo):
                self.assertTrue(self.fuzzy.matches(typo, word))

    def test_different_words_are_not_similar(self):
        for a, b in [("договор", "разговор"), ("кабинет", "кластер"), ("пароль", "портал")]:
            with self.subTest(a=a, b=b):
                self.assertFalse(self.fuzzy.matches(a, b))

    def test_same_root_words_are_linked(self):
        # «оплата» и «оплатить» — одно слово по смыслу; со стеммером они и так совпадали.
        self.assertTrue(self.fuzzy.matches("оплата", "оплатить"))

    def test_short_words_are_never_fuzzy(self):
        self.assertFalse(self.fuzzy.matches("пинг", "пин"))
        self.assertFalse(FuzzyMatcher(threshold=50, min_length=5).matches("ключ", "клюв"))

    def test_typo_gives_reduced_weight(self):
        keywords = [Keyword("рутокен", 6), Keyword("пароль", 4)]
        exact, _ = evaluate(keywords, ["рутокен"], 4, self.fuzzy)
        typo, matched = evaluate(keywords, ["рутокин"], 4, self.fuzzy)
        self.assertAlmostEqual(0.6, exact)
        self.assertAlmostEqual(0.6 * 0.8, typo)
        self.assertEqual(1, matched)
        self.assertEqual(0.0, evaluate(keywords, ["рутокин"], 4, None)[0])
        self.assertEqual([("рутокин", "рутокен")], typo_matches(keywords, ["рутокин"], 4, self.fuzzy))

    def test_disabled_in_config_gives_none(self):
        self.assertIsNone(make_fuzzy(FuzzyConfig(enabled=False)))
        self.assertIsNotNone(make_fuzzy(FuzzyConfig(enabled=True)))


class FuzzyEngineTests(unittest.TestCase):
    def test_typo_in_query_still_finds_question(self):
        config = AppConfig()
        config.fuzzy.enabled = True
        engine = ChatEngine(KnowledgeBase.build(valid()), config)
        reply = engine.ask("класстер ngate")
        self.assertIs(ReplyKind.ANSWER, reply.kind)
        self.assertEqual("2.1.1", reply.question.id)

    def test_real_content_with_fuzzy_passes_acceptance(self):
        engine = ChatEngine.create(CONTENT_DIR, "stemmer", semantic=False)
        self.assertTrue(engine.config.fuzzy.enabled, "в content/config.json учёт опечаток включён")
        failures, _, _ = check_acceptance(engine, CONTENT_DIR)
        self.assertEqual([], failures, "\n".join(failures))

    def test_real_typos_find_right_question(self):
        cases = {"забыл пароль от рутокина": "1.3.2.1", "забыл пароль от кабинед": "1.3.3.2",
                 "нужен дубликт лицензии": "1.1.4.2", "присоеденится к эдо": "1.2.4.1"}
        for query, expected in cases.items():
            with self.subTest(query=query):
                reply = ChatEngine.create(CONTENT_DIR, "stemmer", semantic=False).ask(query)
                first = reply.question.id if reply.question else reply.options[0].question.id
                self.assertEqual(expected, first)


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.path = Path(self._tmp.name) / "logs" / "cryptochat.db"

    def tearDown(self):
        self._tmp.cleanup()

    def query(self, sql):
        # closing(...) обязательно: «with sqlite3.connect()» не закрывает файл, и Windows не даст удалить папку.
        with closing(sqlite3.connect(self.path)) as conn:
            return conn.execute(sql).fetchall()

    def test_writes_session_messages_feedback_unanswered(self):
        db = ChatDatabase(self.path)
        self.assertTrue(db.available)
        db.start_session("s1", "pc", "0.1a", "стеммер")
        db.user_message("s1", "как продлить лицензию")
        question = KnowledgeBase.build(valid()).questions[0]
        db.bot_message("s1", ChatReply(ReplyKind.ANSWER, "Ответ", question=question, score=0.7))
        db.unanswered("s1", "абракадабра", ReplyKind.CLARIFY_BRANCH)
        db.unanswered("s1", "как купить лицензию", ReplyKind.ANSWER)  # ответ — не пишется
        db.feedback("s1", FeedbackEntry(datetime.now(), "как продлить лицензию", "1.1.1", False))
        db.finish_session("s1")
        db.mark_emailed("s1")
        db.close()

        self.assertEqual([("s1", 1)], self.query("SELECT id, messages FROM sessions"))
        self.assertEqual([("user", None, None), ("bot", "1.1.1", 0.7)],
                         self.query("SELECT role, question_id, score FROM messages ORDER BY id"))
        self.assertEqual([("абракадабра", "ClarifyBranch")], self.query("SELECT query, kind FROM unanswered"))
        self.assertEqual([("1.1.1", 0)], self.query("SELECT question_id, helpful FROM feedback"))
        self.assertIsNotNone(self.query("SELECT finished, emailed FROM sessions")[0][1])

    def test_stats_report(self):
        db = ChatDatabase(self.path)
        db.start_session("s1", "pc", "0.1a", "стеммер")
        question = KnowledgeBase.build(valid()).questions[0]
        db.bot_message("s1", ChatReply(ReplyKind.ANSWER, "Ответ", question=question, score=0.7))
        db.feedback("s1", FeedbackEntry(datetime.now(), "купить", "1.1.1", False))
        db.unanswered("s1", "Абракадабра", ReplyKind.ESCALATE)
        db.unanswered("s1", "абракадабра ", ReplyKind.CLARIFY_BRANCH)
        db.close()

        report = stats_report(self.path, {"1.1.1": "Как купить лицензию?"})
        self.assertIn("Бесед: 1", report)
        self.assertIn("Как купить лицензию?", report)
        self.assertIn("не помогло: 1", report)
        self.assertIn("повторов: 2", report)  # одинаковые запросы с разным регистром сложены
        self.assertIn("«купить»", report)

    def test_stats_without_database(self):
        self.assertIn("ещё не создана", stats_report(self.path))

    def test_unwritable_path_does_not_raise(self):
        blocker = Path(self._tmp.name) / "file"
        blocker.write_text("x")
        db = ChatDatabase(blocker / "sub" / "cryptochat.db")
        self.assertFalse(db.available)
        db.user_message("s1", "ничего не сломается")  # просто ничего не делает


class ConversationTests(unittest.TestCase):
    def test_conversation_writes_file_and_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            engine = ChatEngine.create(CONTENT_DIR, "stemmer", semantic=False)
            engine.config.sessions.path = str(Path(tmp) / "sessions")
            engine.config.database.path = str(Path(tmp) / "cryptochat.db")
            engine.config.email.enabled = False

            conversation = Conversation(engine)
            for line in ("забыл логин и пароль от кабинета", "да", "абракадабра", "выход"):
                conversation.user(line)
                conversation.bot(line, engine.ask(line))
            conversation.close()

            sid = conversation.session_id
            self.assertTrue((Path(tmp) / "sessions" / f"{sid}.txt").exists())
            with closing(sqlite3.connect(database_path(engine.config))) as conn:
                self.assertEqual(3, conn.execute("SELECT messages FROM sessions").fetchone()[0])
                self.assertEqual(1, conn.execute("SELECT COUNT(*) FROM feedback WHERE helpful = 1").fetchone()[0])
                self.assertEqual("абракадабра", conn.execute("SELECT query FROM unanswered").fetchone()[0])


if __name__ == "__main__":
    unittest.main()


try:
    import rapidfuzz  # noqa: F401
    HAVE_RAPIDFUZZ = True
except ImportError:
    HAVE_RAPIDFUZZ = False


@unittest.skipUnless(HAVE_RAPIDFUZZ, "RapidFuzz не установлен")
class RealRapidFuzzTests(unittest.TestCase):
    def test_backend_is_rapidfuzz_and_finds_typos(self):
        fuzzy = FuzzyMatcher()
        self.assertEqual("RapidFuzz", fuzzy.name)
        self.assertTrue(fuzzy.matches("рутокин", "рутокен"))
        self.assertFalse(fuzzy.matches("договор", "разговор"))
