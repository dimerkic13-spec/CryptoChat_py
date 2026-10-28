import json
import tempfile
import unittest
from pathlib import Path

from cryptochat import knowledge
from cryptochat.cli import check_acceptance
from cryptochat.engine import ChatEngine
from cryptochat.knowledge import KnowledgeBase
from cryptochat.models import AppConfig, ContentError, ReplyKind

from .support import CONTENT_DIR, two_products, valid


def engine(content=None):
    return ChatEngine(KnowledgeBase.build(content or valid()), AppConfig())


class ChatEngineTests(unittest.TestCase):
    """Обычный режим (dialog.guided = false): поведение до шага 10."""

    def test_confident_question_gets_answer_with_source(self):
        reply = engine().ask("как купить лицензию")
        self.assertIs(ReplyKind.ANSWER, reply.kind)
        self.assertEqual("1.1.1", reply.question.id)
        self.assertIn("Текст ответа.", reply.text)
        self.assertIn("https://cryptopro.ru/faq/test#test", reply.text)
        self.assertIn("2026-10-09", reply.text)

    def test_weak_question_asks_to_confirm_then_shows_answer(self):
        e = engine()
        ask = e.ask("лицензия")
        self.assertIs(ReplyKind.CLARIFY, ask.kind)
        self.assertEqual(1, len(ask.options))
        answer = e.ask("1")
        self.assertIs(ReplyKind.ANSWER, answer.kind)
        self.assertEqual("1.1.1", answer.question.id)

    def test_zero_in_menu_escalates(self):
        e = engine()
        e.ask("лицензия")
        reply = e.ask("0")
        self.assertIs(ReplyKind.ESCALATE, reply.kind)
        self.assertIn("support.cryptopro.ru", reply.text)

    def test_other_input_in_menu_is_new_question(self):
        e = engine()
        e.ask("лицензия")
        reply = e.ask("кластер ngate")
        self.assertIs(ReplyKind.ANSWER, reply.kind)
        self.assertEqual("2.1.1", reply.question.id)

    def test_number_without_menu_is_not_understood(self):
        self.assertIs(ReplyKind.NOT_UNDERSTOOD, engine().ask("1").kind)

    def test_gibberish_escalates(self):
        self.assertIs(ReplyKind.ESCALATE, engine().ask("абракадабра").kind)

    def test_only_stop_words_is_not_understood(self):
        self.assertIs(ReplyKind.NOT_UNDERSTOOD, engine().ask("как и что").kind)

    def test_exit_words(self):
        for word in ("выход", "exit", "quit", "/выход"):
            with self.subTest(word=word):
                reply = engine().ask(word)
                self.assertIs(ReplyKind.EXIT, reply.kind)
                self.assertIn("До свидания", reply.text)

    def test_product_clarification_then_choice(self):
        e = engine(two_products())
        ask = e.ask("лицензия")
        self.assertIs(ReplyKind.CLARIFY_PRODUCT, ask.kind)
        self.assertEqual(2, len(ask.options))
        self.assertEqual("КриптоПро NGate", ask.options[1].label)
        answer = e.ask("2")
        self.assertIs(ReplyKind.ANSWER, answer.kind)
        self.assertEqual("2.2.1", answer.question.id)

    def test_debug_is_shown_only_when_enabled(self):
        e = engine()
        self.assertIsNone(e.ask("как купить лицензию").debug)
        self.assertIs(ReplyKind.COMMAND, e.ask("/отладка вкл").kind)
        debug = e.ask("как купить лицензию").debug
        self.assertIn("леммы", debug)
        self.assertIn("1.1.1", debug)
        e.ask("/отладка выкл")
        self.assertIsNone(e.ask("как купить лицензию").debug)

    def test_commands(self):
        e = engine()
        self.assertIn("/темы", e.ask("/помощь").text)
        tree = e.ask("/темы").text
        self.assertIn("1.1 Лицензии (1)", tree)
        self.assertIn("2.1 NGate (1)", tree)
        self.assertIn("Как купить лицензию?", e.ask("/темы 1.1").text)
        self.assertIn("не найден", e.ask("/темы 9.9").text)
        self.assertIn("Неизвестная команда", e.ask("/что-то").text)


def real_engine():
    # Логика диалога проверяется на стеммере: он есть всегда и даёт одинаковый результат везде.
    # Морфология pymorphy3 проверяется отдельно (test_morphology.py).
    return ChatEngine.create(CONTENT_DIR, "stemmer")


class GuidedFlowTests(unittest.TestCase):
    """Управляемый диалог на настоящей базе: уточнение по веткам, отзыв, перевод в поддержку."""

    def test_real_config_turns_on_guided_dialog(self):
        config = AppConfig.load(CONTENT_DIR)
        self.assertTrue(config.dialog.guided)
        self.assertEqual(4, config.dialog.escalate_after_steps)
        self.assertAlmostEqual(0.6, config.thresholds.answer)
        self.assertAlmostEqual(0.3, config.thresholds.clarify)
        self.assertEqual("auto", config.text.morphology)

    def test_unknown_question_asks_topic_block_question_then_answers(self):
        e = real_engine()
        topics = e.ask("абракадабра")
        self.assertIs(ReplyKind.CLARIFY_BRANCH, topics.kind)
        self.assertEqual(3, len(topics.options))
        self.assertEqual("1.1", topics.options[0].branch.id)
        self.assertTrue(all(o.branch.id.startswith("1.") for o in topics.options))
        self.assertIn("0. Ничего не подходит", topics.text)

        blocks = e.ask("1")
        self.assertIs(ReplyKind.CLARIFY_BRANCH, blocks.kind)
        self.assertEqual("1.1.1", blocks.options[0].branch.id)

        questions = e.ask("2")
        self.assertIs(ReplyKind.CLARIFY_BRANCH, questions.kind)
        self.assertEqual(3, len(questions.options))
        self.assertEqual("1.1.2.1", questions.options[0].question.id)

        answer = e.ask("1")
        self.assertIs(ReplyKind.ANSWER, answer.kind)
        self.assertEqual("1.1.2.1", answer.question.id)
        self.assertIn("0. Обратиться в техподдержку", answer.text)
        self.assertEqual(2, len(answer.options))

    def test_zero_goes_to_support_immediately(self):
        e = real_engine()
        e.ask("абракадабра")
        reply = e.ask("0")
        self.assertIs(ReplyKind.ESCALATE, reply.kind)
        self.assertIn("support.cryptopro.ru", reply.text)

    def test_support_only_after_several_steps(self):
        e = real_engine()
        for _ in range(4):
            self.assertIs(ReplyKind.CLARIFY_BRANCH, e.ask("абракадабра").kind)
        self.assertIs(ReplyKind.ESCALATE, e.ask("абракадабра").kind)
        # После перевода в поддержку счётчик начинается заново.
        self.assertIs(ReplyKind.CLARIFY_BRANCH, e.ask("абракадабра").kind)

    def test_confident_answer_has_similar_questions_feedback_and_support(self):
        reply = real_engine().ask("забыл логин и пароль от кабинета")
        self.assertIs(ReplyKind.ANSWER, reply.kind)
        self.assertEqual("1.3.3.2", reply.question.id)
        self.assertIn("Ответ помог?", reply.text)
        self.assertIn("0. Обратиться в техподдержку", reply.text)
        self.assertEqual("1.3.2.1", reply.options[0].question.id)

    def test_fitting_questions_then_answer_then_positive_feedback(self):
        e = real_engine()
        given = []
        e.feedback_listeners.append(given.append)

        menu = e.ask("Забыл пароль от рутокена")
        self.assertIs(ReplyKind.CLARIFY, menu.kind)
        self.assertEqual(["1.3.2.1", "1.3.3.2"], [o.question.id for o in menu.options])

        answer = e.ask("1")
        self.assertIs(ReplyKind.ANSWER, answer.kind)
        self.assertEqual("1.3.2.1", answer.question.id)

        self.assertIs(ReplyKind.FEEDBACK, e.ask("да").kind)
        self.assertEqual(1, len(given))
        self.assertTrue(given[0].helpful)
        self.assertEqual("1.3.2.1", given[0].question_id)
        self.assertEqual("Забыл пароль от рутокена", given[0].query)

    def test_negative_feedback_keeps_similar_questions(self):
        e = real_engine()
        given = []
        e.feedback_listeners.append(given.append)

        e.ask("забыл логин и пароль от кабинета")
        reply = e.ask("нет")
        self.assertIs(ReplyKind.FEEDBACK, reply.kind)
        self.assertIn("0. Обратиться в техподдержку", reply.text)
        self.assertEqual(1, len(reply.options))
        self.assertFalse(given[0].helpful)

        nxt = e.ask("1")
        self.assertIs(ReplyKind.ANSWER, nxt.kind)
        self.assertEqual("1.3.2.1", nxt.question.id)

    def test_yes_without_answer_is_ordinary_question(self):
        e = real_engine()
        given = []
        e.feedback_listeners.append(given.append)
        e.ask("да")
        self.assertEqual([], given)


class AcceptanceTests(unittest.TestCase):
    """Настоящие вопросы из content/acceptance.json и заголовки всех вопросов базы."""

    def test_all_cases_pass(self):
        failures, checked, cases = check_acceptance(real_engine(), CONTENT_DIR)
        self.assertEqual([], failures, "\n".join(failures))
        self.assertGreaterEqual(cases, 40)


class ConfigTests(unittest.TestCase):
    def test_defaults_without_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = AppConfig.load(tmp)
        self.assertFalse(config.dialog.guided)
        self.assertEqual(3, config.menu.max_options)

    def test_camel_case_keys_are_read(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps(
                {"thresholds": {"dominanceGap": 0.3}, "dialog": {"escalateAfterSteps": 2}}), encoding="utf-8")
            config = AppConfig.load(tmp)
        self.assertAlmostEqual(0.3, config.thresholds.dominance_gap)
        self.assertEqual(2, config.dialog.escalate_after_steps)

    def test_bad_thresholds_raise(self):
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps(
                {"thresholds": {"answer": 0.2, "clarify": 0.5}}), encoding="utf-8")
            with self.assertRaises(ContentError):
                AppConfig.load(tmp)


if __name__ == "__main__":
    unittest.main()
