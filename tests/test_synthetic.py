"""Качество на content/synthetic.json (1900 запросов, по 50 на вопрос). Пороги — чуть ниже достигнутого в 0.2.6a:
если тест упал после правки базы, посмотрите таблицу по вопросам (ключ --synthetic) и верните качество."""
import unittest

from cryptochat.cli import (OUTCOME_ANSWER, OUTCOME_FIRST, OUTCOME_WRONG, evaluate_dialogs, evaluate_synthetic,
                            load_synthetic)
from cryptochat.engine import ChatEngine

from .support import CONTENT_DIR


def measure(morphology: str) -> tuple[float, int, int]:
    engine = ChatEngine.create(CONTENT_DIR, morphology, semantic=False)
    results = evaluate_synthetic(engine, load_synthetic(CONTENT_DIR))
    first = sum(1 for r in results if r[2] in (OUTCOME_ANSWER, OUTCOME_FIRST))
    wrong = sum(1 for r in results if r[2] == OUTCOME_WRONG)
    return first / len(results), wrong, len(results)


def has_pymorphy3() -> bool:
    try:
        import pymorphy3  # noqa: F401
        return True
    except ImportError:
        return False


class SyntheticQualityTests(unittest.TestCase):
    def test_file_has_50_queries_for_every_question(self):
        cases = load_synthetic(CONTENT_DIR)
        ids = [q.id for q in ChatEngine.create(CONTENT_DIR, "stemmer", semantic=False).knowledge_base.questions]
        self.assertEqual(sorted(ids), sorted(cases))
        for queries in cases.values():
            self.assertEqual(50, len(set(queries)))

    def check(self, morphology: str, minimum: float):
        share, wrong, total = measure(morphology)
        self.assertGreaterEqual(share, minimum, f"верный вопрос первым: {share:.1%} из {total}")
        self.assertLessEqual(wrong, 1, f"неверных прямых ответов: {wrong}")

    def test_stemmer_right_question_comes_first(self):
        self.check("stemmer", 0.85)  # 0.2.5a: 0.756, 0.2.6a: 0.843, 0.2.7a: 0.865

    @unittest.skipUnless(has_pymorphy3(), "pymorphy3 не установлен")
    def test_pymorphy3_right_question_comes_first(self):
        self.check("pymorphy3", 0.85)  # 0.2.5a: 0.755, 0.2.6a: 0.852, 0.2.7a: 0.868

    def check_dialogs(self, morphology: str, minimum: float):
        """Пользователь выбирает подходящий пункт меню: верный ответ не больше чем за 3 уточнения."""
        results = evaluate_dialogs(ChatEngine.create(CONTENT_DIR, morphology, semantic=False), load_synthetic(CONTENT_DIR))
        within3 = sum(1 for r in results if r[2] == "answer" and r[3] <= 3) / len(results)
        wrong = sum(1 for r in results if r[2] == "wrong")
        self.assertGreaterEqual(within3, minimum, f"верный ответ за ≤3 уточнения: {within3:.1%}")
        self.assertLessEqual(wrong, 1, f"неверных ответов: {wrong}")

    def test_stemmer_dialog_reaches_answer_within_3_steps(self):
        self.check_dialogs("stemmer", 0.96)  # 0.2.6a: 0.931, 0.2.7a: 0.974

    @unittest.skipUnless(has_pymorphy3(), "pymorphy3 не установлен")
    def test_pymorphy3_dialog_reaches_answer_within_3_steps(self):
        self.check_dialogs("pymorphy3", 0.97)  # 0.2.6a: 0.932, 0.2.7a: 0.979


if __name__ == "__main__":
    unittest.main()
