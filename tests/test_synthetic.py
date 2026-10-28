"""Качество на content/synthetic.json (1900 запросов, по 50 на вопрос). Пороги — чуть ниже достигнутого в 0.2.6a:
если тест упал после правки базы, посмотрите таблицу по вопросам (ключ --synthetic) и верните качество."""
import unittest

from cryptochat.cli import OUTCOME_ANSWER, OUTCOME_FIRST, OUTCOME_WRONG, evaluate_synthetic, load_synthetic
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
        self.check("stemmer", 0.83)  # 0.2.5a: 0.756, 0.2.6a: 0.843

    @unittest.skipUnless(has_pymorphy3(), "pymorphy3 не установлен")
    def test_pymorphy3_right_question_comes_first(self):
        self.check("pymorphy3", 0.84)  # 0.2.5a: 0.755, 0.2.6a: 0.852


if __name__ == "__main__":
    unittest.main()
