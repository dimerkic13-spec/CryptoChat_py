import unittest

from cryptochat.dialog import DecisionKind, DialogPolicy
from cryptochat.knowledge import KnowledgeBase
from cryptochat.models import AppConfig, DialogConfig
from cryptochat.retrieval import Candidate, RetrievalResult

from .support import two_products

KB = KnowledgeBase.build(two_products())
SCOPE = KB.categories["2.1"]


def c(index, score):
    return Candidate(KB.questions[index], score)


def r(scope, *candidates, lemmas=("x",)):
    return RetrievalResult(tuple(lemmas), scope, 0.0, tuple(candidates))


class DialogPolicyTests(unittest.TestCase):
    policy = DialogPolicy(AppConfig(), KB)

    def test_no_lemmas_is_not_understood(self):
        self.assertIs(DecisionKind.NOT_UNDERSTOOD, self.policy.decide(r(None, lemmas=())).kind)

    def test_no_candidates_is_escalate(self):
        self.assertIs(DecisionKind.ESCALATE, self.policy.decide(r(None)).kind)

    def test_weak_top_score_is_escalate(self):
        self.assertIs(DecisionKind.ESCALATE, self.policy.decide(r(SCOPE, c(0, 0.25))).kind)

    def test_medium_score_is_clarify_with_one_option(self):
        decision = self.policy.decide(r(SCOPE, c(0, 0.45)))
        self.assertIs(DecisionKind.CLARIFY, decision.kind)
        self.assertEqual(1, len(decision.candidates))

    def test_clarify_lists_only_candidates_above_threshold(self):
        decision = self.policy.decide(r(SCOPE, c(0, 0.55), c(1, 0.35), c(0, 0.2)))
        self.assertIs(DecisionKind.CLARIFY, decision.kind)
        self.assertEqual(2, len(decision.candidates))

    def test_strong_single_candidate_is_answer(self):
        decision = self.policy.decide(r(SCOPE, c(0, 0.7)))
        self.assertIs(DecisionKind.ANSWER, decision.kind)
        self.assertEqual(KB.questions[0].id, decision.candidates[0].question.id)

    def test_two_close_strong_candidates_is_clarify(self):
        decision = self.policy.decide(r(SCOPE, c(0, 0.7), c(1, 0.65)))
        self.assertIs(DecisionKind.CLARIFY, decision.kind)
        self.assertEqual(2, len(decision.candidates))

    def test_clear_leader_even_with_second_strong_is_answer(self):
        self.assertIs(DecisionKind.ANSWER, self.policy.decide(r(SCOPE, c(0, 0.85), c(1, 0.6))).kind)

    def test_leaders_from_different_products_without_scope_ask_product(self):
        decision = self.policy.decide(r(None, c(0, 0.55), c(1, 0.55)))
        self.assertIs(DecisionKind.CLARIFY_PRODUCT, decision.kind)
        self.assertEqual(["2.1", "2.2"], [p.id for p in decision.products])

    def test_leaders_from_different_products_with_scope_is_plain_clarify(self):
        self.assertIs(DecisionKind.CLARIFY, self.policy.decide(r(SCOPE, c(0, 0.55), c(1, 0.55))).kind)


class GuidedPolicyTests(unittest.TestCase):
    policy = DialogPolicy(AppConfig(dialog=DialogConfig(guided=True)), KB)

    def test_no_candidates_clarifies_by_branch(self):
        self.assertIs(DecisionKind.CLARIFY_BRANCH, self.policy.decide(r(SCOPE)).kind)

    def test_weak_top_clarifies_by_branch(self):
        self.assertIs(DecisionKind.CLARIFY_BRANCH, self.policy.decide(r(SCOPE, c(0, 0.25))).kind)

    def test_one_to_three_fitting_questions_are_offered(self):
        decision = self.policy.decide(r(SCOPE, c(0, 0.45), c(1, 0.4)))
        self.assertIs(DecisionKind.CLARIFY, decision.kind)
        self.assertEqual(2, len(decision.candidates))

    def test_more_than_three_fitting_clarifies_by_branch(self):
        decision = self.policy.decide(r(SCOPE, c(0, 0.5), c(1, 0.45), c(0, 0.4), c(1, 0.35)))
        self.assertIs(DecisionKind.CLARIFY_BRANCH, decision.kind)

    def test_single_strong_is_answered(self):
        self.assertIs(DecisionKind.ANSWER, self.policy.decide(r(SCOPE, c(0, 0.8), c(1, 0.3))).kind)


if __name__ == "__main__":
    unittest.main()
