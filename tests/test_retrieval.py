import unittest

from cryptochat.knowledge import KnowledgeBase
from cryptochat.models import AppConfig, Keyword
from cryptochat.retrieval import KeywordRetriever, ScopeResolver, evaluate, id_key, match

from .support import lemmas, two_products, valid

CONFIG = AppConfig()


class ScorerTests(unittest.TestCase):
    def test_match_follows_prefix_rule(self):
        cases = [("истекл", "истек", True), ("лицензи", "лицензи", True), ("куп", "купил", False),
                 ("пин", "пинг", False), ("порт", "ipsec", False)]
        for lemma, keyword, expected in cases:
            with self.subTest(lemma=lemma, keyword=keyword):
                self.assertEqual(expected, match(lemma, keyword, 4))

    def test_score_is_matched_weight_over_total(self):
        keywords = [Keyword("куп", 5), Keyword("лицензи", 3), Keyword("оплат", 2)]
        score, matched = evaluate(keywords, ["куп", "лицензи"], 4)
        self.assertAlmostEqual(0.8, score, places=6)
        self.assertEqual(2, matched)

    def test_each_keyword_counts_once(self):
        keywords = [Keyword("лицензи", 4), Keyword("оплат", 4)]
        score, matched = evaluate(keywords, ["лицензи", "лицензи", "лицензиат"], 4)
        self.assertAlmostEqual(0.5, score, places=6)
        self.assertEqual(1, matched)

    def test_no_keywords_is_zero(self):
        self.assertEqual((0.0, 0), evaluate([], ["куп"], 4))

    def test_ids_sort_numerically(self):
        ids = ["1.1.1.10", "1.1.1.2", "1.1.1.1", "1.2.1.1"]
        self.assertEqual(["1.1.1.1", "1.1.1.2", "1.1.1.10", "1.2.1.1"], sorted(ids, key=id_key))


class RetrievalTests(unittest.TestCase):
    def test_scope_resolver_finds_product_by_keywords(self):
        kb = KnowledgeBase.build(valid())
        scope = ScopeResolver(kb, CONFIG).resolve(lemmas(kb, "кластер ngate"))
        self.assertEqual("2.1", scope.category.id)

    def test_scope_resolver_no_match_gives_no_scope(self):
        kb = KnowledgeBase.build(valid())
        self.assertIsNone(ScopeResolver(kb, CONFIG).resolve(lemmas(kb, "лицензия")).category)

    def test_scope_resolver_tie_gives_no_scope(self):
        kb = KnowledgeBase.build(two_products())
        self.assertIsNone(ScopeResolver(kb, CONFIG).resolve(lemmas(kb, "лицензия")).category)

    def test_scope_resolver_clear_leader_wins(self):
        kb = KnowledgeBase.build(two_products())
        self.assertEqual("2.1", ScopeResolver(kb, CONFIG).resolve(lemmas(kb, "csp лицензия")).category.id)

    def test_category_scope_off_keeps_only_products(self):
        config = AppConfig()
        config.scoring.category_scope = False
        kb = KnowledgeBase.build(valid())
        self.assertEqual("2.1", ScopeResolver(kb, config).resolve(lemmas(kb, "кластер ngate")).category.id)

    def test_retriever_applies_scope_boost(self):
        kb = KnowledgeBase.build(valid())
        result = KeywordRetriever(kb, CONFIG).search(lemmas(kb, "ngate"))
        self.assertEqual("2.1", result.scope.id)
        self.assertEqual(1, len(result.candidates))
        self.assertEqual("2.1.1", result.candidates[0].question.id)
        self.assertAlmostEqual(0.75, result.candidates[0].score, places=3)  # 6/10 * 1.25

    def test_retriever_without_scope_searches_whole_base(self):
        kb = KnowledgeBase.build(valid())
        result = KeywordRetriever(kb, CONFIG).search(lemmas(kb, "купить лицензию"))
        self.assertIsNone(result.scope)
        self.assertEqual("1.1.1", result.candidates[0].question.id)

    def test_retriever_forced_scope_limits_candidates(self):
        kb = KnowledgeBase.build(two_products())
        result = KeywordRetriever(kb, CONFIG).search(lemmas(kb, "лицензия"), kb.categories["2.2"])
        self.assertEqual("2.2", result.scope.id)
        self.assertEqual(["2.2.1"], [c.question.id for c in result.candidates])

    def test_retriever_ignores_drafts(self):
        content = valid(); content.questions[0].status = "draft"
        kb = KnowledgeBase.build(content)
        self.assertEqual((), KeywordRetriever(kb, CONFIG).search(lemmas(kb, "купить лицензию")).candidates)

    def test_retriever_no_lemmas_gives_no_candidates(self):
        kb = KnowledgeBase.build(valid())
        self.assertEqual((), KeywordRetriever(kb, CONFIG).search([]).candidates)


if __name__ == "__main__":
    unittest.main()
