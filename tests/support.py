"""Общие помощники для тестов: маленькая корректная база и путь к настоящему content/."""
from __future__ import annotations

from pathlib import Path

from cryptochat.knowledge import (ContentSet, KnowledgeBase, RawCategory, RawKeyword,
                                  RawQuestion, RawSource)
from cryptochat.text import DefaultTextProcessor

CONTENT_DIR = Path(__file__).resolve().parent.parent / "content"


def kw(*items: tuple[str, float]) -> list[RawKeyword]:
    return [RawKeyword(word=w, weight=float(x)) for w, x in items]


def q(id: str, category: str, title: str, *keywords: tuple[str, float]) -> RawQuestion:
    return RawQuestion(
        id=id, category=category, title=title, answer="Текст ответа.",
        source=RawSource(url="https://cryptopro.ru/faq/test", anchor="test", revision="2026-10-09"),
        keywords=kw(*keywords),
    )


def valid() -> ContentSet:
    """Маленькая корректная база. Каждый тест портит одно место."""
    return ContentSet(
        categories=[
            RawCategory(id="1", name="Коммерческие", kind="topic",
                        children=[RawCategory(id="1.1", name="Лицензии", kind="subtopic")]),
            RawCategory(id="2", name="Продукты", kind="topic",
                        children=[RawCategory(id="2.1", name="NGate", kind="product",
                                              contact="ngate@cryptopro.ru",
                                              keywords=kw(("ngate", 6), ("кластер", 3)))]),
        ],
        synonym_groups=[["вернуть", "возврат"]],
        questions=[
            q("1.1.1", "1.1", "Как купить лицензию?", ("купить", 5), ("лицензия", 3), ("оплата", 1)),
            q("2.1.1", "2.1", "Что такое кластер NGate?", ("ngate", 6), ("кластер", 3), ("узел", 1)),
        ],
    )


def two_products() -> ContentSet:
    """Тестовая база с двумя продуктами, у которых совпадает слово «лицензия»."""
    return ContentSet(
        categories=[
            RawCategory(id="2", name="Технические вопросы", kind="topic", children=[
                RawCategory(id="2.1", name="КриптоПро CSP", kind="product",
                            keywords=kw(("csp", 6), ("лицензия", 3))),
                RawCategory(id="2.2", name="КриптоПро NGate", kind="product", contact="ngate@cryptopro.ru",
                            keywords=kw(("ngate", 6), ("лицензия", 3))),
            ]),
        ],
        questions=[
            q("2.1.1", "2.1", "Лицензия CSP?", ("лицензия", 5), ("csp", 3), ("ключ", 1)),
            q("2.2.1", "2.2", "Лицензия NGate?", ("лицензия", 5), ("ngate", 3), ("ключ", 1)),
        ],
    )


def lemmas(kb: KnowledgeBase, text: str) -> list[str]:
    return DefaultTextProcessor(synonyms=kb.synonyms).lemmatize(text)
