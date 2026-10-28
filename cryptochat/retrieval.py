"""Поиск: оценка вопросов по ключевым словам, область поиска, топ кандидатов."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Protocol, Sequence

from .knowledge import KnowledgeBase
from .models import AppConfig, Category, Keyword, Question


# ---------- Оценка ----------

def match(lemma: str, keyword: str, prefix_min_length: int) -> bool:
    """Основы совпадают, если равны. Для длинных основ (от prefix_min_length символов)
    достаточно, чтобы одна начиналась с другой."""
    return lemma == keyword or (
        min(len(lemma), len(keyword)) >= prefix_min_length
        and (lemma.startswith(keyword) or keyword.startswith(lemma))
    )


def evaluate(keywords: Sequence[Keyword], lemmas: Sequence[str], prefix_min_length: int) -> tuple[float, int]:
    """score = вес совпавших ключей / вес всех ключей. Каждый ключ считается один раз.
    Возвращает (score, число совпавших ключей)."""
    total = 0.0
    matched_weight = 0.0
    matched = 0
    for keyword in keywords:
        total += keyword.weight
        if any(match(l, keyword.word, prefix_min_length) for l in lemmas):
            matched_weight += keyword.weight
            matched += 1
    if total <= 0:
        return 0.0, 0
    return matched_weight / total, matched


def id_key(identifier: str) -> tuple:
    """Ключ сортировки id по числам, а не по буквам: 1.1.1.2 идёт раньше 1.1.1.10."""
    parts = []
    for part in identifier.split("."):
        parts.append((0, int(part), "") if part.isdigit() else (1, 0, part))
    return tuple(parts)


# ---------- Результат поиска ----------

@dataclass(frozen=True)
class Candidate:
    question: Question
    score: float


@dataclass(frozen=True)
class RetrievalResult:
    """Итог поиска: основы слов запроса, найденная область (или None) и кандидаты по убыванию score."""
    lemmas: tuple[str, ...]
    scope: Optional[Category]
    scope_score: float
    candidates: tuple[Candidate, ...]


class Retriever(Protocol):
    def search(self, lemmas: Sequence[str], forced_scope: Optional[Category] = None) -> RetrievalResult: ...


# ---------- Область поиска ----------

@dataclass(frozen=True)
class ScopeResult:
    category: Optional[Category]
    score: float


def _depth(identifier: str) -> int:
    return identifier.count(".")


def _same_branch(a: str, b: str) -> bool:
    return a == b or a.startswith(b + ".") or b.startswith(a + ".")


class ScopeResolver:
    """Определяет область поиска по ключевым словам разделов. Если уверенного лидера нет,
    область не определена и поиск идёт по всей базе."""

    def __init__(self, kb: KnowledgeBase, config: AppConfig):
        self._config = config
        self._categories = [
            c for c in kb.categories.values()
            if c.keywords and (config.scoring.category_scope or c.kind == "product")
        ]

    def resolve(self, lemmas: Sequence[str]) -> ScopeResult:
        if not self._categories or not lemmas:
            return ScopeResult(None, 0.0)

        prefix = self._config.scoring.prefix_min_length
        scored = [(c, evaluate(c.keywords, lemmas, prefix)[0]) for c in self._categories]
        best_cat, best_score = min(scored, key=lambda x: (-x[1], -_depth(x[0].id), id_key(x[0].id)))

        t = self._config.thresholds
        if best_score <= 0 or best_score < t.scope_min:
            return ScopeResult(None, best_score)

        rival = max([s for c, s in scored if not _same_branch(c.id, best_cat.id)], default=0.0)
        if best_score - rival >= t.scope_margin:
            return ScopeResult(best_cat, best_score)
        return ScopeResult(None, best_score)


# ---------- Поиск по ключевым словам ----------

MAX_CANDIDATES = 5


class KeywordRetriever:
    """Поиск по ключевым словам: область → оценка вопросов → топ-5."""

    def __init__(self, kb: KnowledgeBase, config: AppConfig):
        self._kb = kb
        self._config = config
        self._scope_resolver = ScopeResolver(kb, config)

    def search(self, lemmas: Sequence[str], forced_scope: Optional[Category] = None) -> RetrievalResult:
        lemmas = tuple(lemmas)
        if not lemmas:
            return RetrievalResult(lemmas, None, 0.0, ())

        if forced_scope is not None:
            scope, scope_score = forced_scope, 1.0
        else:
            resolved = self._scope_resolver.resolve(lemmas)
            scope, scope_score = resolved.category, resolved.score

        if scope is not None:
            in_scope = self._rank(self._kb.questions_under(scope.id), lemmas, self._config.scoring.scope_boost)
            if in_scope or forced_scope is not None:
                return RetrievalResult(lemmas, scope, scope_score, in_scope)
            # В области ничего не нашлось: ищем по всей базе.

        everywhere = self._rank(self._kb.questions, lemmas, 0.0)
        return RetrievalResult(lemmas, None, scope_score if scope is None else 0.0, everywhere)

    def _rank(self, pool: Iterable[Question], lemmas: Sequence[str], boost: float) -> tuple[Candidate, ...]:
        prefix = self._config.scoring.prefix_min_length
        scored = []
        for question in pool:
            if not question.is_active:
                continue
            score, matched = evaluate(question.keywords, lemmas, prefix)
            if score <= 0:
                continue
            final = min(1.0, score * (1 + boost)) if boost > 0 else score
            scored.append((question, final, matched))
        scored.sort(key=lambda x: (-x[1], -x[2], id_key(x[0].id)))
        return tuple(Candidate(q, s) for q, s, _ in scored[:MAX_CANDIDATES])
