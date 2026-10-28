"""Учёт опечаток: нечёткое сравнение слова запроса с ключевым словом.

«рутокин» ≈ «рутокен», «лицензя» ≈ «лицензия», «кабинед» ≈ «кабинет».

Похожесть считается от 0 до 100 библиотекой RapidFuzz (быстрая, на C++). Если она не установлена,
используется встроенный в Python difflib — медленнее, но для базы из сотен вопросов разницы не видно.

Опечатка засчитывается, только если:
  • обе основы не короче min_length букв (короткие слова слишком легко спутать: «пин» и «пинг»);
  • похожесть не ниже threshold (по умолчанию 85 из 100).
За совпадение с опечаткой ключ даёт не полный вес, а weight от него (по умолчанию 0.8),
чтобы точное совпадение всегда было весомее.
"""
from __future__ import annotations

from difflib import SequenceMatcher
from typing import Callable, Optional

from .text import stem


def _difflib_ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b).ratio() * 100


def _load_backend() -> tuple[Callable[[str, str], float], str]:
    try:
        from rapidfuzz import fuzz  # type: ignore[import-not-found]
        return fuzz.ratio, "RapidFuzz"
    except ImportError:
        return _difflib_ratio, "difflib (RapidFuzz не установлен)"


class FuzzyMatcher:
    def __init__(self, threshold: float = 85, min_length: int = 5, weight: float = 0.8,
                 ratio: Optional[Callable[[str, str], float]] = None, name: Optional[str] = None):
        if ratio is None:
            ratio, backend = _load_backend()
            name = name or backend
        self.threshold = threshold
        self.min_length = min_length
        self.weight = weight
        self.name = name or "своя функция"
        self._ratio = ratio
        self._cache: dict[tuple[str, str], float] = {}

    def similarity(self, a: str, b: str) -> float:
        """Похожесть двух основ 0–100. Сравниваются и сами основы, и их укороченные стеммером формы:
        так «рутокина» (неизвестное словарю слово с опечаткой) всё равно похоже на «рутокен»."""
        key = (a, b)
        cached = self._cache.get(key)
        if cached is not None:
            return cached
        value = max(self._ratio(a, b), self._ratio(stem(a), stem(b)))
        self._cache[key] = value
        return value

    def matches(self, lemma: str, keyword: str) -> bool:
        if len(lemma) < self.min_length or len(keyword) < self.min_length:
            return False
        return self.similarity(lemma, keyword) >= self.threshold


def make_fuzzy(config) -> Optional[FuzzyMatcher]:
    """FuzzyMatcher по настройкам config.json → fuzzy, или None, если учёт опечаток выключен."""
    if not config.enabled:
        return None
    return FuzzyMatcher(config.threshold, config.min_length, config.weight)
