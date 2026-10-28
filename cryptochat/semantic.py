"""Поиск по смыслу: нейросетевая модель sentence-transformers (работает на PyTorch).

Модель превращает фразу в вектор чисел так, что близкие по смыслу фразы дают близкие векторы.
Поэтому «не могу зайти в аккаунт» находит «Забыл логин/пароль от личного кабинета»,
хотя общих слов у них нет. Текст ответа по-прежнему берётся дословно из базы:
модель только помогает выбрать вопрос, сама она ничего не пишет. Работает на этом компьютере,
запросы никуда не отправляются.

Как используется (config.json → semantic):
  • при запуске модель один раз считает векторы для заголовков вопросов (и для примеров
    формулировок из поля "examples", если они есть у вопроса);
  • для запроса считается похожесть (косинус) с каждым вопросом;
  • похожесть переводится в оценку 0–weight одним из двух способов (semantic.mode):
      "relative" (по умолчанию) — насколько вопрос выделяется на фоне остальных: считается
        z-оценка = (похожесть − средняя по всем вопросам) / разброс. Учитываются только topK самых
        похожих вопросов; ниже zLow — 0, выше zHigh — weight, между ними — линейно.
        Так модель не «заливает» поиск, если ей все вопросы кажутся похожими;
      "absolute" — по самой похожести: ниже low — 0, выше high — weight;
  • итоговая оценка вопроса (semantic.combine):
      "max" — лучшее из оценки по ключам и по смыслу;
      "or"  — ключи + смысл − ключи × смысл: совпадения по обоим признакам усиливают друг друга.

Библиотека необязательная. Если sentence-transformers не установлен или модель не загрузилась,
бот пишет об этом при запуске и работает как раньше — по ключам.

Модель скачивается из Hugging Face при первом запуске (нужен интернет) в папку models/
рядом с программой. Дальше работает без интернета.
"""
from __future__ import annotations

import importlib
import math
from typing import Callable, Iterable, Optional, Protocol, Sequence

from .models import Question


class Encoder(Protocol):
    """То, что умеет превращать список фраз в список векторов. В тестах подменяется."""
    def encode(self, texts: list[str]) -> list[list[float]]: ...


class SemanticUnavailable(RuntimeError):
    """sentence-transformers / PyTorch не установлены или модель не загрузилась."""


class SentenceTransformerEncoder:
    """Обёртка над sentence_transformers.SentenceTransformer."""

    def __init__(self, model_name: str, cache_dir: Optional[str] = None, device: str = "cpu",
                 local_files_only: bool = False):
        try:
            # Импорт через importlib: так PyInstaller не тянет PyTorch в обычную сборку .exe.
            module = importlib.import_module("sentence_transformers")
        except ImportError as ex:
            raise SemanticUnavailable(
                "sentence-transformers не установлен. Установка: python -m pip install -r requirements-semantic.txt"
            ) from ex
        try:
            kwargs = {"device": device}
            if cache_dir:
                kwargs["cache_folder"] = cache_dir
            if local_files_only:
                kwargs["local_files_only"] = True
            self._model = module.SentenceTransformer(model_name, **kwargs)
        except Exception as ex:  # нет интернета, неверное имя модели, повреждённые файлы
            raise SemanticUnavailable(f"Не удалось загрузить модель «{model_name}»: {ex}") from ex
        self.name = model_name

    def encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, convert_to_numpy=True,
                                     show_progress_bar=False)
        return [list(map(float, v)) for v in vectors]


def _normalize(vector: Sequence[float]) -> list[float]:
    length = math.sqrt(sum(x * x for x in vector))
    return [x / length for x in vector] if length > 0 else list(vector)


def _dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


class SemanticIndex:
    """Векторы вопросов и расчёт смысловой оценки для запроса."""

    def __init__(self, encoder: Encoder, questions: Iterable[Question], low: float = 0.55,
                 high: float = 0.9, weight: float = 0.8, name: Optional[str] = None,
                 mode: str = "absolute", z_low: float = 2.5, z_high: float = 4.5, top_k: int = 3,
                 query_prefix: str = "", passage_prefix: str = ""):
        self.encoder = encoder
        self.low = low
        self.high = high
        self.weight = weight
        self.mode = mode
        self.z_low = z_low
        self.z_high = z_high
        self.top_k = top_k
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.name = name or getattr(encoder, "name", "модель")
        self._owners: list[str] = []      # для каждой фразы — id вопроса
        texts: list[str] = []
        for q in questions:
            if not q.is_active:
                continue
            for text in [q.title, *q.examples]:
                if text and text.strip():
                    texts.append(passage_prefix + text.strip())
                    self._owners.append(q.id)
        self._vectors = [_normalize(v) for v in encoder.encode(texts)] if texts else []
        self._last: tuple[str, dict[str, float]] = ("", {})

    @property
    def phrases(self) -> int:
        return len(self._vectors)

    def similarities(self, query: str) -> dict[str, float]:
        """Косинус запроса с каждым вопросом (лучшая из его фраз). Результат последнего запроса запоминается."""
        query = (query or "").strip()
        if not query or not self._vectors:
            return {}
        if self._last[0] == query:
            return self._last[1]
        q = _normalize(self.encoder.encode([self.query_prefix + query])[0])
        best: dict[str, float] = {}
        for owner, vector in zip(self._owners, self._vectors):
            value = _dot(q, vector)
            if value > best.get(owner, -1.0):
                best[owner] = value
        self._last = (query, best)
        return best

    def to_score(self, cosine: float) -> float:
        """Косинус → оценка 0–1 с учётом веса смыслового поиска."""
        if cosine <= self.low:
            return 0.0
        if cosine >= self.high:
            return self.weight
        return self.weight * (cosine - self.low) / (self.high - self.low)

    @staticmethod
    def z_scores(similarities: dict[str, float]) -> dict[str, float]:
        """Насколько каждый вопрос выделяется: (похожесть − средняя) / стандартное отклонение."""
        values = list(similarities.values())
        if len(values) < 2:
            return {k: 0.0 for k in similarities}
        mean = sum(values) / len(values)
        std = math.sqrt(sum((v - mean) ** 2 for v in values) / len(values))
        if std <= 1e-9:
            return {k: 0.0 for k in similarities}
        return {k: (v - mean) / std for k, v in similarities.items()}

    def _ramp(self, value: float, low: float, high: float) -> float:
        if value <= low:
            return 0.0
        if value >= high:
            return self.weight
        return self.weight * (value - low) / (high - low)

    def scores(self, query: str) -> dict[str, float]:
        """Смысловая оценка (уже с весом) для каждого вопроса, у которого она больше нуля."""
        sims = self.similarities(query)
        if self.mode == "relative":
            z = self.z_scores(sims)
            top = sorted(sims, key=sims.get, reverse=True)[:self.top_k]
            result = {qid: self._ramp(z[qid], self.z_low, self.z_high) for qid in top}
        else:
            result = {qid: self.to_score(cos) for qid, cos in sims.items()}
        return {qid: s for qid, s in result.items() if s > 0}


def make_semantic(config, questions: Sequence[Question], app_dir=None,
                  encoder_factory: Optional[Callable[..., Encoder]] = None) -> tuple[Optional[SemanticIndex], Optional[str]]:
    """Создаёт индекс по настройкам config.json → semantic.

    Возвращает (индекс или None, примечание для пользователя). Если поиск по смыслу выключен,
    примечание пустое; если включён, но недоступен, — объяснение, почему.
    """
    if not config.enabled:
        return None, None
    cache_dir = None
    if config.cache_dir:
        from pathlib import Path
        path = Path(config.cache_dir)
        cache_dir = str(path if path.is_absolute() or app_dir is None else Path(app_dir) / path)
    factory = encoder_factory or SentenceTransformerEncoder
    try:
        encoder = factory(config.model, cache_dir=cache_dir, device=config.device,
                          local_files_only=config.local_files_only)
    except SemanticUnavailable as ex:
        return None, str(ex)
    name = getattr(encoder, "name", config.model)
    try:
        index = SemanticIndex(encoder, questions, config.low, config.high, config.weight, name,
                              mode=config.mode, z_low=config.z_low, z_high=config.z_high, top_k=config.top_k,
                              query_prefix=config.query_prefix, passage_prefix=config.passage_prefix)
    except Exception as ex:
        return None, f"Не удалось посчитать векторы вопросов: {ex}"
    return index, None
