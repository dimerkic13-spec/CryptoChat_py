"""Политика диалога (что делать с результатом поиска) и состояние беседы."""
from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum
from typing import Optional, Protocol

from .knowledge import KnowledgeBase
from .models import AppConfig, Category, ChatOption, Question, ReplyKind
from .retrieval import Candidate, RetrievalResult


class DecisionKind(Enum):
    NOT_UNDERSTOOD = "NotUnderstood"
    ESCALATE = "Escalate"
    ANSWER = "Answer"
    CLARIFY = "Clarify"
    CLARIFY_BRANCH = "ClarifyBranch"
    CLARIFY_PRODUCT = "ClarifyProduct"


@dataclass(frozen=True)
class DialogDecision:
    kind: DecisionKind
    candidates: tuple[Candidate, ...] = ()
    products: tuple[Category, ...] = ()


class Policy(Protocol):
    def decide(self, result: RetrievalResult) -> DialogDecision: ...


class DialogPolicy:
    """Решает по результату поиска: ответить, показать подходящие вопросы, уточнить тему,
    спросить про продукт или передать в поддержку.

    В обычном режиме при слабом результате бот сразу направляет в поддержку.
    В режиме dialog.guided вместо этого он уточняет тему по веткам дерева.
    """

    def __init__(self, config: AppConfig, kb: KnowledgeBase):
        self._config = config
        self._kb = kb

    def decide(self, result: RetrievalResult) -> DialogDecision:
        t = self._config.thresholds
        guided = self._config.dialog.guided
        max_options = self._config.menu.max_options

        if not result.lemmas:
            return DialogDecision(DecisionKind.NOT_UNDERSTOOD)

        candidates = result.candidates
        if not candidates or candidates[0].score < t.clarify:
            if guided:
                return DialogDecision(DecisionKind.CLARIFY_BRANCH, candidates)
            return DialogDecision(DecisionKind.ESCALATE)

        top = candidates[0].score
        if top < t.answer:
            fits = [c for c in candidates if c.score >= t.clarify]
            if guided and len(fits) > max_options:
                return DialogDecision(DecisionKind.CLARIFY_BRANCH, candidates)
            options = list(fits[:max_options])
        else:
            strong = [c for c in candidates if c.score >= t.answer]
            second = candidates[1].score if len(candidates) > 1 else 0.0
            if len(strong) == 1 or top - second >= t.dominance_gap:
                return DialogDecision(DecisionKind.ANSWER, (candidates[0],))
            if guided and len(strong) > max_options:
                return DialogDecision(DecisionKind.CLARIFY_BRANCH, candidates)
            options = list(strong[:max_options])

        if result.scope is None:
            products: list[Category] = []
            for option in options:
                product = self._product_of(option.question)
                if product is not None and all(p.id != product.id for p in products):
                    products.append(product)
            if len(products) >= 2:
                return DialogDecision(DecisionKind.CLARIFY_PRODUCT, tuple(options), tuple(products))

        fill = self._config.dialog.menu_fill_score
        if guided and fill > 0:
            # Меню из одного-двух вопросов часто не содержит нужного: добавляем следующие по оценке.
            shown = {c.question.id for c in options}
            for c in candidates:
                if len(options) >= max_options:
                    break
                if c.score >= fill and c.question.id not in shown:
                    options.append(c)
                    shown.add(c.question.id)

        return DialogDecision(DecisionKind.CLARIFY, tuple(options))

    def _product_of(self, question: Question) -> Optional[Category]:
        return self._kb.nearest_ancestor_or_self(question.category_id, lambda c: c.kind == "product")


# ---------- Состояние беседы ----------

@dataclass(frozen=True)
class PendingChoice:
    """Меню, на выбор из которого бот сейчас ждёт ответа.
    feedback_question_id заполнен, если бот только что выдал ответ и ждёт отзыв «да» / «нет»."""
    kind: ReplyKind
    options: tuple[ChatOption, ...]
    lemmas: tuple[str, ...]
    feedback_question_id: Optional[str] = None
    query: Optional[str] = None

    def without_feedback(self) -> "PendingChoice":
        return replace(self, feedback_question_id=None)


class ChatSession:
    """Состояние диалога: ждём ли выбор из меню и сколько шагов уточнения уже прошло."""

    def __init__(self) -> None:
        self.pending: Optional[PendingChoice] = None
        self.steps = 0  # сколько раз подряд бот уточнял, не дав ответа

    def wait(self, choice: PendingChoice) -> None:
        self.pending = choice

    def reset(self) -> None:
        """Сбрасывает только ожидание выбора; счётчик шагов уточнения сохраняется."""
        self.pending = None

    def next_step(self) -> None:
        self.steps += 1

    def reset_steps(self) -> None:
        self.steps = 0

    def reset_all(self) -> None:
        """Сбрасывает всё: ожидание выбора и счётчик шагов."""
        self.pending = None
        self.steps = 0
