"""Ядро: единственная точка входа ChatEngine.ask(текст) → ChatReply.

Консоль, веб-страница или мессенджер только вызывают ask() и показывают reply.text.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, Optional, Protocol, Sequence

from . import knowledge
from .dialog import ChatSession, DecisionKind, DialogPolicy, PendingChoice, Policy
from .knowledge import KnowledgeBase
from .models import (AppConfig, Category, ChatOption, ChatReply, ContentError, FeedbackEntry, Question,
                     ReplyKind)
from .fuzzy import FuzzyMatcher, make_fuzzy
from .retrieval import KeywordRetriever, RetrievalResult, Retriever, evaluate, id_key, typo_matches
from .semantic import SemanticIndex, make_semantic
from .text import (DefaultTextProcessor, MorphologyUnavailable, TextProcessor, lemmatizer_name,
                   make_lemmatizer)

EXIT_WORDS = {"выход", "exit", "quit", "/выход", "/exit", "/quit"}
FEEDBACK_YES = {"да", "+", "помог", "помогло"}
FEEDBACK_NO = {"нет", "-", "не помог", "не помогло"}


def fmt(value: float) -> str:
    return f"{value:.2f}"


# ---------- Сборка ответа ----------

class AnswerComposer(Protocol):
    def compose(self, question: Question, score: float, section: Optional[Category]) -> str: ...


class TemplateComposer:
    """Готовый ответ из базы, ссылка на источник и служебная строка."""

    def compose(self, question: Question, score: float, section: Optional[Category]) -> str:
        url = question.source_url if not question.anchor else f"{question.source_url}#{question.anchor}"
        revision = f" (ред. {question.revision})" if question.revision else ""
        section_part = f", раздел: {section.id} {section.name}" if section is not None else ""
        return (f"{question.answer}\n"
                f"📎 Источник: {url}{revision}\n"
                f"(score: {fmt(score)}{section_part})")


# ---------- Движок ----------

class ChatEngine:
    def __init__(self, kb: KnowledgeBase, config: AppConfig,
                 text: Optional[TextProcessor] = None,
                 retriever: Optional[Retriever] = None,
                 policy: Optional[Policy] = None,
                 composer: Optional[AnswerComposer] = None,
                 semantic: Optional[SemanticIndex] = None):
        self.knowledge_base = kb
        self.config = config
        self._text = text or DefaultTextProcessor(synonyms=kb.synonyms, lemmatize=kb.lemmatize)
        # Один и тот же учёт опечаток для поиска вопросов и для меню уточнения по веткам.
        self.fuzzy: Optional[FuzzyMatcher] = getattr(retriever, "fuzzy", None) if retriever else make_fuzzy(config.fuzzy)
        self._retriever = retriever or KeywordRetriever(kb, config, self.fuzzy, semantic)
        # Поиск по смыслу (None — выключен или недоступен; причина — в semantic_note).
        self.semantic: Optional[SemanticIndex] = getattr(self._retriever, "semantic", None)
        self.semantic_note: Optional[str] = None
        self._policy = policy or DialogPolicy(config, kb)
        self._composer = composer or TemplateComposer()
        self._session = ChatSession()
        self.debug_enabled = False
        # Вызывается, когда пользователь оценил ответ словом «да» или «нет».
        self.feedback_listeners: list[Callable[[FeedbackEntry], None]] = []
        # Чем приводятся слова к основе и почему (например, «pymorphy3 не установлен»).
        self.morphology = lemmatizer_name(kb.lemmatize)
        self.morphology_note: Optional[str] = None
        # Номер сессии задаёт консоль (или другой интерфейс); показывается по команде /сессия.
        self.session_id: Optional[str] = None

    @staticmethod
    def create(content_directory: str | Path, morphology: Optional[str] = None,
               semantic: Optional[bool] = None, app_dir: Optional[Path] = None,
               semantic_index: Optional[SemanticIndex] = None) -> "ChatEngine":
        """Загружает и проверяет каталог content/ и создаёт готовый движок.

        morphology переопределяет text.morphology из config.json (нужно для сравнения режимов);
        semantic=False выключает поиск по смыслу (тесты, сравнение), True — включает;
        app_dir — папка программы (от неё считается папка с моделью);
        semantic_index — готовый индекс (чтобы не загружать модель повторно)."""
        config = AppConfig.load(content_directory)
        if semantic is not None:
            config.semantic.enabled = semantic
        lemmatize, note = create_lemmatizer(morphology or config.text.morphology)
        kb = knowledge.load(content_directory, lemmatize)
        semantic_note = None
        if semantic_index is None and config.semantic.enabled:
            semantic_index, semantic_note = make_semantic(config.semantic, kb.questions, app_dir)
        engine = ChatEngine(kb, config, semantic=semantic_index if config.semantic.enabled else None)
        engine.morphology_note = note
        engine.semantic_note = semantic_note
        return engine

    @property
    def _guided(self) -> bool:
        return self.config.dialog.guided

    # ---------- Главный метод ----------

    def ask(self, user_input: Optional[str]) -> ChatReply:
        text = (user_input or "").strip()
        if not text:
            return self._simple(ReplyKind.NOT_UNDERSTOOD, "Введите вопрос или команду /помощь.")

        lower = text.lower()
        if lower in EXIT_WORDS:
            self._session.reset_all()
            return self._simple(ReplyKind.EXIT, "До свидания!")

        if text.startswith("/"):
            self._session.reset()
            return self._handle_command(text)

        pending = self._session.pending
        number = _parse_int(text)
        if pending is not None and number is not None:
            if number == 0:
                self._session.reset_all()
                return self._escalate_reply(None, None)
            if 1 <= number <= len(pending.options):
                self._session.reset()
                return self._choose(pending, pending.options[number - 1])

        if pending is not None and pending.feedback_question_id is not None:
            if lower in FEEDBACK_YES:
                return self._handle_feedback(pending, True)
            if lower in FEEDBACK_NO:
                return self._handle_feedback(pending, False)

        # Любой другой ввод в режиме ожидания выбора считается новым вопросом.
        self._session.reset()
        lemmas = self._text.lemmatize(text)
        return self._respond(self._retriever.search(lemmas, query=text), None, text)

    # ---------- Отзыв об ответе ----------

    def _handle_feedback(self, pending: PendingChoice, helpful: bool) -> ChatReply:
        entry = FeedbackEntry(datetime.now(), pending.query or "", pending.feedback_question_id or "", helpful)
        for listener in self.feedback_listeners:
            listener(entry)

        if helpful:
            self._session.reset()
            return self._simple(ReplyKind.FEEDBACK, "Спасибо за отзыв!")

        # Ответ не помог: оставляем похожие вопросы и предлагаем переформулировать.
        self._session.wait(pending.without_feedback())
        lines = ["Жаль, что ответ не помог. " + (
            "Возможно, поможет один из похожих вопросов (введите номер) или задайте вопрос другими словами:"
            if pending.options else "Попробуйте задать вопрос другими словами.")]
        lines += [f"  {o.number}. {o.label}" for o in pending.options]
        lines.append("  0. Обратиться в техподдержку")
        return ChatReply(ReplyKind.FEEDBACK, "\n".join(lines), pending.options)

    # ---------- Ответ на вопрос ----------

    def _choose(self, pending: PendingChoice, option: ChatOption) -> ChatReply:
        # Выбор продукта или ветки — это навигация по меню, а не неудачная попытка:
        # такие шаги не приближают перевод в поддержку.
        if option.product is not None:
            return self._respond(self._retriever.search(pending.lemmas, option.product, query=pending.query),
                                 option.product, pending.query, from_menu=True)
        if option.branch is not None:
            return self._respond(self._retriever.search(pending.lemmas, option.branch, query=pending.query),
                                 option.branch, pending.query, from_menu=True)

        reply = self._answer_reply(option.question, option.score, None)
        if not self._guided:
            return reply
        others = [(o.question, o.score) for o in pending.options
                  if o.question is not None and o.number != option.number]
        return self._with_feedback(reply, others, pending.lemmas, pending.query)

    def _respond(self, result: RetrievalResult, node: Optional[Category], query: Optional[str],
                 from_menu: bool = False) -> ChatReply:
        decision = self._policy.decide(result)
        debug = self._build_debug(result) if self.debug_enabled else None
        kind = decision.kind

        if kind is DecisionKind.NOT_UNDERSTOOD:
            return self._simple(ReplyKind.NOT_UNDERSTOOD, "Не понял вопрос. Попробуйте переформулировать.", debug)

        if kind is DecisionKind.ESCALATE:
            self._session.reset_all()
            return self._escalate_reply(result.scope, debug)

        if kind is DecisionKind.ANSWER:
            best = decision.candidates[0]
            reply = self._answer_reply(best.question, best.score, debug)
            if not self._guided:
                return reply
            others = [(c.question, c.score) for c in result.candidates[1:]
                      if c.score >= self.config.dialog.min_score]
            return self._with_feedback(reply, others, result.lemmas, query)

        if kind is DecisionKind.CLARIFY_PRODUCT:
            options = tuple(ChatOption(i + 1, p.name, 0.0, product=p) for i, p in enumerate(decision.products))
            self._session.wait(PendingChoice(ReplyKind.CLARIFY_PRODUCT, options, result.lemmas, None, query))
            lines = ["О каком продукте вы спрашиваете?"]
            lines += [f"  {o.number}. {o.label}" for o in options]
            lines += ["  0. Другое / связаться с поддержкой", "Введите номер:"]
            return ChatReply(ReplyKind.CLARIFY_PRODUCT, "\n".join(lines), options, debug)

        # Clarify или ClarifyBranch. В управляемом диалоге бот уточняет несколько раз,
        # и только потом направляет в поддержку. Переходы по меню тем (from_menu) не считаются.
        if self._guided and not from_menu:
            if self._session.steps >= self.config.dialog.escalate_after_steps:
                self._session.reset_all()
                return self._escalate_reply(result.scope or node, debug)
            self._session.next_step()

        if kind is DecisionKind.CLARIFY_BRANCH:
            # Свободный вопрос без явного совпадения: сначала предложить близкие варианты,
            # и только если их нет — меню тем. При переходе по меню тем показываем ветку целиком.
            if node is None and not from_menu:
                fallback = self._suggestion_menu(result, query, debug)
                if fallback is not None:
                    return fallback
            return self._branch_reply(node or result.scope, result, query, debug)

        options = tuple(ChatOption(i + 1, c.question.title, c.score, question=c.question)
                        for i, c in enumerate(decision.candidates))
        self._session.wait(PendingChoice(ReplyKind.CLARIFY, options, result.lemmas, None, query))

        if len(options) == 1:
            lines = [f"Возможно, вы спрашиваете: «{options[0].label}»?",
                     f"  1. Да, показать ответ  (score: {fmt(options[0].score)})",
                     "  0. Нет / связаться с поддержкой"]
        else:
            lines = ["Уточните, что вы имеете в виду?"]
            lines += [f"  {o.number}. {o.label}  (score: {fmt(o.score)})" for o in options]
            lines.append("  0. Другое / связаться с поддержкой")
        lines.append("Введите номер:")
        return ChatReply(ReplyKind.CLARIFY, "\n".join(lines), options, debug)

    # ---------- Запасное меню по смыслу ----------

    def _suggestion_menu(self, result: RetrievalResult, query: Optional[str],
                         debug: Optional[str]) -> Optional[ChatReply]:
        """Явного совпадения нет: показать близкие варианты вместо меню тем.

        Сначала — вопросы, которые по ключам почти дотянули до порога (оценка не ниже
        dialog.suggestScore), затем — самые похожие по смыслу (если поиск по смыслу включён).
        Возвращает None, если предложить нечего."""
        if not result.lemmas:
            return None
        kb = self.knowledge_base
        pool = {q.id: q for q in kb.questions if q.is_active}
        size = self.config.menu.max_options
        chosen: list[tuple[str, float]] = []

        # 1) почти совпавшие по ключам
        threshold = self.config.dialog.suggest_score
        for c in result.candidates:
            if threshold > 0 and c.score >= threshold and c.question.id in pool:
                chosen.append((c.question.id, c.score))

        # 2) похожие по смыслу
        cfg = self.config.semantic
        if self.semantic is not None and cfg.menu_fallback and result.cosines:
            cosines = {qid: c for qid, c in result.cosines.items() if qid in pool}
            if cosines:
                z = SemanticIndex.z_scores(cosines)
                for qid in sorted(cosines, key=cosines.get, reverse=True)[:cfg.menu_size]:
                    if z[qid] >= cfg.menu_z and all(qid != q for q, _ in chosen):
                        chosen.append((qid, cosines[qid]))

        chosen = chosen[:size]
        if not chosen:
            return None
        options = tuple(ChatOption(i + 1, pool[qid].title, score, question=pool[qid])
                        for i, (qid, score) in enumerate(chosen))
        self._session.wait(PendingChoice(ReplyKind.CLARIFY, options, result.lemmas, None, query))
        lines = ["Точного совпадения не нашёл. Возможно, вы имели в виду:"]
        lines += [f"  {o.number}. {o.label}" for o in options]
        lines += ["  0. Ничего не подходит — обратиться в техподдержку", "Введите номер:"]
        return ChatReply(ReplyKind.CLARIFY, "\n".join(lines), options, debug)

    # ---------- Уточнение по веткам дерева ----------

    def _branch_reply(self, node: Optional[Category], result: RetrievalResult,
                      query: Optional[str], debug: Optional[str]) -> ChatReply:
        """Предлагает выбрать тему (или блок), а на последнем уровне — конкретный вопрос.
        Ветки с единственным вложенным разделом пропускаются."""
        lemmas = result.lemmas
        prefix = self.config.scoring.prefix_min_length
        max_options = self.config.dialog.max_branch_options
        kb = self.knowledge_base

        children = self._children_with_questions(node)
        while len(children) == 1:
            node = children[0]
            children = self._children_with_questions(node)

        if not children:
            pool = kb.questions if node is None else kb.questions_under(node.id)
            sem = result.semantic
            ranked = sorted(((q, max(evaluate(q.keywords, lemmas, prefix, self.fuzzy)[0], sem.get(q.id, 0.0)))
                             for q in pool if q.is_active),
                            key=lambda x: (-x[1], id_key(x[0].id)))[:max_options]
            options = tuple(ChatOption(i + 1, q.title, s, question=q) for i, (q, s) in enumerate(ranked))
            self._session.wait(PendingChoice(ReplyKind.CLARIFY_BRANCH, options, lemmas, None, query))
            lines = ["Выберите подходящий вопрос:" if node is None
                     else f"В разделе «{node.name}» есть такие вопросы. Выберите подходящий:"]
            lines += [f"  {o.number}. {o.label}" for o in options]
            lines += ["  0. Ничего не подходит — обратиться в техподдержку", "Введите номер:"]
            return ChatReply(ReplyKind.CLARIFY_BRANCH, "\n".join(lines), options, debug)

        ranked_branches = sorted(((c, self._branch_score(c, lemmas, result.semantic)) for c in children),
                                 key=lambda x: (-x[1], id_key(x[0].id)))[:max_options]
        options = tuple(ChatOption(i + 1, c.name, s, branch=c) for i, (c, s) in enumerate(ranked_branches))
        self._session.wait(PendingChoice(ReplyKind.CLARIFY_BRANCH, options, lemmas, None, query))
        lines = ["Уточните, к какой теме относится ваш вопрос:" if node is None
                 else f"Уточните, к какой теме относится ваш вопрос ({node.name}):"]
        lines += [f"  {o.number}. {o.label}" for o in options]
        lines += ["  0. Ничего не подходит — обратиться в техподдержку", "Введите номер:"]
        return ChatReply(ReplyKind.CLARIFY_BRANCH, "\n".join(lines), options, debug)

    def _children_with_questions(self, node: Optional[Category]) -> list[Category]:
        kb = self.knowledge_base
        children = kb.roots if node is None else node.children
        return [c for c in children if any(q.is_active for q in kb.questions_under(c.id))]

    def _branch_score(self, category: Category, lemmas: Sequence[str], semantic: Optional[dict] = None) -> float:
        """Оценка ветки: лучшее из ключей самой ветки, ключей её вопросов и смысловой оценки вопросов."""
        prefix = self.config.scoring.prefix_min_length
        semantic = semantic or {}
        best = max((max(evaluate(q.keywords, lemmas, prefix, self.fuzzy)[0], semantic.get(q.id, 0.0))
                    for q in self.knowledge_base.questions_under(category.id) if q.is_active), default=0.0)
        own = evaluate(category.keywords, lemmas, prefix, self.fuzzy)[0] if category.keywords else 0.0
        return max(best, own)

    # ---------- Ответ с похожими вопросами и отзывом ----------

    def _with_feedback(self, reply: ChatReply, others: Iterable[tuple[Question, float]],
                       lemmas: Sequence[str], query: Optional[str]) -> ChatReply:
        """Добавляет под ответом похожие вопросы, просьбу оценить ответ и пункт «0. Техподдержка»."""
        options = tuple(ChatOption(i + 1, q.title, s, question=q)
                        for i, (q, s) in enumerate(list(others)[:self.config.menu.max_options]))
        feedback_id = reply.question.id if self.config.dialog.feedback and reply.question else None
        self._session.wait(PendingChoice(ReplyKind.CLARIFY, options, tuple(lemmas), feedback_id, query))
        self._session.reset_steps()

        lines = [reply.text, ""]
        if options:
            lines.append("Не то, что искали? Похожие вопросы:")
            lines += [f"  {o.number}. {o.label}" for o in options]
        if self.config.dialog.feedback:
            lines.append("Ответ помог? Напишите «да» или «нет».")
        lines += ["  0. Обратиться в техподдержку", "Введите номер или задайте новый вопрос:"]
        return ChatReply(reply.kind, "\n".join(lines), options, reply.debug, reply.question, reply.score)

    def _answer_reply(self, question: Question, score: float, debug: Optional[str]) -> ChatReply:
        section = self.knowledge_base.categories.get(question.category_id)
        text = self._composer.compose(question, score, section)
        return ChatReply(ReplyKind.ANSWER, text, (), debug, question, score)

    def _escalate_reply(self, scope: Optional[Category], debug: Optional[str]) -> ChatReply:
        text = f"Не нашёл ответа. Обратитесь в {self.config.support.url}"
        if scope is not None:
            holder = self.knowledge_base.nearest_ancestor_or_self(scope.id, lambda c: bool(c.contact))
            if holder is not None:
                text += f"\nПо вопросам «{holder.name}»: {holder.contact}"
        return ChatReply(ReplyKind.ESCALATE, text, (), debug)

    # ---------- Команды ----------

    def _handle_command(self, text: str) -> ChatReply:
        parts = text.split(None, 1)
        command = parts[0].lower()
        argument = parts[1].strip() if len(parts) > 1 else ""

        if command in ("/помощь", "/help"):
            return self._simple(ReplyKind.COMMAND,
                                "Напишите свой вопрос обычным текстом. Команды:\n"
                                "  /темы            показать разделы базы знаний\n"
                                "  /темы 1.2        показать вопросы раздела\n"
                                "  /отладка вкл     показывать, как бот ищет ответ\n"
                                "  /отладка выкл    выключить отладку\n"
                                "  /сессия          номер текущей беседы\n"
                                "  /помощь          эта подсказка\n"
                                "  выход            завершить работу")

        if command in ("/темы", "/topics"):
            return self._topics_tree() if not argument else self._topic_questions(argument)

        if command in ("/сессия", "/session"):
            if self.session_id:
                return self._simple(ReplyKind.COMMAND, f"Номер сессии: {self.session_id}")
            return self._simple(ReplyKind.COMMAND, "Журнал бесед выключен (config.json → sessions.enabled).")

        if command in ("/отладка", "/debug"):
            arg = argument.lower()
            if arg in ("вкл", "on"):
                self.debug_enabled = True
                return self._simple(ReplyKind.COMMAND, "Отладка включена.")
            if arg in ("выкл", "off"):
                self.debug_enabled = False
                return self._simple(ReplyKind.COMMAND, "Отладка выключена.")
            state = "включена" if self.debug_enabled else "выключена"
            return self._simple(ReplyKind.COMMAND, f"Отладка сейчас {state}. Команды: /отладка вкл, /отладка выкл.")

        return self._simple(ReplyKind.COMMAND, "Неизвестная команда. Список команд: /помощь")

    def _topics_tree(self) -> ChatReply:
        kb = self.knowledge_base
        lines = ["Разделы базы знаний:"]

        def walk(category: Category, level: int) -> None:
            count = len(kb.questions_under(category.id))
            lines.append(" " * (2 + level * 2) + f"{category.id} {category.name} ({count})")
            for child in category.children:
                walk(child, level + 1)

        for root in kb.roots:
            walk(root, 0)
        lines.append("Вопросы раздела: /темы <номер>, например /темы 1.2")
        return self._simple(ReplyKind.COMMAND, "\n".join(lines))

    def _topic_questions(self, category_id: str) -> ChatReply:
        kb = self.knowledge_base
        category = kb.categories.get(category_id)
        if category is None:
            return self._simple(ReplyKind.COMMAND, f"Раздел «{category_id}» не найден. Список разделов: /темы")
        questions = sorted(kb.questions_under(category_id), key=lambda q: id_key(q.id))
        if not questions:
            return self._simple(ReplyKind.COMMAND, f"В разделе «{category.name}» пока нет вопросов.")
        lines = [f"{category.id} {category.name}:"]
        lines += [f"  {q.id}  {q.title}" for q in questions]
        return self._simple(ReplyKind.COMMAND, "\n".join(lines))

    # ---------- Вспомогательное ----------

    @staticmethod
    def _simple(kind: ReplyKind, text: str, debug: Optional[str] = None) -> ChatReply:
        return ChatReply(kind, text, (), debug)

    def _build_debug(self, result: RetrievalResult) -> str:
        lines = [f"--- леммы: [{', '.join(result.lemmas)}]"]
        lines.append("--- область: не определена" if result.scope is None
                     else f"--- область: {result.scope.id} {result.scope.name} (score {fmt(result.scope_score)})")
        prefix = self.config.scoring.prefix_min_length
        for c in result.candidates:
            typos = typo_matches(c.question.keywords, result.lemmas, prefix, self.fuzzy)
            note = ("  (опечатка: " + ", ".join(f"{a}≈{b}" for a, b in typos) + ")") if typos else ""
            if c.cosine is not None:
                note += f"  [ключи {fmt(c.keyword_score)}, смысл: похожесть {fmt(c.cosine)}]"
            lines.append(f"  {c.question.id} \"{c.question.title}\" → {fmt(c.score)}{note}")
        fuzzy = self.fuzzy.name if self.fuzzy else "выкл"
        semantic = self.semantic.name if self.semantic else "выкл"
        lines.append(f"--- шагов уточнения без ответа: {self._session.steps}; морфология: {self.morphology}; "
                     f"опечатки: {fuzzy}; смысл: {semantic}")
        return "\n".join(lines)


def create_lemmatizer(mode: str):
    """make_lemmatizer, но с понятной ошибкой ContentError, если pymorphy3 нужен, а его нет."""
    try:
        return make_lemmatizer(mode)
    except MorphologyUnavailable as ex:
        raise ContentError(f"{ex} Или поставьте в config.json \"text\": {{\"morphology\": \"auto\"}}.") from ex
    except ValueError as ex:
        raise ContentError(str(ex)) from ex


def _parse_int(text: str) -> Optional[int]:
    """Число, как его понимал int.TryParse в C#: необязательный знак и цифры."""
    stripped = text.strip()
    body = stripped[1:] if stripped[:1] in "+-" else stripped
    if body.isdigit() and body.isascii():
        return int(stripped)
    return None
