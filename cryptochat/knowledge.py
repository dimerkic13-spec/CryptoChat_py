"""База знаний: чтение каталога content/, проверка данных (V001–V016) и построение базы в памяти."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from datetime import date
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

from .models import Category, ContentError, Keyword, Question
from .text import DEFAULT_STOP_WORDS, Lemmatizer, SynonymMap, normalize, stem


# ---------- «Сырые» данные: ровно в том виде, как они записаны в JSON ----------

@dataclass
class RawKeyword:
    word: str = ""
    weight: float = 0.0


@dataclass
class RawCategory:
    id: str = ""
    name: str = ""
    kind: str = ""
    description: Optional[str] = None
    contact: Optional[str] = None
    keywords: list[RawKeyword] = field(default_factory=list)
    children: list["RawCategory"] = field(default_factory=list)


@dataclass
class RawSource:
    url: str = ""
    anchor: Optional[str] = None
    revision: Optional[str] = None


@dataclass
class RawQuestion:
    id: str = ""
    category: str = ""
    title: str = ""
    answer: str = ""
    source: Optional[RawSource] = field(default_factory=RawSource)
    status: str = "active"
    keywords: list[RawKeyword] = field(default_factory=list)


@dataclass
class ContentSet:
    """Всё содержимое каталога content/ в сыром виде."""
    categories: list[RawCategory] = field(default_factory=list)
    synonym_groups: list[list[str]] = field(default_factory=list)
    questions: list[RawQuestion] = field(default_factory=list)


def _get(data: dict, key: str, default: Any = None) -> Any:
    """Чтение ключа без учёта регистра (как в исходной версии на C#)."""
    if key in data:
        return data[key]
    lower = key.lower()
    for k, v in data.items():
        if isinstance(k, str) and k.lower() == lower:
            return v
    return default


def _str(value: Any) -> str:
    return "" if value is None else str(value)


def _keywords(items: Any) -> list[RawKeyword]:
    result = []
    for item in items or []:
        if not isinstance(item, dict):
            raise ContentError(f"Ключевое слово должно быть объектом {{\"word\", \"weight\"}}: {item!r}")
        weight = _get(item, "weight", 0)
        if isinstance(weight, bool) or not isinstance(weight, (int, float)):
            raise ContentError(f"Вес ключевого слова должен быть числом: {item!r}")
        result.append(RawKeyword(word=_str(_get(item, "word", "")), weight=float(weight)))
    return result


def _category(data: Any) -> RawCategory:
    if not isinstance(data, dict):
        raise ContentError(f"Категория должна быть объектом: {data!r}")
    return RawCategory(
        id=_str(_get(data, "id", "")),
        name=_str(_get(data, "name", "")),
        kind=_str(_get(data, "kind", "")),
        description=_get(data, "description"),
        contact=_get(data, "contact"),
        keywords=_keywords(_get(data, "keywords")),
        children=[_category(c) for c in (_get(data, "children") or [])],
    )


def _question(data: Any) -> RawQuestion:
    if not isinstance(data, dict):
        raise ContentError(f"Вопрос должен быть объектом: {data!r}")
    source_data = _get(data, "source", {})
    source = None
    if isinstance(source_data, dict):
        source = RawSource(
            url=_str(_get(source_data, "url", "")),
            anchor=_get(source_data, "anchor"),
            revision=_get(source_data, "revision"),
        )
    return RawQuestion(
        id=_str(_get(data, "id", "")),
        category=_str(_get(data, "category", "")),
        title=_str(_get(data, "title", "")),
        answer=_str(_get(data, "answer", "")),
        source=source,
        status=_str(_get(data, "status", "active")),
        keywords=_keywords(_get(data, "keywords")),
    )


def _read_json(path: Path, required: bool) -> Any:
    if not path.exists():
        if required:
            raise ContentError(f"Файл не найден: {path}")
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError as ex:
        raise ContentError(f"Ошибка в файле {path}: {ex}") from ex
    if data is None:
        raise ContentError(f"Файл пуст: {path}")
    return data


def load_content(directory: str | Path) -> ContentSet:
    """Читает файлы как есть, без проверки содержимого."""
    root = Path(directory)
    if not root.is_dir():
        raise ContentError(f"Каталог с данными не найден: {root}")

    taxonomy_path = root / "taxonomy.json"
    taxonomy = _read_json(taxonomy_path, required=True)
    synonyms = _read_json(root / "synonyms.json", required=False) or {}

    try:
        categories = [_category(c) for c in (_get(taxonomy, "categories") or [])]
    except ContentError as ex:
        raise ContentError(f"Ошибка в файле {taxonomy_path}: {ex}") from ex

    groups = []
    for group in _get(synonyms, "groups") or []:
        groups.append([_str(w) for w in group])

    questions: list[RawQuestion] = []
    questions_dir = root / "questions"
    if questions_dir.is_dir():
        for file in sorted(questions_dir.glob("*.json"), key=lambda p: p.name):
            data = _read_json(file, required=True)
            if not isinstance(data, list):
                raise ContentError(f"Ошибка в файле {file}: ожидался список вопросов [ ... ].")
            try:
                questions.extend(_question(q) for q in data)
            except ContentError as ex:
                raise ContentError(f"Ошибка в файле {file}: {ex}") from ex

    return ContentSet(categories=categories, synonym_groups=groups, questions=questions)


# ---------- Проверка данных ----------

class Severity(Enum):
    ERROR = "error"
    WARNING = "warning"


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    severity: Severity
    where: str
    message: str

    def __str__(self) -> str:
        label = "ОШИБКА" if self.severity is Severity.ERROR else "предупреждение"
        return f"{label} {self.code} [{self.where}]: {self.message}"


_LATIN = re.compile(r"[a-z]")
_CYRILLIC = re.compile(r"[а-яё]")
_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def _is_date(text: str) -> bool:
    if not _DATE.match(text):
        return False
    try:
        date.fromisoformat(text)
        return True
    except ValueError:
        return False


def validate(content: ContentSet, stop_words: Iterable[str] | None = None,
             lemmatize: Lemmatizer = stem) -> list[ValidationIssue]:
    """Ошибки (V001–V012) мешают запуску, предупреждения (V013, V014, V016) только сообщают о подозрительных местах.

    lemmatize — та же функция приведения слова, что и при поиске (стеммер или pymorphy3).
    С морфологией повтор ключа (V008) — только предупреждение: разные формы одного слова
    («купил», «купить») сливаются в одну, и при загрузке берётся больший вес.
    """
    strict_duplicates = lemmatize is stem
    issues: list[ValidationIssue] = []
    stop = {normalize(w) for w in (stop_words if stop_words is not None else DEFAULT_STOP_WORDS)}

    def err(code: str, where: str, message: str) -> None:
        issues.append(ValidationIssue(code, Severity.ERROR, where, message))

    def warn(code: str, where: str, message: str) -> None:
        issues.append(ValidationIssue(code, Severity.WARNING, where, message))

    def check_keywords(where: str, keywords: list[RawKeyword], is_question: bool) -> None:
        if is_question and len(keywords) < 3:
            err("V006", where, f"Ключевых слов должно быть не меньше 3, сейчас {len(keywords)}.")

        seen: set[str] = set()
        for k in keywords:
            word = normalize((k.word or "").strip())

            if k.weight < 0.1 or k.weight > 10:
                err("V007", where, f"Вес ключа «{k.word}» вне диапазона 0.1–10: {k.weight:g}.")

            if not word or any(ch.isspace() for ch in word):
                err("V010", where, f"Ключ «{k.word}» пуст или состоит из нескольких слов.")
                continue

            if _LATIN.search(word) and _CYRILLIC.search(word):
                err("V009", where, f"Ключ «{k.word}» смешивает латиницу и кириллицу.")

            s = lemmatize(word)
            if not s:
                err("V010", where, f"Ключ «{k.word}» пуст после стемминга.")
                continue

            if s in seen:
                (err if strict_duplicates else warn)("V008", where, f"Ключ «{k.word}» повторяется (основа «{s}»).")
            seen.add(s)

            if word in stop:
                warn("V014", where, f"Ключ «{k.word}» совпадает со стоп-словом и никогда не сработает.")

    # ---------- Категории ----------
    categories: dict[str, RawCategory] = {}

    def walk(category: RawCategory, parent: Optional[RawCategory]) -> None:
        where = f"категория {category.id}"
        if not category.id.strip():
            err("V001", "категория", "У категории пустой id.")
        elif category.id in categories:
            err("V001", where, "Id категории повторяется.")
        else:
            categories[category.id] = category

        if parent is not None and not category.id.startswith(parent.id + "."):
            err("V011", where, f"Id дочерней категории должен начинаться с «{parent.id}.».")

        check_keywords(where, category.keywords, False)
        for child in category.children:
            walk(child, category)

    for root in content.categories:
        walk(root, None)

    # ---------- Вопросы ----------
    question_ids: set[str] = set()
    with_questions: set[str] = set()

    for q in content.questions:
        where = f"вопрос {q.id}"

        if not q.id.strip():
            err("V001", "вопрос", "У вопроса пустой id.")
        elif q.id in question_ids:
            err("V001", where, "Id вопроса повторяется.")
        else:
            question_ids.add(q.id)

        category = categories.get(q.category)
        if category is None:
            err("V002", where, f"Категория «{q.category}» не найдена в taxonomy.json.")
        elif category.children:
            err("V002", where, f"Категория «{q.category}» не конечная: вопрос можно привязать только к категории без дочерних.")
        else:
            with_questions.add(category.id)

        if not q.id.startswith(q.category + "."):
            err("V003", where, f"Id вопроса должен начинаться с «{q.category}.».")

        if not q.title.strip():
            err("V004", where, "Пустой заголовок (title).")
        if not q.answer.strip():
            err("V004", where, "Пустой ответ (answer).")

        if q.source is None or not q.source.url.strip() or not q.source.url.startswith("https://"):
            err("V005", where, "Нужна ссылка на источник source.url, начинающаяся с https://.")

        check_keywords(where, q.keywords, True)

        revision = q.source.revision if q.source is not None else None
        if revision is not None and not _is_date(str(revision)):
            warn("V016", where, f"Дата редакции «{revision}» не в формате ГГГГ-ММ-ДД.")

    # ---------- Синонимы ----------
    synonym_owner: dict[str, int] = {}  # основа → номер группы
    for i, group in enumerate(content.synonym_groups):
        where = f"синонимы, группа {i + 1}"
        if len(group) < 2:
            err("V012", where, "В группе должно быть минимум два слова.")
            continue
        if " " in normalize(group[0]).strip():
            err("V012", where, "Первое (каноническое) слово группы должно быть одним словом.")
        for member in group:
            n = normalize(member).strip()
            if not n:
                continue
            key = n if " " in n else lemmatize(n)
            if synonym_owner.setdefault(key, i) != i:
                err("V012", where, f"Слово «{member}» входит в несколько групп.")

    # ---------- Пустые разделы ----------
    for category in categories.values():
        if not category.children and category.id not in with_questions:
            warn("V013", f"категория {category.id}", f"В разделе «{category.name}» нет ни одного вопроса.")

    return issues


# ---------- База знаний в памяти ----------

class KnowledgeBase:
    """Дерево разделов, вопросы и синонимы. Ключевые слова стеммятся один раз, при построении."""

    def __init__(self, roots: list[Category], categories: dict[str, Category],
                 questions: list[Question], synonyms: SynonymMap, lemmatize: Lemmatizer = stem):
        self.roots = roots
        self.categories = categories
        self.questions = questions
        self.synonyms = synonyms
        self.lemmatize = lemmatize  # та же функция нужна для обработки запросов

    def questions_under(self, category_id: str) -> list[Question]:
        """Вопросы раздела и всех его вложенных разделов."""
        prefix = category_id + "."
        return [q for q in self.questions if q.category_id == category_id or q.category_id.startswith(prefix)]

    def parent_of(self, category: Category) -> Optional[Category]:
        return self.categories.get(category.parent_id) if category.parent_id else None

    def nearest_ancestor_or_self(self, category_id: str, predicate: Callable[[Category], bool]) -> Optional[Category]:
        """Идёт вверх по дереву от раздела и возвращает первый, подходящий под условие."""
        current = self.categories.get(category_id)
        while current is not None:
            if predicate(current):
                return current
            current = self.parent_of(current)
        return None

    @staticmethod
    def build(content: ContentSet, lemmatize: Lemmatizer = stem) -> "KnowledgeBase":
        synonyms = SynonymMap(content.synonym_groups, lemmatize)
        flat: dict[str, Category] = {}

        def to_keywords(raws: list[RawKeyword]) -> list[Keyword]:
            # Если два ключа дали одну основу (с морфологией: «купил» и «купить»),
            # остаётся один ключ с большим весом.
            merged: dict[str, float] = {}
            for k in raws:
                word = synonyms.map_lemma(lemmatize(normalize(k.word.strip())))
                merged[word] = max(merged.get(word, 0.0), k.weight)
            return [Keyword(word, weight) for word, weight in merged.items()]

        def make(raw: RawCategory, parent_id: Optional[str]) -> Category:
            category = Category(raw.id, parent_id, raw.name, raw.kind, raw.contact, to_keywords(raw.keywords))
            category.children = [make(child, raw.id) for child in raw.children]
            flat[category.id] = category
            return category

        roots = [make(c, None) for c in content.categories]
        questions = [
            Question(
                id=q.id,
                category_id=q.category,
                title=q.title,
                answer=q.answer,
                source_url=q.source.url if q.source else "",
                anchor=q.source.anchor if q.source else None,
                revision=q.source.revision if q.source else None,
                is_active=q.status.lower() != "draft",
                keywords=to_keywords(q.keywords),
            )
            for q in content.questions
        ]
        return KnowledgeBase(roots, flat, questions, synonyms, lemmatize)


def load(directory: str | Path, lemmatize: Lemmatizer = stem) -> KnowledgeBase:
    """Читает, проверяет и строит базу. При ошибках в данных бросает ContentError."""
    content = load_content(directory)
    errors = [i for i in validate(content, lemmatize=lemmatize) if i.severity is Severity.ERROR]
    if errors:
        text = "\n".join(str(e) for e in errors)
        raise ContentError(f"В данных найдены ошибки ({len(errors)}):\n{text}")
    try:
        return KnowledgeBase.build(content, lemmatize)
    except ValueError as ex:  # ошибка в синонимах, не пойманная проверкой
        raise ContentError(f"Ошибка в синонимах: {ex}") from ex
