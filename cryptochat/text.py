"""Подготовка текста: нормализация, синонимы, разбиение на слова, стоп-слова, приведение слова к основе.

Один и тот же код применяется и к запросу пользователя, и к ключевым словам базы,
поэтому они всегда сравниваются в одном виде.

Приводить слово к основе можно двумя способами (config.json → text.morphology):
  • "stemmer"   — простой стеммер: отрезает окончания по 5 правилам («лицензию» → «лицензи»);
  • "pymorphy3" — морфологический словарь: даёт начальную форму слова («лицензию» → «лицензия»,
                  «присоединились» → «присоединиться»). Нужна библиотека pymorphy3;
  • "auto"      — pymorphy3, если он установлен, иначе стеммер.
"""
from __future__ import annotations

import re
from typing import Callable, Iterable, Optional, Protocol, Sequence

Lemmatizer = Callable[[str], str]
"""Функция «слово → основа». Получает уже нормализованное слово."""


def normalize(text: str) -> str:
    """Нижний регистр, «ё» → «е»."""
    return text.lower().replace("ё", "е")


# ---------- Стеммер ----------

_RULES = [
    re.compile(r"(ал|ала|ало|али)$"),
    re.compile(r"(ать|ять|ить|еть|уть|ыть)$"),
    re.compile(r"(ть|ти)$"),
    re.compile(r"(ами|ями|ого|ему|ому|ыми|ими|ая|яя|ое|ее|ые|ие|ый|ий|ой|ей|ую|юю|ах|ях|ов|ев|ам|ям|ом|ем)$"),
    re.compile(r"(а|я|ы|и|у|ю|е|о)$"),
]


def stem(word: str) -> str:
    """Грубый стеммер: по очереди отрезает окончания русских слов (5 правил)."""
    word = word.lower()
    for rule in _RULES:
        word = rule.sub("", word, count=1)
    return word


stem.name = "стеммер"  # type: ignore[attr-defined]


# ---------- Морфология pymorphy3 ----------

class MorphologyUnavailable(RuntimeError):
    """Библиотека pymorphy3 не установлена или её словари не найдены."""


_CYRILLIC = re.compile(r"[а-яё]")


class MorphLemmatizer:
    """Начальная форма слова по словарю pymorphy3 («рутокена» → «рутокен»).

    Слова без кириллицы («ngate», «5.0», «r4») оставляются как есть.
    Результаты запоминаются: одно слово разбирается только один раз.
    """

    name = "pymorphy3"

    def __init__(self, analyzer=None):
        if analyzer is None:
            analyzer = _create_analyzer()
        self._morph = analyzer
        self._cache: dict[str, str] = {}

    def __call__(self, word: str) -> str:
        word = normalize(word)
        cached = self._cache.get(word)
        if cached is not None:
            return cached
        result = word
        if _CYRILLIC.search(word):
            parses = self._morph.parse(word)
            if parses:
                result = normalize(parses[0].normal_form) or word
        self._cache[word] = result
        return result


def _create_analyzer():
    try:
        import pymorphy3  # type: ignore[import-not-found]
    except ImportError as ex:
        raise MorphologyUnavailable(
            "Библиотека pymorphy3 не установлена. Установите её командой: python -m pip install pymorphy3"
        ) from ex

    # Явный путь к словарям нужен для собранного CryptoChat.exe: там поиск словарей
    # через метаданные пакета может не работать.
    path = None
    try:
        import pymorphy3_dicts_ru  # type: ignore[import-not-found]
        path = pymorphy3_dicts_ru.get_path()
    except (ImportError, AttributeError):
        path = None

    try:
        return pymorphy3.MorphAnalyzer(path=path, lang="ru") if path else pymorphy3.MorphAnalyzer(lang="ru")
    except Exception as ex:  # словари не найдены или повреждены
        raise MorphologyUnavailable(f"Не удалось загрузить словари pymorphy3: {ex}") from ex


MORPHOLOGY_MODES = ("stemmer", "pymorphy3", "auto")


def make_lemmatizer(mode: str = "stemmer") -> tuple[Lemmatizer, Optional[str]]:
    """Создаёт функцию приведения слова по режиму из config.json.

    Возвращает (функция, примечание). Примечание не пустое, если режим «auto»
    пришлось заменить стеммером, потому что pymorphy3 не установлен.
    """
    mode = (mode or "stemmer").lower()
    if mode == "stemmer":
        return stem, None
    if mode == "pymorphy3":
        return MorphLemmatizer(), None  # MorphologyUnavailable, если библиотеки нет
    if mode == "auto":
        try:
            return MorphLemmatizer(), None
        except MorphologyUnavailable:
            return stem, "pymorphy3 не установлен, используется стеммер"
    raise ValueError(f"Неизвестный режим морфологии «{mode}». Допустимо: {', '.join(MORPHOLOGY_MODES)}.")


def lemmatizer_name(lemmatize: Lemmatizer) -> str:
    return getattr(lemmatize, "name", "своя функция")


# ---------- Синонимы ----------

class SynonymError(ValueError):
    """Ошибка в группах синонимов."""


class SynonymMap:
    """Группы синонимов. Первое слово группы — каноническое (одно слово).

    Остальные члены заменяются на него: одиночные слова — на уровне основ,
    фразы из нескольких слов — в тексте до разбиения на слова.
    """

    def __init__(self, groups: Iterable[Sequence[str]] = (), lemmatize: Lemmatizer = stem):
        self._phrases: list[tuple[re.Pattern[str], str]] = []
        self._lemma_map: dict[str, str] = {}
        owner: dict[str, int] = {}  # основа → номер группы, в которую она уже входит

        def claim(lemma: str, group_index: int, word: str) -> None:
            # Внутри одной группы совпадение основ допустимо («купил», «куплю» → «купить»),
            # между разными группами — нет.
            if owner.setdefault(lemma, group_index) != group_index:
                raise SynonymError(f"Слово «{word}» входит в несколько групп синонимов.")

        for index, group in enumerate(groups):
            if len(group) < 2:
                raise SynonymError("Группа синонимов должна содержать минимум два слова.")

            canonical = normalize(group[0]).strip()
            if " " in canonical:
                raise SynonymError(f"Каноническое слово группы должно быть одним словом: «{canonical}».")
            canonical_lemma = lemmatize(canonical)
            claim(canonical_lemma, index, canonical)

            for raw in group[1:]:
                member = normalize(raw).strip()
                if " " in member:
                    pattern = re.compile(r"(?<![^\W_])" + re.escape(member) + r"(?![^\W_])")
                    self._phrases.append((pattern, canonical))
                    continue
                member_lemma = lemmatize(member)
                claim(member_lemma, index, member)
                if member_lemma != canonical_lemma:
                    self._lemma_map[member_lemma] = canonical_lemma

    def replace_phrases(self, normalized_text: str) -> str:
        """Заменяет фразы-синонимы в уже нормализованном тексте."""
        for pattern, canonical in self._phrases:
            normalized_text = pattern.sub(canonical, normalized_text)
        return normalized_text

    def map_lemma(self, lemma: str) -> str:
        """Заменяет основу слова на каноническую, если она входит в группу."""
        return self._lemma_map.get(lemma, lemma)


# ---------- Конвейер ----------

DEFAULT_STOP_WORDS = (
    "как", "что", "где", "какое", "для", "мне", "можно", "нужно", "ли",
    "вы", "вас", "ваш", "меня", "у", "я", "не", "в", "на", "с", "по",
    "это", "если", "или", "а", "но", "же", "бы", "чтобы", "хочу", "хотел",
    "вижу", "видеть", "видел", "есть", "был", "была", "было", "были",
    "такое", "какие", "какой", "какая", "такой", "при", "из", "от", "до",
    "за", "к", "о", "об", "то", "так",
)

# Слово — это буквы, цифры и точки (чтобы «5.0» оставалось одним словом).
_TOKEN = re.compile(r"(?:[^\W_]|\.)+")


class TextProcessor(Protocol):
    def lemmatize(self, text: str) -> list[str]: ...


class DefaultTextProcessor:
    """Нормализация → синонимы-фразы → слова → стоп-слова → основа слова → синонимы-слова."""

    def __init__(self, stop_words: Iterable[str] | None = None, synonyms: SynonymMap | None = None,
                 lemmatize: Lemmatizer = stem):
        self._stop_words = {normalize(w) for w in (stop_words if stop_words is not None else DEFAULT_STOP_WORDS)}
        self._synonyms = synonyms or SynonymMap(lemmatize=lemmatize)
        self._lemmatize = lemmatize

    def lemmatize(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []

        normalized = self._synonyms.replace_phrases(normalize(text))
        lemmas: list[str] = []
        for match in _TOKEN.finditer(normalized):
            token = match.group(0).strip(".")
            if len(token) < 2 or token in self._stop_words:
                continue
            lemma = self._lemmatize(token)
            if not lemma:
                continue
            lemmas.append(self._synonyms.map_lemma(lemma))
        return lemmas
