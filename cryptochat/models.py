"""Модели данных: раздел, вопрос, ключевое слово, ответ бота, настройки."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, fields, is_dataclass
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Optional


class ContentError(Exception):
    """Ошибка в данных каталога content/: файл не найден, сломан JSON, данные не прошли проверку."""


# ---------- База знаний ----------

@dataclass(frozen=True)
class Keyword:
    """Ключевое слово: основа слова (после стемминга) и вес."""
    word: str
    weight: float


@dataclass(eq=False)
class Category:
    """Раздел дерева: тема, продукт, подтема, блок или подблок."""
    id: str
    parent_id: Optional[str]
    name: str
    kind: str
    contact: Optional[str]
    keywords: list[Keyword]
    children: list["Category"] = field(default_factory=list)


@dataclass(eq=False)
class Question:
    """Вопрос с готовым ответом."""
    id: str
    category_id: str
    title: str
    answer: str
    source_url: str
    anchor: Optional[str]
    revision: Optional[str]
    is_active: bool
    keywords: list[Keyword]
    # Другие формулировки того же вопроса — для поиска по смыслу (необязательно).
    examples: list[str] = field(default_factory=list)


# ---------- Ответ бота ----------

class ReplyKind(Enum):
    ANSWER = "Answer"
    CLARIFY = "Clarify"
    CLARIFY_PRODUCT = "ClarifyProduct"
    CLARIFY_BRANCH = "ClarifyBranch"
    ESCALATE = "Escalate"
    NOT_UNDERSTOOD = "NotUnderstood"
    FEEDBACK = "Feedback"
    COMMAND = "Command"
    EXIT = "Exit"


@dataclass(frozen=True)
class ChatOption:
    """Один пункт меню: вопрос, продукт или ветка дерева (тема, блок)."""
    number: int
    label: str
    score: float
    question: Optional[Question] = None
    product: Optional[Category] = None
    branch: Optional[Category] = None
    # Пункт «посмотреть по списку»: показать вопросы раздела branch (или список тем, если branch нет)
    # без нового поиска — так пользователь доходит до нужного вопроса, даже если меню не угадало.
    browse: bool = False


@dataclass(frozen=True)
class ChatReply:
    """Реакция бота на ввод пользователя. Консоль только печатает text."""
    kind: ReplyKind
    text: str
    options: tuple[ChatOption, ...] = ()
    debug: Optional[str] = None
    question: Optional[Question] = None
    score: float = 0.0


@dataclass(frozen=True)
class FeedbackEntry:
    """Отзыв пользователя об ответе («да» — помог, «нет» — не помог)."""
    time: datetime
    query: str
    question_id: str
    helpful: bool


# ---------- Настройки (content/config.json) ----------

@dataclass
class ThresholdsConfig:
    answer: float = 0.6
    clarify: float = 0.3
    scope_min: float = 0.1
    scope_margin: float = 0.05
    dominance_gap: float = 0.2


@dataclass
class ScoringConfig:
    scope_boost: float = 0.25
    prefix_min_length: int = 4
    # true: ключи любой категории могут сузить поиск до её ветки.
    # false: сужают поиск только продукты (kind = "product"), а ключи тем и блоков
    # служат для уточнения по веткам.
    category_scope: bool = True


@dataclass
class SessionsConfig:
    # Сохранять каждую беседу в отдельный файл с номером сессии.
    enabled: bool = True
    path: str = "logs/sessions"


@dataclass
class DatabaseConfig:
    # База бесед SQLite: сессии, сообщения, отзывы, неотвеченные запросы (для --stats).
    enabled: bool = True
    path: str = "logs/cryptochat.db"


@dataclass
class EmailConfig:
    # Отправлять журнал беседы на почту после закрытия программы.
    enabled: bool = False
    smtp_host: str = ""
    smtp_port: int = 465
    security: str = "ssl"          # "ssl" (порт 465), "starttls" (порт 587) или "none"
    username: str = ""             # логин ящика-отправителя
    sender: str = ""               # адрес «От кого»; пусто — берётся username
    to: list = field(default_factory=list)  # кому: список адресов или одна строка через запятую
    subject_prefix: str = "CryptoChat"
    password_env: str = "CRYPTOCHAT_SMTP_PASSWORD"  # переменная окружения с паролем
    send_empty: bool = False       # отправлять ли беседы без единого вопроса
    send_pending_on_start: bool = True  # при запуске досылать то, что не ушло в прошлый раз
    timeout_seconds: int = 20

    def recipients(self) -> list[str]:
        items = self.to if isinstance(self.to, list) else [self.to]
        # Каждый элемент может содержать несколько адресов через запятую или точку с запятой.
        result = []
        for item in items:
            for address in str(item).replace(";", ",").split(","):
                if address.strip():
                    result.append(address.strip())
        return result

    def from_address(self) -> str:
        return (self.sender or self.username).strip()


@dataclass
class TextConfig:
    # Как приводить слово к основе: "stemmer" (простой стеммер), "pymorphy3" (морфологический словарь)
    # или "auto" (pymorphy3, если установлен, иначе стеммер).
    morphology: str = "stemmer"


@dataclass
class FuzzyConfig:
    # Учёт опечаток (RapidFuzz, без него — встроенный difflib).
    enabled: bool = False
    threshold: float = 85      # минимальная похожесть слов, 0–100
    min_length: int = 5        # слова короче не сравниваются нечётко
    weight: float = 0.8        # какую долю веса даёт ключ, совпавший с опечаткой


@dataclass
class SemanticConfig:
    # Поиск по смыслу: sentence-transformers на PyTorch (необязательно).
    enabled: bool = False
    model: str = "cointegrated/rubert-tiny2"   # небольшая русская модель (~120 МБ)
    cache_dir: str = "models"                  # куда скачивается модель (рядом с программой)
    device: str = "cpu"                        # "cpu" или "cuda" (видеокарта NVIDIA)
    local_files_only: bool = False             # true — не ходить в интернет, брать модель только из cache_dir
    mode: str = "relative"                     # "relative" (на фоне остальных вопросов) или "absolute"
    z_low: float = 0.0                         # relative: z-оценка ниже — смысловая оценка 0
    z_high: float = 5.0                        # relative: z-оценка выше — смысловая оценка = weight
    top_k: int = 3                             # relative: сколько самых похожих вопросов учитывать
    low: float = 0.55                          # absolute: похожесть ниже — смысловая оценка 0
    high: float = 0.9                          # absolute: похожесть выше — смысловая оценка = weight
    weight: float = 0.3                        # максимальная оценка от смысла
    combine: str = "or"                        # "or" (смысл усиливает ключи) или "max"
    # Запасное меню: если ключи не нашли ничего, показать самые похожие по смыслу вопросы
    # вместо меню тем. Вопрос попадает в меню, если его z-оценка не ниже menu_z.
    menu_fallback: bool = True
    menu_z: float = 1.5
    menu_size: int = 3
    query_prefix: str = ""                     # для моделей e5/USER: "query: "
    passage_prefix: str = ""                   # для моделей e5/USER: "passage: "


@dataclass
class MenuConfig:
    max_options: int = 3


@dataclass
class SupportConfig:
    url: str = "support.cryptopro.ru"


@dataclass
class LoggingConfig:
    unanswered: bool = False
    path: str = "logs/unanswered.jsonl"
    feedback: bool = False
    feedback_path: str = "logs/feedback.jsonl"


@dataclass
class DialogConfig:
    # Управляемый диалог. Выключен: бот сразу направляет в поддержку, если не уверен.
    guided: bool = False
    # Сколько шагов уточнения бот делает, прежде чем направить в поддержку.
    escalate_after_steps: int = 4
    # Сколько веток или вопросов показывать в меню уточнения.
    max_branch_options: int = 7
    # Минимальная оценка вопроса для списка «Похожие вопросы» после ответа.
    min_score: float = 0.05
    # Если подходящих вопросов нет, вопросы с оценкой по ключам не ниже этой предлагаются
    # как «Возможно, вы имели в виду…» вместо меню тем (0 — не предлагать).
    suggest_score: float = 0.15
    # Дозаполнять меню «Уточните…» до menu.maxOptions вопросами с оценкой не ниже этой (0 — не дозаполнять).
    menu_fill_score: float = 0.0
    # Добавлять в меню вопросов пункты «Другие вопросы раздела …» и «Другая тема — выбрать из списка тем».
    browse_options: bool = False
    # Спрашивать после ответа, помог ли он.
    feedback: bool = True


def _snake(name: str) -> str:
    """dominanceGap → dominance_gap."""
    return re.sub(r"(?<!^)(?=[A-Z])", "_", name).lower()


_ALIASES = {"from": "sender"}  # «from» нельзя назвать полем в Python


def _apply(target: Any, data: Any, where: str) -> None:
    """Переносит значения из JSON в объект настроек. Неизвестные ключи пропускаются."""
    if not isinstance(data, dict):
        raise ContentError(f"{where}: ожидался объект {{ ... }}.")
    known = {f.name: f for f in fields(target)}
    for key, value in data.items():
        name = _ALIASES.get(_snake(key), _snake(key))
        if name not in known:
            continue
        current = getattr(target, name)
        if is_dataclass(current):
            _apply(current, value, f"{where}.{key}")
        else:
            if isinstance(current, bool):
                if not isinstance(value, bool):
                    raise ContentError(f"{where}.{key}: ожидалось true или false.")
            elif isinstance(current, (int, float)):
                if isinstance(value, bool) or not isinstance(value, (int, float)):
                    raise ContentError(f"{where}.{key}: ожидалось число.")
                value = int(value) if isinstance(current, int) else float(value)
            elif isinstance(current, str) and not isinstance(value, str):
                raise ContentError(f"{where}.{key}: ожидалась строка.")
            elif isinstance(current, list):
                if isinstance(value, str):
                    value = [value]
                if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
                    raise ContentError(f"{where}.{key}: ожидался список строк [\"...\"].")
            setattr(target, name, value)


@dataclass
class AppConfig:
    """Параметры поведения бота. Значения по умолчанию заданы здесь, файл config.json их переопределяет."""
    thresholds: ThresholdsConfig = field(default_factory=ThresholdsConfig)
    scoring: ScoringConfig = field(default_factory=ScoringConfig)
    menu: MenuConfig = field(default_factory=MenuConfig)
    support: SupportConfig = field(default_factory=SupportConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    dialog: DialogConfig = field(default_factory=DialogConfig)
    text: TextConfig = field(default_factory=TextConfig)
    fuzzy: FuzzyConfig = field(default_factory=FuzzyConfig)
    semantic: SemanticConfig = field(default_factory=SemanticConfig)
    sessions: SessionsConfig = field(default_factory=SessionsConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    email: EmailConfig = field(default_factory=EmailConfig)

    @staticmethod
    def load(directory: str | Path) -> "AppConfig":
        path = Path(directory) / "config.json"
        config = AppConfig()
        if not path.exists():
            return config
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError as ex:
            raise ContentError(f"Ошибка в файле {path}: {ex}") from ex
        _apply(config, data, str(path))
        config.validate(str(path))
        return config

    def validate(self, source: str = "config.json") -> None:
        t = self.thresholds
        if t.clarify <= 0 or t.answer > 1 or t.clarify > t.answer:
            raise ContentError(f"{source}: должно выполняться 0 < clarify <= answer <= 1.")
        if self.menu.max_options < 1:
            raise ContentError(f"{source}: menu.maxOptions должно быть не меньше 1.")
        if self.dialog.escalate_after_steps < 1:
            raise ContentError(f"{source}: dialog.escalateAfterSteps должно быть не меньше 1.")
        if self.dialog.max_branch_options < 2:
            raise ContentError(f"{source}: dialog.maxBranchOptions должно быть не меньше 2.")
        if not 0 <= self.dialog.suggest_score <= 1:
            raise ContentError(f"{source}: dialog.suggestScore должно быть от 0 до 1.")
        if not 0 <= self.dialog.menu_fill_score <= 1:
            raise ContentError(f"{source}: dialog.menuFillScore должно быть от 0 до 1.")
        if not 0 <= self.dialog.min_score <= 1:
            raise ContentError(f"{source}: dialog.minScore должно быть от 0 до 1.")
        if self.text.morphology.lower() not in ("stemmer", "pymorphy3", "auto"):
            raise ContentError(f"{source}: text.morphology должно быть stemmer, pymorphy3 или auto.")
        s = self.semantic
        if not -1 <= s.low < s.high <= 1:
            raise ContentError(f"{source}: semantic.low и semantic.high — от -1 до 1, low меньше high.")
        if not 0 < s.weight <= 1:
            raise ContentError(f"{source}: semantic.weight должно быть больше 0 и не больше 1.")
        if s.mode not in ("relative", "absolute"):
            raise ContentError(f"{source}: semantic.mode должно быть relative или absolute.")
        if s.combine not in ("max", "or"):
            raise ContentError(f"{source}: semantic.combine должно быть max или or.")
        if not s.z_low < s.z_high:
            raise ContentError(f"{source}: semantic.zLow должно быть меньше semantic.zHigh.")
        if s.menu_size < 1:
            raise ContentError(f"{source}: semantic.menuSize должно быть не меньше 1.")
        if s.top_k < 1:
            raise ContentError(f"{source}: semantic.topK должно быть не меньше 1.")
        if s.enabled and not s.model.strip():
            raise ContentError(f"{source}: semantic.model не задан.")
        if not 50 <= self.fuzzy.threshold <= 100:
            raise ContentError(f"{source}: fuzzy.threshold должно быть от 50 до 100.")
        if not 0 < self.fuzzy.weight <= 1:
            raise ContentError(f"{source}: fuzzy.weight должно быть больше 0 и не больше 1.")
        if self.fuzzy.min_length < 3:
            raise ContentError(f"{source}: fuzzy.minLength должно быть не меньше 3.")
        if self.email.security.lower() not in ("ssl", "starttls", "none"):
            raise ContentError(f"{source}: email.security должно быть ssl, starttls или none.")
        if self.email.enabled:
            if not self.email.smtp_host.strip():
                raise ContentError(f"{source}: email.smtpHost не задан (например smtp.yandex.ru).")
            if not self.email.from_address():
                raise ContentError(f"{source}: email.username или email.sender не задан.")
            if not self.email.recipients():
                raise ContentError(f"{source}: email.to не задан — укажите, кому отправлять журнал.")
        if self.scoring.prefix_min_length < 1:
            raise ContentError(f"{source}: scoring.prefixMinLength должно быть не меньше 1.")
