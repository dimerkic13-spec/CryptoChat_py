"""Консольная программа: только ввод и вывод. Вся логика живёт в ядре (engine.py и ниже)."""
from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

from . import __version__, knowledge
from .engine import EXIT_WORDS, ChatEngine, create_lemmatizer
from .mailer import MailError, send_pending, send_test
from .session import SessionRecorder
from .models import AppConfig, ChatReply, ContentError, FeedbackEntry, ReplyKind

GOODBYE = "До свидания!"


# ---------- Ключи запуска ----------

@dataclass
class Options:
    content_path: Optional[str] = None
    validate: bool = False
    selftest: bool = False
    compare: bool = False
    morphology: Optional[str] = None
    debug: bool = False
    test_email: bool = False
    send_pending: bool = False
    show_version: bool = False
    show_help: bool = False
    error: Optional[str] = None

    @staticmethod
    def parse(args: list[str]) -> "Options":
        options = Options()
        i = 0
        while i < len(args):
            arg = args[i].lower()
            if arg == "--content":
                if i + 1 >= len(args):
                    options.error = "После --content нужно указать путь к каталогу."
                else:
                    i += 1
                    options.content_path = args[i]
            elif arg == "--validate":
                options.validate = True
            elif arg == "--selftest":
                options.selftest = True
            elif arg == "--compare":
                options.compare = True
            elif arg == "--morphology":
                if i + 1 >= len(args):
                    options.error = "После --morphology нужно указать stemmer, pymorphy3 или auto."
                else:
                    i += 1
                    options.morphology = args[i].lower()
            elif arg == "--debug":
                options.debug = True
            elif arg == "--test-email":
                options.test_email = True
            elif arg == "--send-pending":
                options.send_pending = True
            elif arg == "--version":
                options.show_version = True
            elif arg in ("--help", "-h", "/?"):
                options.show_help = True
            else:
                options.error = f"Неизвестный ключ: {args[i]}"
            i += 1
        return options


# ---------- Где лежит программа и данные ----------

def app_directory() -> Path:
    """Папка программы: рядом с CryptoChat.exe (сборка PyInstaller) или корень проекта (запуск из исходников)."""
    if getattr(sys, "frozen", False):
        return Path(sys.executable).resolve().parent
    return Path(__file__).resolve().parent.parent


def resolve_content_directory(options: Options) -> Path:
    if options.content_path and options.content_path.strip():
        return Path(options.content_path).resolve()

    from_env = os.environ.get("CRYPTOCHAT_CONTENT")
    if from_env and from_env.strip():
        return Path(from_env).resolve()

    beside_program = app_directory() / "content"
    if beside_program.is_dir():
        return beside_program

    in_current = Path.cwd() / "content"
    return in_current if in_current.is_dir() else beside_program


def _log_path(path: str) -> Path:
    p = Path(path)
    return p if p.is_absolute() else app_directory() / p


# ---------- Журналы ----------

class UnansweredLog:
    """Необязательный журнал вопросов, на которые бот не смог сразу ответить: он уточнял тему
    или направил в поддержку (config.json → logging.unanswered). Ошибки записи игнорируются."""

    KINDS = {ReplyKind.ESCALATE, ReplyKind.NOT_UNDERSTOOD, ReplyKind.CLARIFY_BRANCH}

    def __init__(self, config: AppConfig, session_id: Optional[str] = None):
        self._path = _log_path(config.logging.path) if config.logging.unanswered else None
        self._session = session_id

    def write(self, query: str, kind: ReplyKind) -> None:
        if self._path is None or kind not in self.KINDS:
            return
        _append_json(self._path, {"time": _now(), "session": self._session, "query": query, "kind": kind.value})


class FeedbackLog:
    """Необязательный журнал отзывов об ответах (config.json → logging.feedback)."""

    def __init__(self, config: AppConfig, session_id: Optional[str] = None):
        self._path = _log_path(config.logging.feedback_path) if config.logging.feedback else None
        self._session = session_id

    def write(self, entry: FeedbackEntry) -> None:
        if self._path is None:
            return
        _append_json(self._path, {
            "time": entry.time.isoformat(timespec="seconds"),
            "session": self._session,
            "query": entry.query,
            "questionId": entry.question_id,
            "helpful": entry.helpful,
        })


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def _append_json(path: Path, record: dict) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    except OSError:
        pass  # журнал не должен ломать диалог


# ---------- Самопроверка ----------

def _first_question_id(reply: ChatReply) -> Optional[str]:
    if reply.kind is ReplyKind.ANSWER:
        return reply.question.id if reply.question else None
    if reply.kind is ReplyKind.CLARIFY:
        return reply.options[0].question.id if reply.options and reply.options[0].question else None
    return None


def check_acceptance(engine: ChatEngine, content_dir: Path) -> tuple[list[str], int, int]:
    """Прогоняет acceptance.json и заголовки всех вопросов. Возвращает (провалы, число проверок, число кейсов)."""
    failures: list[str] = []
    checked = 0
    cases_count = 0

    def fresh() -> ChatEngine:
        return ChatEngine(engine.knowledge_base, engine.config)

    path = content_dir / "acceptance.json"
    if path.exists():
        cases = json.loads(path.read_text(encoding="utf-8-sig"))
        cases_count = len(cases)
        for case in cases:
            checked += 1
            q = case.get("q", "")
            reply = fresh().ask(q)
            first = _first_question_id(reply)
            expect = case.get("expect")
            expect_kind = case.get("expectKind")
            if expect is not None and first != expect:
                failures.append(f"«{q}»: ожидался {expect}, получено {reply.kind.value} {first}")
            if expect_kind is not None and reply.kind.value not in expect_kind.split("|"):
                failures.append(f"«{q}»: ожидался тип {expect_kind}, получено {reply.kind.value}")

    for question in engine.knowledge_base.questions:
        checked += 1
        reply = fresh().ask(question.title)
        first = _first_question_id(reply)
        if first != question.id:
            failures.append(f"заголовок {question.id} «{question.title}»: получено {reply.kind.value} {first}")

    return failures, checked, cases_count


def run_selftest(engine: ChatEngine, content_dir: Path) -> int:
    failures, checked, cases = check_acceptance(engine, content_dir)
    if (content_dir / "acceptance.json").exists():
        print(f"Приёмочных вопросов: {cases}")
    else:
        print("Файл acceptance.json не найден, приёмочные вопросы пропущены.")
    print(f"Заголовков вопросов: {len(engine.knowledge_base.questions)}")
    for failure in failures:
        print("  ПРОВАЛ: " + failure)
    if failures:
        print(f"Самопроверка НЕ пройдена: {len(failures)} из {checked}.")
        return 1
    print(f"Самопроверка пройдена ({checked} проверок).")
    return 0


# ---------- Сравнение стеммера и морфологии ----------

def _outcome(engine: ChatEngine, query: str) -> tuple[str, Optional[str], float]:
    reply = ChatEngine(engine.knowledge_base, engine.config).ask(query)
    first = _first_question_id(reply)
    score = reply.score if reply.kind is ReplyKind.ANSWER else (reply.options[0].score if reply.options else 0.0)
    return reply.kind.value, first, score


def run_compare(content_dir: Path) -> int:
    """Прогоняет контрольные запросы и заголовки двумя способами и показывает, где результаты различаются."""
    stemmer = ChatEngine.create(content_dir, "stemmer")
    try:
        morph = ChatEngine.create(content_dir, "pymorphy3")
    except ContentError as ex:
        print("Сравнение невозможно: " + str(ex))
        return 1

    cases = []
    path = content_dir / "acceptance.json"
    if path.exists():
        for case in json.loads(path.read_text(encoding="utf-8-sig")):
            cases.append((case.get("q", ""), case.get("expect"), case.get("expectKind")))
    for question in stemmer.knowledge_base.questions:
        cases.append((question.title, question.id, None))

    def ok(result, expect, expect_kind):
        kind, first, _ = result
        return (expect is None or first == expect) and (expect_kind is None or kind in expect_kind.split("|"))

    passed = {"stemmer": 0, "pymorphy3": 0}
    differences = []
    for query, expect, expect_kind in cases:
        a = _outcome(stemmer, query)
        b = _outcome(morph, query)
        passed["stemmer"] += ok(a, expect, expect_kind)
        passed["pymorphy3"] += ok(b, expect, expect_kind)
        if a[:2] != b[:2] or ok(a, expect, expect_kind) != ok(b, expect, expect_kind):
            differences.append((query, expect or expect_kind, a, b, ok(a, expect, expect_kind), ok(b, expect, expect_kind)))

    def show(result, good):
        kind, first, score = result
        return f"{'OK ' if good else 'ПРОВАЛ'} {kind} {first or '-'} ({score:.2f})"

    print(f"Проверок: {len(cases)}")
    print(f"  стеммер:   прошло {passed['stemmer']}")
    print(f"  pymorphy3: прошло {passed['pymorphy3']}")
    if differences:
        print(f"\nРазличия ({len(differences)}):")
        for query, expected, a, b, good_a, good_b in differences:
            print(f"  «{query}»  ожидалось: {expected}")
            print(f"      стеммер:   {show(a, good_a)}")
            print(f"      pymorphy3: {show(b, good_b)}")
            print(f"      основы:    {', '.join(stemmer._text.lemmatize(query))}  |  "
                  f"{', '.join(morph._text.lemmatize(query))}")
    else:
        print("Различий нет.")
    return 0


# ---------- Почта ----------

def run_email_command(content_dir: Path, options: Options) -> int:
    config = AppConfig.load(content_dir)
    email = config.email
    if not email.enabled:
        print("Отправка на почту выключена: в config.json поставьте \"email\": {\"enabled\": true, ...}.")
        return 1
    try:
        if options.test_email:
            send_test(email, app_directory())
            print(f"Тестовое письмо отправлено на {', '.join(email.recipients())}.")
        if options.send_pending:
            sent = send_pending(email, app_directory(), sessions_folder(config))
            print(f"Отправлено журналов: {sent}.")
    except MailError as ex:
        print("Ошибка: " + str(ex))
        return 1
    return 0


# ---------- Проверка данных ----------

def run_validate(content_dir: Path, morphology: Optional[str]) -> int:
    config = AppConfig.load(content_dir)
    lemmatize, note = create_lemmatizer(morphology or config.text.morphology)
    content = knowledge.load_content(content_dir)
    issues = knowledge.validate(content, lemmatize=lemmatize)
    print(f"Каталог с данными: {content_dir}")
    print(f"Морфология: {getattr(lemmatize, 'name', '?')}" + (f" ({note})" if note else ""))
    print(f"Вопросов: {len(content.questions)}")
    for issue in issues:
        print("  " + str(issue))
    errors = sum(1 for i in issues if i.severity is knowledge.Severity.ERROR)
    print(f"Ошибок: {errors}, предупреждений: {len(issues) - errors}")
    return 0 if errors == 0 else 1


# ---------- Диалог ----------

def print_reply(reply: ChatReply) -> None:
    if reply.debug:
        print()
        print(reply.debug)
    lines = reply.text.split("\n")
    print()
    print("Бот: " + lines[0])
    for line in lines[1:]:
        print("     " + line)


def sessions_folder(config: AppConfig) -> Path:
    return _log_path(config.sessions.path)


def _send_logs(config: AppConfig, skip: Optional[str] = None) -> None:
    """Досылает готовые журналы на почту. Ошибка отправки не мешает работе: журнал уйдёт в следующий раз."""
    if not (config.email.enabled and config.sessions.enabled):
        return
    try:
        send_pending(config.email, app_directory(), sessions_folder(config), skip=skip)
    except MailError as ex:
        print(f"Журнал не отправлен: {ex}")
        print("Он сохранён и будет отправлен при следующем запуске.")


def chat(engine: ChatEngine) -> int:
    kb = engine.knowledge_base
    config = engine.config

    recorder = None
    if config.sessions.enabled:
        recorder = SessionRecorder(sessions_folder(config), __version__, engine.morphology)
        if recorder.available:
            engine.session_id = recorder.session_id
        else:
            print(f"Не удалось создать журнал беседы в {sessions_folder(config)}; беседа не сохраняется.")
            recorder = None

    log = UnansweredLog(config, engine.session_id)
    feedback_log = FeedbackLog(config, engine.session_id)
    engine.feedback_listeners.append(feedback_log.write)

    if config.email.enabled and config.email.send_pending_on_start:
        _send_logs(config, skip=engine.session_id)

    try:
        return _chat_loop(engine, kb, recorder, log)
    finally:
        if recorder is not None:
            recorder.close()
            if config.email.enabled:
                print(f"Отправляю журнал беседы на {', '.join(config.email.recipients())}...")
            _send_logs(config)


def _chat_loop(engine: ChatEngine, kb, recorder: Optional[SessionRecorder], log: "UnansweredLog") -> int:
    print(f"CryptoChat {__version__}")
    parts = "; ".join(f"{r.name}: {len(kb.questions_under(r.id))}" for r in kb.roots)
    print(f"Загружено тем: {len(kb.roots)}, вопросов: {len(kb.questions)} ({parts})")
    note = f" ({engine.morphology_note})" if engine.morphology_note else ""
    print(f"Морфология: {engine.morphology}{note}")
    if engine.session_id:
        print(f"Номер сессии: {engine.session_id}")
    print("=" * 60)
    print("  Помощник КриптоПро")
    print("  Задайте вопрос обычным текстом. Подсказка: /помощь. Выход: выход")
    print("=" * 60)

    awaiting_choice = False
    while True:
        try:
            line = input("\nВаш выбор: " if awaiting_choice else "\nВы: ")
        except (EOFError, KeyboardInterrupt):
            print()
            print("Бот: " + GOODBYE)
            return 0
        if not line.strip():
            continue

        if recorder is not None and line.strip().lower() not in EXIT_WORDS:
            recorder.user(line)
        reply = engine.ask(line)
        if recorder is not None:
            recorder.bot(reply)
        print_reply(reply)
        if not line.strip().isdigit():
            log.write(line, reply.kind)

        if reply.kind is ReplyKind.EXIT:
            return 0
        awaiting_choice = reply.kind in (ReplyKind.CLARIFY, ReplyKind.CLARIFY_PRODUCT, ReplyKind.CLARIFY_BRANCH)


def print_usage() -> None:
    print("CryptoChat — помощник по вопросам КриптоПро")
    print()
    print("Запуск:  CryptoChat [ключи]")
    print("  --content <путь>   каталог с данными (иначе ./content рядом с программой)")
    print("  --validate         проверить данные и выйти")
    print("  --selftest         прогнать приёмочные вопросы и выйти")
    print("  --compare          сравнить стеммер и pymorphy3 на приёмочных вопросах")
    print("  --morphology <m>   stemmer, pymorphy3 или auto (вместо text.morphology из config.json)")
    print("  --debug            показывать, как бот ищет ответ")
    print("  --test-email       отправить тестовое письмо по настройкам email из config.json")
    print("  --send-pending     отправить на почту журналы, которые не ушли раньше")
    print("  --version          показать версию")
    print("  --help             эта подсказка")


def _setup_console() -> None:
    """Русский текст в консоли Windows: UTF-8 для перенаправленного ввода и вывода."""
    for stream in (sys.stdout, sys.stderr, sys.stdin):
        try:
            if not stream.isatty():
                stream.reconfigure(encoding="utf-8")
        except (AttributeError, ValueError, OSError):
            pass


def main(argv: Optional[list[str]] = None) -> int:
    _setup_console()
    options = Options.parse(sys.argv[1:] if argv is None else argv)
    if options.error:
        print(options.error, file=sys.stderr)
        print_usage()
        return 2
    if options.show_version:
        print(f"CryptoChat {__version__}")
        return 0
    if options.show_help:
        print_usage()
        return 0

    content_dir = resolve_content_directory(options)
    try:
        if options.validate:
            return run_validate(content_dir, options.morphology)
        if options.compare:
            return run_compare(content_dir)
        if options.test_email or options.send_pending:
            return run_email_command(content_dir, options)
        engine = ChatEngine.create(content_dir, options.morphology)
        engine.debug_enabled = options.debug
        if options.selftest:
            return run_selftest(engine, content_dir)
        return chat(engine)
    except ContentError as ex:
        print("Ошибка: " + str(ex), file=sys.stderr)
        print(f"Каталог с данными: {content_dir}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
