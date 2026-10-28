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
from .session import SessionRecorder, hostname, new_session_id
from .storage import ChatDatabase, stats_report
from .models import AppConfig, ChatReply, ContentError, FeedbackEntry, ReplyKind

GOODBYE = "До свидания!"


# ---------- Ключи запуска ----------

@dataclass
class Options:
    content_path: Optional[str] = None
    validate: bool = False
    selftest: bool = False
    compare: bool = False
    compare_semantic: bool = False
    download_model: bool = False
    semantic_dump: bool = False
    morphology: Optional[str] = None
    debug: bool = False
    test_email: bool = False
    send_pending: bool = False
    stats: bool = False
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
            elif arg == "--compare-semantic":
                options.compare_semantic = True
            elif arg == "--download-model":
                options.download_model = True
            elif arg == "--semantic-dump":
                options.semantic_dump = True
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
            elif arg == "--stats":
                options.stats = True
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
        return ChatEngine(engine.knowledge_base, engine.config, semantic=engine.semantic)

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
    reply = ChatEngine(engine.knowledge_base, engine.config, semantic=engine.semantic).ask(query)
    first = _first_question_id(reply)
    score = reply.score if reply.kind is ReplyKind.ANSWER else (reply.options[0].score if reply.options else 0.0)
    return reply.kind.value, first, score


def run_compare(content_dir: Path) -> int:
    """Прогоняет контрольные запросы и заголовки двумя способами и показывает, где результаты различаются."""
    stemmer = ChatEngine.create(content_dir, "stemmer", semantic=False)
    try:
        morph = ChatEngine.create(content_dir, "pymorphy3", semantic=False)
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


# ---------- Статистика ----------

def run_stats(content_dir: Path) -> int:
    config = AppConfig.load(content_dir)
    titles = {}
    try:
        titles = {q.id: q.title for q in knowledge.load_content(content_dir).questions}
    except ContentError:
        pass  # статистика нужна и тогда, когда в базе знаний ошибка
    print(stats_report(database_path(config), titles))
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


# ---------- Модель для поиска по смыслу ----------

def run_download_model(content_dir: Path) -> int:
    """Скачивает модель в папку models/ рядом с программой, чтобы дальше работать без интернета."""
    from .semantic import make_semantic
    config = AppConfig.load(content_dir)
    config.semantic.enabled = True
    config.semantic.local_files_only = False
    print(f"Скачиваю модель {config.semantic.model} в {app_directory() / config.semantic.cache_dir} ...")
    kb = knowledge.load(content_dir)
    index, note = make_semantic(config.semantic, kb.questions, app_directory())
    if index is None:
        print("Ошибка: " + (note or "не удалось загрузить модель."))
        return 1
    print(f"Готово: модель загружена, посчитано {index.phrases} фраз. Дальше она работает без интернета.")
    return 0


# ---------- Сравнение с поиском по смыслу и без ----------

def _semantic_cases(content_dir: Path, engine: ChatEngine) -> list[tuple[str, Optional[str], Optional[str], str]]:
    """(запрос, ожидаемый вопрос, ожидаемый тип, откуда): acceptance.json, заголовки, paraphrases.json."""
    cases = []
    for name in ("acceptance.json", "paraphrases.json"):
        path = content_dir / name
        if path.exists():
            for case in json.loads(path.read_text(encoding="utf-8-sig")):
                cases.append((case.get("q", ""), case.get("expect"), case.get("expectKind"), name))
    for question in engine.knowledge_base.questions:
        cases.append((question.title, question.id, None, "заголовки"))
    return cases


def run_compare_semantic(content_dir: Path, morphology: Optional[str]) -> int:
    """Показывает, где поиск по смыслу помог, а где помешал. По выводу подбираются low, high и weight."""
    print("Загружаю модель...")
    with_model = ChatEngine.create(content_dir, morphology, semantic=True, app_dir=app_directory())
    if with_model.semantic is None:
        print("Сравнение невозможно: " + (with_model.semantic_note or "поиск по смыслу выключен."))
        return 1
    without = ChatEngine(with_model.knowledge_base, with_model.config, semantic=None)
    sem = with_model.semantic
    print(f"Модель: {sem.name}; low={sem.low}, high={sem.high}, weight={sem.weight}")

    def ok(result, expect, expect_kind):
        kind, first, _ = result
        return (expect is None or first == expect) and (expect_kind is None or kind in expect_kind.split("|"))

    totals: dict[str, list[int]] = {}
    better, worse, changed = [], [], []
    for query, expect, expect_kind, source in _semantic_cases(content_dir, with_model):
        a = _outcome(without, query)
        b = _outcome(with_model, query)
        good_a, good_b = ok(a, expect, expect_kind), ok(b, expect, expect_kind)
        t = totals.setdefault(source, [0, 0, 0])
        t[0] += 1; t[1] += good_a; t[2] += good_b
        if good_b and not good_a:
            better.append((query, expect, a, b))
        elif good_a and not good_b:
            worse.append((query, expect, a, b))
        elif a[:2] != b[:2]:
            changed.append((query, expect, a, b))

    print("\nПрошло проверок:          без модели   с моделью")
    for source, (n, good_a, good_b) in totals.items():
        print(f"  {source:22} {good_a:>4} из {n:<4} {good_b:>4} из {n}")

    def show(title, rows):
        if not rows:
            return
        print(f"\n{title} ({len(rows)}):")
        for query, expect, a, b in rows:
            cos = sem.similarities(query)
            best = max(cos.items(), key=lambda x: x[1]) if cos else ("-", 0.0)
            print(f"  «{query}»  ожидалось: {expect or '-'}")
            print(f"      без модели: {a[0]} {a[1] or '-'} ({a[2]:.2f})   с моделью: {b[0]} {b[1] or '-'} ({b[2]:.2f})"
                  f"   самый похожий по смыслу: {best[0]} ({best[1]:.2f})")

    show("Модель помогла", better)
    show("Модель помешала — стоит поднять low или снизить weight", worse)
    show("Изменился тип ответа, но оба правильные", changed)
    return 0


# ---------- Выгрузка для настройки поиска по смыслу ----------

DUMP_FILE = "semantic_dump.json"


def run_semantic_dump(content_dir: Path, morphology: Optional[str]) -> int:
    """Сохраняет для каждого контрольного запроса оценки по ключам и похожести по смыслу для ВСЕХ вопросов.
    По этому файлу подбираются mode, zLow, zHigh, weight и combine."""
    from .retrieval import evaluate
    from .semantic import SemanticIndex
    print("Загружаю модель...")
    engine = ChatEngine.create(content_dir, morphology, semantic=True, app_dir=app_directory())
    if engine.semantic is None:
        print("Выгрузка невозможна: " + (engine.semantic_note or "поиск по смыслу выключен."))
        return 1
    sem = engine.semantic
    kb = engine.knowledge_base
    prefix = engine.config.scoring.prefix_min_length

    records = []
    first_by_cos = in_top3 = first_by_kw = with_expect = 0
    right_cos, wrong_cos, right_z = [], [], []
    for query, expect, expect_kind, source in _semantic_cases(content_dir, engine):
        lemmas = engine._text.lemmatize(query)
        keyword = {q.id: round(evaluate(q.keywords, lemmas, prefix, engine.fuzzy)[0], 4)
                   for q in kb.questions if q.is_active}
        cosine = {k: round(v, 4) for k, v in sem.similarities(query).items()}
        records.append({"q": query, "expect": expect, "expectKind": expect_kind, "source": source,
                        "lemmas": lemmas, "keyword": keyword, "cosine": cosine})
        if expect and cosine:
            with_expect += 1
            ranked = sorted(cosine, key=cosine.get, reverse=True)
            first_by_cos += ranked[0] == expect
            in_top3 += expect in ranked[:3]
            first_by_kw += max(keyword, key=keyword.get) == expect and keyword[expect] > 0
            right_cos.append(cosine.get(expect, 0.0))
            wrong_cos.append(max(v for k, v in cosine.items() if k != expect))
            right_z.append(SemanticIndex.z_scores(cosine).get(expect, 0.0))

    path = Path.cwd() / DUMP_FILE
    data = {"model": sem.name, "morphology": engine.morphology,
            "fuzzy": engine.fuzzy.name if engine.fuzzy else None,
            "settings": {"mode": sem.mode, "zLow": sem.z_low, "zHigh": sem.z_high, "topK": sem.top_k,
                         "low": sem.low, "high": sem.high, "weight": sem.weight,
                         "combine": engine.config.semantic.combine},
            "titles": {q.id: q.title for q in kb.questions},
            "cases": records}
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")

    def avg(values):
        return sum(values) / len(values) if values else 0.0

    print(f"Модель: {sem.name}; запросов с ожидаемым ответом: {with_expect}")
    print(f"  правильный вопрос первый по смыслу:  {first_by_cos} из {with_expect}")
    print(f"  правильный вопрос в тройке по смыслу: {in_top3} из {with_expect}")
    print(f"  правильный вопрос первый по ключам:  {first_by_kw} из {with_expect}")
    print(f"  средняя похожесть: правильного {avg(right_cos):.2f}, лучшего неправильного {avg(wrong_cos):.2f}")
    print(f"  средняя z-оценка правильного: {avg(right_z):.2f}")
    print(f"Файл сохранён: {path}")
    print("Пришлите этот файл — по нему подбираются настройки поиска по смыслу.")
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


def database_path(config: AppConfig) -> Path:
    return _log_path(config.database.path)


def _send_logs(config: AppConfig, skip: Optional[str] = None, db: Optional[ChatDatabase] = None) -> None:
    """Досылает готовые журналы на почту. Ошибка отправки не мешает работе: журнал уйдёт в следующий раз."""
    if not (config.email.enabled and config.sessions.enabled):
        return
    try:
        send_pending(config.email, app_directory(), sessions_folder(config), skip=skip,
                     on_sent=db.mark_emailed if db is not None else None)
    except MailError as ex:
        print(f"Журнал не отправлен: {ex}")
        print("Он сохранён и будет отправлен при следующем запуске.")


class Conversation:
    """Всё, что сохраняется о беседе: файл журнала, база SQLite, журналы отзывов и неотвеченных."""

    def __init__(self, engine: ChatEngine):
        config = engine.config
        self.config = config
        self.session_id: Optional[str] = None
        self.recorder: Optional[SessionRecorder] = None
        self.db: Optional[ChatDatabase] = None

        if config.sessions.enabled or config.database.enabled:
            self.session_id = new_session_id()

        if config.sessions.enabled:
            recorder = SessionRecorder(sessions_folder(config), __version__, engine.morphology, self.session_id)
            if recorder.available:
                self.recorder = recorder
            else:
                print(f"Не удалось создать журнал беседы в {sessions_folder(config)}; файл беседы не пишется.")

        if config.database.enabled:
            db = ChatDatabase(database_path(config))
            if db.available:
                self.db = db
                db.start_session(self.session_id, hostname(), __version__, engine.morphology)
            else:
                print(f"Не удалось открыть базу бесед {database_path(config)}: {db.error}")

        engine.session_id = self.session_id
        self._log = UnansweredLog(config, self.session_id)
        feedback_log = FeedbackLog(config, self.session_id)
        engine.feedback_listeners.append(feedback_log.write)
        if self.db is not None:
            engine.feedback_listeners.append(lambda entry: self.db.feedback(self.session_id, entry))

    def user(self, line: str) -> None:
        if line.strip().lower() in EXIT_WORDS:
            return
        if self.recorder is not None:
            self.recorder.user(line)
        if self.db is not None:
            self.db.user_message(self.session_id, line)

    def bot(self, line: str, reply: ChatReply) -> None:
        if self.recorder is not None:
            self.recorder.bot(reply)
        if self.db is not None:
            self.db.bot_message(self.session_id, reply)
        if not line.strip().isdigit():  # номер пункта меню — не вопрос
            self._log.write(line, reply.kind)
            if self.db is not None:
                self.db.unanswered(self.session_id, line, reply.kind)

    def close(self) -> None:
        if self.recorder is not None:
            self.recorder.close()
        if self.db is not None:
            self.db.finish_session(self.session_id)
        if self.recorder is not None and self.config.email.enabled:
            print(f"Отправляю журнал беседы на {', '.join(self.config.email.recipients())}...")
            _send_logs(self.config, db=self.db)
        if self.db is not None:
            self.db.close()


def chat(engine: ChatEngine) -> int:
    conversation = Conversation(engine)
    if engine.config.email.enabled and engine.config.email.send_pending_on_start:
        _send_logs(engine.config, skip=conversation.session_id, db=conversation.db)
    try:
        return _chat_loop(engine, engine.knowledge_base, conversation)
    finally:
        conversation.close()


def _chat_loop(engine: ChatEngine, kb, conversation: Conversation) -> int:
    print(f"CryptoChat {__version__}")
    parts = "; ".join(f"{r.name}: {len(kb.questions_under(r.id))}" for r in kb.roots)
    print(f"Загружено тем: {len(kb.roots)}, вопросов: {len(kb.questions)} ({parts})")
    note = f" ({engine.morphology_note})" if engine.morphology_note else ""
    print(f"Морфология: {engine.morphology}{note}")
    print(f"Учёт опечаток: {engine.fuzzy.name if engine.fuzzy else 'выключен'}")
    if engine.semantic is not None:
        print(f"Поиск по смыслу: {engine.semantic.name} ({engine.semantic.phrases} фраз)")
    else:
        print("Поиск по смыслу: выключен" + (" (недоступен)" if engine.semantic_note else ""))
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

        conversation.user(line)
        reply = engine.ask(line)
        conversation.bot(line, reply)
        print_reply(reply)

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
    print("  --compare-semantic сравнить поиск с моделью и без неё (контрольные запросы и перефразы)")
    print("  --download-model   скачать модель для поиска по смыслу в папку models")
    print("  --semantic-dump    выгрузить оценки модели в semantic_dump.json (для настройки порогов)")
    print("  --debug            показывать, как бот ищет ответ")
    print("  --test-email       отправить тестовое письмо по настройкам email из config.json")
    print("  --send-pending     отправить на почту журналы, которые не ушли раньше")
    print("  --stats            статистика по базе бесед: частые вопросы, отзывы, неотвеченные запросы")
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
        if options.compare_semantic:
            return run_compare_semantic(content_dir, options.morphology)
        if options.download_model:
            return run_download_model(content_dir)
        if options.semantic_dump:
            return run_semantic_dump(content_dir, options.morphology)
        if options.test_email or options.send_pending:
            return run_email_command(content_dir, options)
        if options.stats:
            return run_stats(content_dir)
        if AppConfig.load(content_dir).semantic.enabled:
            print("Загружаю модель для поиска по смыслу (первый запуск может занять несколько минут)...")
        engine = ChatEngine.create(content_dir, options.morphology, app_dir=app_directory())
        if engine.semantic_note:
            print("Поиск по смыслу недоступен: " + engine.semantic_note)
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
