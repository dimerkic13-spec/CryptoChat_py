"""База бесед SQLite (встроена в Python, ничего устанавливать не нужно).

Файл logs/cryptochat.db хранит:
  sessions    — беседы: номер, время начала и конца, компьютер, версия, морфология, отправлена ли на почту;
  messages    — все реплики пользователя и бота (тип ответа, id вопроса из базы, оценка);
  feedback    — отзывы «да» / «нет» об ответах;
  unanswered  — запросы, на которые бот не ответил сразу (уточнял тему, не понял или отправил в поддержку).

По этим данным команда  python run.py --stats  показывает, какие вопросы спрашивают чаще,
какие ответы не помогают и чего не хватает в базе знаний.

База бесед не заменяет базу знаний: вопросы и ответы остаются в content/*.json — их удобно
править и хранить в Git. Ошибки записи в базу не останавливают диалог.
"""
from __future__ import annotations

import sqlite3
from contextlib import closing
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ChatReply, FeedbackEntry, ReplyKind

SCHEMA_VERSION = 1

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id          TEXT PRIMARY KEY,
    started     TEXT NOT NULL,
    finished    TEXT,
    computer    TEXT,
    version     TEXT,
    morphology  TEXT,
    messages    INTEGER NOT NULL DEFAULT 0,
    emailed     TEXT
);
CREATE TABLE IF NOT EXISTS messages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(id),
    time        TEXT NOT NULL,
    role        TEXT NOT NULL CHECK (role IN ('user', 'bot')),
    kind        TEXT,
    text        TEXT NOT NULL,
    question_id TEXT,
    score       REAL
);
CREATE TABLE IF NOT EXISTS feedback (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    time        TEXT NOT NULL,
    query       TEXT,
    question_id TEXT NOT NULL,
    helpful     INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS unanswered (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT,
    time        TEXT NOT NULL,
    query       TEXT NOT NULL,
    kind        TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_messages_session ON messages(session_id);
CREATE INDEX IF NOT EXISTS ix_messages_question ON messages(question_id);
CREATE INDEX IF NOT EXISTS ix_feedback_question ON feedback(question_id);
"""

UNANSWERED_KINDS = {ReplyKind.ESCALATE, ReplyKind.NOT_UNDERSTOOD, ReplyKind.CLARIFY_BRANCH}


def _now() -> str:
    return datetime.now().isoformat(sep=" ", timespec="seconds")


class ChatDatabase:
    """Запись бесед в SQLite. Если база недоступна, методы молча ничего не делают (available = False)."""

    def __init__(self, path: Path):
        self.path = path
        self.available = True
        self.error: Optional[str] = None
        self._conn = None
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(path), timeout=5, isolation_level=None)
            self._conn.execute("PRAGMA foreign_keys = ON")
            self._conn.executescript(SCHEMA)
            self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
        except (sqlite3.Error, OSError) as ex:
            self.available = False
            self.error = str(ex)
            if getattr(self, "_conn", None) is not None:
                self._conn.close()  # не оставлять файл базы занятым
            self._conn = None

    # ---------- запись ----------

    def _run(self, sql: str, params: tuple = ()) -> None:
        if not self.available:
            return
        try:
            self._conn.execute(sql, params)
        except sqlite3.Error as ex:
            self.available = False  # не мешаем диалогу: дальше беседа просто не пишется в базу
            self.error = str(ex)

    def start_session(self, session_id: str, computer: str, version: str, morphology: str) -> None:
        self._run("INSERT OR IGNORE INTO sessions (id, started, computer, version, morphology) VALUES (?, ?, ?, ?, ?)",
                  (session_id, _now(), computer, version, morphology))

    def user_message(self, session_id: str, text: str) -> None:
        self._run("INSERT INTO messages (session_id, time, role, text) VALUES (?, ?, 'user', ?)",
                  (session_id, _now(), text))
        self._run("UPDATE sessions SET messages = messages + 1 WHERE id = ?", (session_id,))

    def bot_message(self, session_id: str, reply: ChatReply) -> None:
        question_id = reply.question.id if reply.question is not None else None
        score = reply.score if reply.question is not None else None
        self._run("INSERT INTO messages (session_id, time, role, kind, text, question_id, score) "
                  "VALUES (?, ?, 'bot', ?, ?, ?, ?)",
                  (session_id, _now(), reply.kind.value, reply.text, question_id, score))

    def unanswered(self, session_id: Optional[str], query: str, kind: ReplyKind) -> None:
        if kind in UNANSWERED_KINDS:
            self._run("INSERT INTO unanswered (session_id, time, query, kind) VALUES (?, ?, ?, ?)",
                      (session_id, _now(), query, kind.value))

    def feedback(self, session_id: Optional[str], entry: FeedbackEntry) -> None:
        self._run("INSERT INTO feedback (session_id, time, query, question_id, helpful) VALUES (?, ?, ?, ?, ?)",
                  (session_id, entry.time.isoformat(sep=" ", timespec="seconds"), entry.query,
                   entry.question_id, 1 if entry.helpful else 0))

    def finish_session(self, session_id: str) -> None:
        self._run("UPDATE sessions SET finished = ? WHERE id = ?", (_now(), session_id))

    def mark_emailed(self, session_id: str) -> None:
        self._run("UPDATE sessions SET emailed = ? WHERE id = ?", (_now(), session_id))

    def close(self) -> None:
        if self._conn is not None:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
            self._conn = None
            self.available = False


# ---------- Статистика ----------

def _rows(conn: sqlite3.Connection, sql: str, params: tuple = ()) -> list[tuple]:
    return conn.execute(sql, params).fetchall()


def _desc(text: str) -> str:
    """Ключ сортировки «сначала новые» для строк времени ГГГГ-ММ-ДД ЧЧ:ММ:СС."""
    return "".join(chr(0x10FFFF - ord(ch)) for ch in text)


def stats_report(path: Path, titles: Optional[dict[str, str]] = None, limit: int = 10) -> str:
    """Текстовый отчёт по базе бесед для команды --stats."""
    if not path.exists():
        return f"База бесед ещё не создана: {path}\nОна появится после первой беседы (config.json → database.enabled)."
    titles = titles or {}

    def title(question_id: str) -> str:
        text = titles.get(question_id, "")
        return (text[:60] + "…") if len(text) > 60 else text

    with closing(sqlite3.connect(f"file:{path}?mode=ro", uri=True, timeout=5)) as conn:
        sessions, first, last = _rows(conn, "SELECT COUNT(*), MIN(started), MAX(started) FROM sessions")[0]
        users = _rows(conn, "SELECT COUNT(*) FROM messages WHERE role = 'user'")[0][0]
        answers = _rows(conn, "SELECT COUNT(*) FROM messages WHERE role = 'bot' AND kind = 'Answer'")[0][0]
        yes, no = _rows(conn, "SELECT COALESCE(SUM(helpful), 0), COALESCE(SUM(1 - helpful), 0) FROM feedback")[0]
        unanswered_total = _rows(conn, "SELECT COUNT(*) FROM unanswered")[0][0]
        emailed = _rows(conn, "SELECT COUNT(*) FROM sessions WHERE emailed IS NOT NULL")[0][0]

        lines = [f"База бесед: {path}"]
        if not sessions:
            lines.append("Бесед пока нет.")
            return "\n".join(lines)
        lines.append(f"Бесед: {sessions} (с {first[:10]} по {last[:10]}), отправлено на почту: {emailed}")
        lines.append(f"Сообщений пользователей: {users}, выдано ответов: {answers}, "
                     f"не ответил сразу: {unanswered_total}")
        lines.append(f"Отзывы: помогло {yes}, не помогло {no}")

        top = _rows(conn, """
            SELECT m.question_id, COUNT(*) AS shown,
                   (SELECT COALESCE(SUM(helpful), 0) FROM feedback f WHERE f.question_id = m.question_id),
                   (SELECT COALESCE(SUM(1 - helpful), 0) FROM feedback f WHERE f.question_id = m.question_id)
            FROM messages m WHERE m.role = 'bot' AND m.kind = 'Answer' AND m.question_id IS NOT NULL
            GROUP BY m.question_id ORDER BY shown DESC, m.question_id LIMIT ?""", (limit,))
        if top:
            lines.append(f"\nЧаще всего выдавались ответы (топ {limit}):")
            for question_id, shown, good, bad in top:
                lines.append(f"  {question_id:10} выдан: {shown:<4} помогло: {good:<3} не помогло: {bad:<3} {title(question_id)}")

        bad = _rows(conn, """
            SELECT question_id, COUNT(*) FROM feedback WHERE helpful = 0
            GROUP BY question_id ORDER BY COUNT(*) DESC, question_id LIMIT ?""", (limit,))
        if bad:
            lines.append("\nОтветы с отзывом «не помогло» — стоит проверить текст и ключевые слова:")
            for question_id, count in bad:
                queries = _rows(conn, "SELECT query FROM feedback WHERE helpful = 0 AND question_id = ? "
                                      "ORDER BY time DESC LIMIT 3", (question_id,))
                examples = "; ".join(f"«{q[0]}»" for q in queries if q[0])
                lines.append(f"  {question_id:10} не помогло: {count:<4} {title(question_id)}")
                if examples:
                    lines.append(f"             запросы: {examples}")

        # Запросы складываются без учёта регистра. Делается в Python: LOWER() в SQLite
        # переводит в нижний регистр только латиницу.
        grouped: dict[str, list] = {}
        for query, time in _rows(conn, "SELECT query, time FROM unanswered"):
            key = " ".join(query.lower().replace("ё", "е").split())
            item = grouped.setdefault(key, [0, ""])
            item[0] += 1
            item[1] = max(item[1], time)
        # Сначала частые, при равенстве — более свежие.
        missing = sorted(((q, n, t) for q, (n, t) in grouped.items()), key=lambda x: (-x[1], _desc(x[2])))[:limit]
        if missing:
            lines.append("\nЗапросы, на которые бот не ответил сразу — кандидаты в новые вопросы или ключи:")
            for query, count, last_time in missing:
                lines.append(f"  повторов: {count:<4} «{query}»  (последний: {last_time[:16]})")

    return "\n".join(lines)
