"""Журнал беседы: у каждой беседы свой номер сессии и свой файл в папке logs/sessions/.

Реплики записываются в файл сразу, поэтому журнал не теряется, даже если окно закрыли крестиком
или компьютер выключился. В конце нормальной беседы в файл добавляется строка END_MARKER —
по ней видно, что сессия завершена и её можно отправлять на почту.
"""
from __future__ import annotations

import secrets
import socket
from datetime import datetime
from pathlib import Path
from typing import Optional

from .models import ChatReply

END_MARKER = "=== Конец сессии"
SENT_SUFFIX = ".sent"


def new_session_id(now: Optional[datetime] = None) -> str:
    """Номер сессии: дата, время и 4 случайных символа, например 20261028-084012-7f3a."""
    now = now or datetime.now()
    return f"{now:%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


class SessionRecorder:
    """Пишет беседу в файл <папка>/<номер сессии>.txt."""

    def __init__(self, folder: Path, version: str, morphology: str,
                 session_id: Optional[str] = None, now: Optional[datetime] = None):
        self.session_id = session_id or new_session_id(now)
        self.started = now or datetime.now()
        self.path = folder / f"{self.session_id}.txt"
        self.messages = 0  # сколько сообщений написал пользователь
        self._closed = False
        self._ok = True
        try:
            folder.mkdir(parents=True, exist_ok=True)
        except OSError:
            self._ok = False
        self._write(
            f"CryptoChat {version} — журнал беседы\n"
            f"Сессия: {self.session_id}\n"
            f"Начало: {self.started:%Y-%m-%d %H:%M:%S}\n"
            f"Компьютер: {_hostname()}\n"
            f"Морфология: {morphology}\n"
            f"{'-' * 60}\n"
        )

    @property
    def available(self) -> bool:
        """False, если файл журнала создать не удалось (например, нет прав на папку)."""
        return self._ok

    def user(self, text: str) -> None:
        self.messages += 1
        self._write(f"\n[{datetime.now():%H:%M:%S}] Пользователь: {text}\n")

    def bot(self, reply: ChatReply) -> None:
        lines = reply.text.split("\n")
        body = lines[0] + "".join("\n    " + line for line in lines[1:])
        self._write(f"[{datetime.now():%H:%M:%S}] Бот ({reply.kind.value}): {body}\n")

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        finished = datetime.now()
        self._write(f"\n{END_MARKER}: {finished:%Y-%m-%d %H:%M:%S}, сообщений: {self.messages}\n")

    def _write(self, text: str) -> None:
        if not self._ok:
            return
        try:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(text)
        except OSError:
            self._ok = False  # журнал не должен ломать диалог


def hostname() -> str:
    return _hostname()


def _hostname() -> str:
    try:
        return socket.gethostname()
    except OSError:
        return "неизвестно"


# ---------- Поиск сессий для отправки ----------

def sent_marker(path: Path) -> Path:
    return path.with_suffix(path.suffix + SENT_SUFFIX)


def is_sent(path: Path) -> bool:
    return sent_marker(path).exists()


def mark_sent(path: Path) -> None:
    try:
        sent_marker(path).write_text(f"{datetime.now():%Y-%m-%d %H:%M:%S}\n", encoding="utf-8")
    except OSError:
        pass


def count_messages(text: str) -> int:
    return text.count("] Пользователь: ")


def pending_sessions(folder: Path, stale_hours: float = 12.0, skip: Optional[str] = None,
                     now: Optional[datetime] = None) -> list[Path]:
    """Неотправленные журналы, которые пора отправить.

    Берутся завершённые сессии (есть END_MARKER) и «брошенные» — без маркера, но не менявшиеся
    дольше stale_hours часов (окно закрыли крестиком или программа упала). Сессии, которые,
    возможно, ещё идут в другом окне, пропускаются.
    """
    if not folder.is_dir():
        return []
    now = now or datetime.now()
    result = []
    for path in sorted(folder.glob("*.txt")):
        if path.stem == skip or is_sent(path):
            continue
        try:
            text = path.read_text(encoding="utf-8")
            age_hours = (now.timestamp() - path.stat().st_mtime) / 3600
        except OSError:
            continue
        if END_MARKER in text or age_hours >= stale_hours:
            result.append(path)
    return result
