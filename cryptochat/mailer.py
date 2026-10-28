"""Отправка журнала беседы на почту (SMTP). Используется только стандартная библиотека Python.

Пароль ящика-отправителя НЕ хранится в config.json (он попадает в Git). Он берётся:
  1) из переменной окружения, имя которой задано в email.passwordEnv (по умолчанию CRYPTOCHAT_SMTP_PASSWORD);
  2) иначе из файла secrets.json рядом с программой: { "smtpPassword": "..." }.
Файл secrets.json исключён из Git (.gitignore).
"""
from __future__ import annotations

import json
import os
import smtplib
import ssl
from email.message import EmailMessage
from pathlib import Path
from typing import Callable, Optional

from .models import EmailConfig
from .session import count_messages, mark_sent, pending_sessions

SECRETS_FILE = "secrets.json"


class MailError(Exception):
    """Письмо не отправлено: понятное объяснение причины."""


def load_password(config: EmailConfig, app_dir: Path) -> str:
    from_env = os.environ.get(config.password_env or "", "")
    if from_env:
        return from_env
    path = app_dir / SECRETS_FILE
    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8-sig"))
        except (OSError, json.JSONDecodeError) as ex:
            raise MailError(f"Не удалось прочитать {path}: {ex}") from ex
        password = data.get("smtpPassword", "") if isinstance(data, dict) else ""
        if password:
            return str(password)
    raise MailError(
        f"Не задан пароль почты. Укажите его в файле {path} как {{\"smtpPassword\": \"...\"}} "
        f"или в переменной окружения {config.password_env}."
    )


def build_message(config: EmailConfig, subject: str, body: str,
                  attachment: Optional[Path] = None) -> EmailMessage:
    message = EmailMessage()
    message["From"] = config.from_address()
    message["To"] = ", ".join(config.recipients())
    message["Subject"] = subject
    message.set_content(body)
    if attachment is not None:
        message.add_attachment(attachment.read_bytes(), maintype="text", subtype="plain",
                               filename=attachment.name)
    return message


def send_message(config: EmailConfig, password: str, message: EmailMessage,
                 smtp_factory: Optional[Callable] = None) -> None:
    """Отправляет письмо. smtp_factory подменяется в тестах."""
    security = config.security.lower()
    timeout = config.timeout_seconds
    try:
        if smtp_factory is not None:
            server = smtp_factory(config.smtp_host, config.smtp_port, timeout)
        elif security == "ssl":
            server = smtplib.SMTP_SSL(config.smtp_host, config.smtp_port, timeout=timeout,
                                      context=ssl.create_default_context())
        else:
            server = smtplib.SMTP(config.smtp_host, config.smtp_port, timeout=timeout)
        with server:
            if security == "starttls" and smtp_factory is None:
                server.starttls(context=ssl.create_default_context())
            if config.username:
                server.login(config.username, password)
            server.send_message(message)
    except smtplib.SMTPAuthenticationError as ex:
        raise MailError("Почтовый сервер не принял логин или пароль. Для Яндекс, Mail.ru и Gmail "
                        "нужен специальный «пароль приложения», а не обычный пароль от ящика.") from ex
    except (smtplib.SMTPException, OSError) as ex:
        raise MailError(f"Не удалось отправить письмо через {config.smtp_host}:{config.smtp_port}: {ex}") from ex


def send_session(config: EmailConfig, password: str, path: Path,
                 smtp_factory: Optional[Callable] = None) -> bool:
    """Отправляет один журнал. Возвращает False, если журнал пустой и пустые не отправляются."""
    text = path.read_text(encoding="utf-8")
    messages = count_messages(text)
    if messages == 0 and not config.send_empty:
        mark_sent(path)  # пустую сессию больше не проверяем
        return False
    subject = f"{config.subject_prefix}: беседа {path.stem} (сообщений: {messages})"
    send_message(config, password, build_message(config, subject, text, path), smtp_factory)
    mark_sent(path)
    return True


def send_pending(config: EmailConfig, app_dir: Path, folder: Path, skip: Optional[str] = None,
                 log: Callable[[str], None] = print, smtp_factory: Optional[Callable] = None) -> int:
    """Отправляет все готовые неотправленные журналы. Возвращает число отправленных писем."""
    paths = pending_sessions(folder, skip=skip)
    if not paths:
        return 0
    password = load_password(config, app_dir)
    sent = 0
    for path in paths:
        if send_session(config, password, path, smtp_factory):
            sent += 1
            log(f"Журнал {path.stem} отправлен на {', '.join(config.recipients())}.")
    return sent


def send_test(config: EmailConfig, app_dir: Path, smtp_factory: Optional[Callable] = None) -> None:
    password = load_password(config, app_dir)
    message = build_message(config, f"{config.subject_prefix}: проверка почты",
                            "Это тестовое письмо CryptoChat. Если вы его получили, отправка журналов настроена.")
    send_message(config, password, message, smtp_factory)
