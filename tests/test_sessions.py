"""Номер сессии, журнал беседы и отправка на почту (без настоящего почтового сервера в интернете)."""
import os
import socket
import tempfile
import threading
import time
import unittest
from datetime import datetime, timedelta
from pathlib import Path

from cryptochat.mailer import MailError, load_password, send_message, send_pending, send_session, build_message
from cryptochat.models import AppConfig, ChatReply, EmailConfig, ReplyKind
from cryptochat.session import (END_MARKER, SessionRecorder, is_sent, new_session_id, pending_sessions)


def email_config(**overrides):
    config = EmailConfig(enabled=True, smtp_host="localhost", smtp_port=25, security="none",
                         username="", sender="bot@example.ru", to=["boss@example.ru"])
    for key, value in overrides.items():
        setattr(config, key, value)
    return config


class FakeSMTP:
    """Подмена почтового сервера: запоминает отправленные письма."""
    sent = []

    def __init__(self, host, port, timeout):
        self.logged_in = None

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def login(self, user, password):
        self.logged_in = (user, password)

    def send_message(self, message):
        FakeSMTP.sent.append(message)


class SessionTests(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.folder = Path(self._tmp.name) / "sessions"
        FakeSMTP.sent = []

    def tearDown(self):
        self._tmp.cleanup()

    def recorder(self, **kw):
        return SessionRecorder(self.folder, "0.1a", "стеммер", **kw)

    def test_session_id_format(self):
        sid = new_session_id(datetime(2026, 10, 28, 8, 40, 12))
        self.assertRegex(sid, r"^20261028-084012-[0-9a-f]{4}$")

    def test_ids_are_unique(self):
        self.assertEqual(50, len({new_session_id() for _ in range(50)}))

    def test_transcript_is_written_immediately_and_closed(self):
        rec = self.recorder()
        rec.user("Забыл пароль")
        rec.bot(ChatReply(ReplyKind.CLARIFY, "Уточните:\n  1. Вариант"))
        text = rec.path.read_text(encoding="utf-8")
        self.assertIn(rec.session_id, text)
        self.assertIn("Пользователь: Забыл пароль", text)
        self.assertIn("Бот (Clarify): Уточните:\n      1. Вариант", text)
        self.assertNotIn(END_MARKER, text)
        rec.close(); rec.close()
        text = rec.path.read_text(encoding="utf-8")
        self.assertEqual(1, text.count(END_MARKER))
        self.assertIn("сообщений: 1", text)

    def test_pending_takes_finished_and_stale_but_not_running(self):
        done = self.recorder(); done.user("вопрос"); done.close()
        running = self.recorder(); running.user("ещё идёт")
        abandoned = self.recorder(); abandoned.user("окно закрыли крестиком")
        old = time.time() - 13 * 3600
        os.utime(abandoned.path, (old, old))

        names = [p.stem for p in pending_sessions(self.folder)]
        self.assertIn(done.session_id, names)
        self.assertIn(abandoned.session_id, names)
        self.assertNotIn(running.session_id, names)
        self.assertNotIn(done.session_id, [p.stem for p in pending_sessions(self.folder, skip=done.session_id)])

    def test_send_session_marks_sent(self):
        rec = self.recorder(); rec.user("как купить лицензию"); rec.close()
        self.assertTrue(send_session(email_config(), "", rec.path, FakeSMTP))
        self.assertTrue(is_sent(rec.path))
        message = FakeSMTP.sent[0]
        self.assertIn(rec.session_id, message["Subject"])
        self.assertEqual("boss@example.ru", message["To"])
        self.assertEqual("bot@example.ru", message["From"])
        attachments = list(message.iter_attachments())
        self.assertEqual(f"{rec.session_id}.txt", attachments[0].get_filename())
        self.assertEqual([], pending_sessions(self.folder))

    def test_empty_session_is_not_sent(self):
        rec = self.recorder(); rec.close()
        self.assertFalse(send_session(email_config(), "", rec.path, FakeSMTP))
        self.assertEqual([], FakeSMTP.sent)
        self.assertTrue(is_sent(rec.path))  # больше не проверяется

    def test_send_pending_sends_all_ready(self):
        for i in range(3):
            rec = self.recorder(); rec.user(f"вопрос {i}"); rec.close()
        with tempfile.TemporaryDirectory() as app:
            Path(app, "secrets.json").write_text('{"smtpPassword": "pw"}', encoding="utf-8")
            self.assertEqual(3, send_pending(email_config(), Path(app), self.folder, log=lambda m: None,
                                             smtp_factory=FakeSMTP))
        self.assertEqual(3, len(FakeSMTP.sent))

    def test_login_uses_username_and_password(self):
        captured = {}

        class Capturing(FakeSMTP):
            def login(self, user, password):
                captured["login"] = (user, password)

        config = email_config(username="bot@example.ru")
        send_message(config, "secret", build_message(config, "s", "b"), Capturing)
        self.assertEqual(("bot@example.ru", "secret"), captured["login"])


class PasswordTests(unittest.TestCase):
    def test_env_var_wins(self):
        os.environ["CRYPTOCHAT_TEST_PW"] = "from-env"
        try:
            with tempfile.TemporaryDirectory() as app:
                self.assertEqual("from-env", load_password(email_config(password_env="CRYPTOCHAT_TEST_PW"), Path(app)))
        finally:
            del os.environ["CRYPTOCHAT_TEST_PW"]

    def test_secrets_file(self):
        with tempfile.TemporaryDirectory() as app:
            Path(app, "secrets.json").write_text('{"smtpPassword": "from-file"}', encoding="utf-8")
            self.assertEqual("from-file", load_password(email_config(password_env="NO_SUCH_VAR_X"), Path(app)))

    def test_missing_password_explains(self):
        with tempfile.TemporaryDirectory() as app:
            with self.assertRaises(MailError) as ctx:
                load_password(email_config(password_env="NO_SUCH_VAR_X"), Path(app))
        self.assertIn("secrets.json", str(ctx.exception))


class EmailConfigTests(unittest.TestCase):
    def test_from_alias_and_string_recipient(self):
        import json
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({"email": {
                "enabled": True, "smtpHost": "smtp.yandex.ru", "from": "bot@ya.ru", "to": "a@x.ru, b@x.ru"}}),
                encoding="utf-8")
            config = AppConfig.load(tmp)
        self.assertEqual("bot@ya.ru", config.email.from_address())
        self.assertEqual(["a@x.ru", "b@x.ru"], config.email.recipients())

    def test_enabled_without_recipient_is_error(self):
        import json
        from cryptochat.models import ContentError
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "config.json").write_text(json.dumps({"email": {
                "enabled": True, "smtpHost": "smtp.yandex.ru", "username": "bot@ya.ru"}}), encoding="utf-8")
            with self.assertRaises(ContentError):
                AppConfig.load(tmp)


class _MiniSMTPServer(threading.Thread):
    """Крошечный SMTP-сервер на localhost: проверяет, что настоящий smtplib доставляет письмо."""

    def __init__(self):
        super().__init__(daemon=True)
        self.sock = socket.socket()
        self.sock.bind(("127.0.0.1", 0))
        self.sock.listen(1)
        self.port = self.sock.getsockname()[1]
        self.data = b""

    def run(self):
        conn, _ = self.sock.accept()
        f = conn.makefile("rb")
        send = lambda line: conn.sendall(line.encode() + b"\r\n")
        send("220 test")
        in_data = False
        for raw in f:
            line = raw.decode("utf-8", "replace").rstrip("\r\n")
            if in_data:
                if line == ".":
                    in_data = False
                    send("250 OK")
                else:
                    self.data += raw
                continue
            cmd = line.split(" ", 1)[0].upper()
            if cmd in ("EHLO", "HELO"):
                send("250-test"); send("250 8BITMIME")
            elif cmd == "DATA":
                in_data = True; send("354 go")
            elif cmd == "QUIT":
                send("221 bye"); break
            else:
                send("250 OK")
        conn.close(); self.sock.close()


class RealSmtplibTests(unittest.TestCase):
    def test_delivers_to_local_server(self):
        server = _MiniSMTPServer(); server.start()
        config = email_config(smtp_port=server.port, security="none")
        send_message(config, "", build_message(config, "CryptoChat: проверка", "Тело письма"))
        server.join(5)
        self.assertIn(b"Subject:", server.data)
        self.assertIn(b"boss@example.ru", server.data)

    def test_unreachable_server_gives_mail_error(self):
        config = email_config(smtp_port=1, security="none", timeout_seconds=2)
        with self.assertRaises(MailError):
            send_message(config, "", build_message(config, "s", "b"))


if __name__ == "__main__":
    unittest.main()
