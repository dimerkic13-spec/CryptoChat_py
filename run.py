"""Точка входа для сборки CryptoChat.exe (PyInstaller) и для запуска: python run.py"""
import sys

from cryptochat.cli import main

if __name__ == "__main__":
    sys.exit(main())
