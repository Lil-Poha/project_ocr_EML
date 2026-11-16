"""Настройки сервиса. Читаются из переменных окружения (их задаёт docker-compose)."""

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    # Какой распознаватель использовать: пока доступен только "stub".
    backend: str = "stub"
    # Необязательный текст для заглушки; если не задан, берётся текст по умолчанию.
    stub_text: str | None = None


def load_settings() -> Settings:
    return Settings(
        backend=os.getenv("RECOGNIZER_BACKEND", "stub"),
        stub_text=os.getenv("STUB_TEXT") or None,
    )
