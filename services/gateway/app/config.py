"""Настройки gateway. Читаются из переменных окружения (их задаёт docker-compose)."""

import os
from dataclasses import dataclass

# Форматы, которые принимаем от пользователя. Сканы архивов часто бывают в TIFF.
ALLOWED_CONTENT_TYPES = frozenset(
    {"image/jpeg", "image/png", "image/tiff", "image/bmp", "image/webp"}
)


@dataclass(frozen=True)
class Settings:
    # Адрес recognizer внутри docker-сети: имя сервиса из docker-compose.yml
    recognizer_url: str = "http://recognizer:8000"
    # Максимальный размер загружаемого файла, в мегабайтах
    max_upload_mb: int = 10
    # Сколько секунд ждать ответа от recognizer
    recognizer_timeout: float = 30.0

    @property
    def max_upload_bytes(self) -> int:
        return self.max_upload_mb * 1024 * 1024


def load_settings() -> Settings:
    return Settings(
        recognizer_url=os.getenv("RECOGNIZER_URL", Settings.recognizer_url).rstrip("/"),
        max_upload_mb=int(os.getenv("MAX_UPLOAD_MB", Settings.max_upload_mb)),
        recognizer_timeout=float(os.getenv("RECOGNIZER_TIMEOUT", Settings.recognizer_timeout)),
    )
