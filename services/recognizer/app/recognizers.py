"""Распознаватели текста.

Сервис работает с любым распознавателем через общий интерфейс `Recognizer`.
Сейчас есть только заглушка `StubRecognizer`. Когда будет готова обученная
модель (своя CRNN, CRNN с готовым backbone или TrOCR), достаточно добавить
новый класс с методом `predict` и зарегистрировать его в `build_recognizer`.
Остальной код сервиса менять не придётся.
"""

from abc import ABC, abstractmethod

from PIL import Image

# Фиксированный ответ заглушки. Похож на текст из исторического документа,
# чтобы на демонстрации было понятно, как будет выглядеть результат.
DEFAULT_STUB_TEXT = (
    "Милостивый государь, честь имею уведомить Вас, "
    "что прошение Ваше получено и принято к рассмотрению."
)


class Recognizer(ABC):
    """Общий интерфейс распознавателя."""

    #: Короткое имя модели, возвращается клиенту в ответе.
    name: str
    #: True, если это заглушка, а не настоящая модель.
    is_stub: bool = False

    @abstractmethod
    def predict(self, image: Image.Image) -> str:
        """Распознать текст на изображении."""


class StubRecognizer(Recognizer):
    """Заглушка: игнорирует картинку и всегда возвращает один и тот же текст."""

    name = "stub"
    is_stub = True

    def __init__(self, text: str = DEFAULT_STUB_TEXT) -> None:
        self._text = text

    def predict(self, image: Image.Image) -> str:
        return self._text


def build_recognizer(backend: str, stub_text: str | None = None) -> Recognizer:
    """Создать распознаватель по имени из настроек (переменная RECOGNIZER_BACKEND)."""
    if backend == "stub":
        return StubRecognizer(stub_text or DEFAULT_STUB_TEXT)
    # Сюда позже добавится, например: if backend == "crnn": return CRNNRecognizer(...)
    raise ValueError(f"Неизвестный RECOGNIZER_BACKEND: {backend!r}")
