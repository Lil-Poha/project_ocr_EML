"""Форматы ответов API gateway."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])


class RecognitionResponse(BaseModel):
    text: str = Field(description="Распознанный текст")
    model: str = Field(description="Имя распознавателя, который дал ответ", examples=["stub"])
    stub: bool = Field(description="True, если ответ дала заглушка")
