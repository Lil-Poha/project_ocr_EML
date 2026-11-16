"""Форматы ответов API (описаны через Pydantic, попадают в документацию /docs)."""

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(examples=["ok"])
    model: str = Field(description="Имя загруженного распознавателя", examples=["stub"])
    stub: bool = Field(description="True, если вместо модели работает заглушка")


class RecognitionResponse(BaseModel):
    text: str = Field(description="Распознанный текст")
    model: str = Field(description="Имя распознавателя, который дал ответ", examples=["stub"])
    stub: bool = Field(description="True, если ответ дала заглушка")
