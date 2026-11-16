"""HTTP API gateway.

Эндпоинты:
    GET  /health         — проверка, что gateway жив (используется docker-compose)
    POST /api/recognize  — принимает фото (multipart, поле "file"), проверяет его
                           и передаёт в recognizer, возвращает распознанный текст

Сейчас gateway просто пересылает фото. Позже здесь появится оркестрация:
сначала preprocessing (нарезка листа на строки), потом распознавание каждой строки.
"""

from contextlib import asynccontextmanager

import httpx
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile

from .config import ALLOWED_CONTENT_TYPES, Settings, load_settings
from .schemas import HealthResponse, RecognitionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = load_settings()
    app.state.settings = settings
    # Один HTTP-клиент на всё время работы: он переиспользует соединения
    async with httpx.AsyncClient(
        base_url=settings.recognizer_url, timeout=settings.recognizer_timeout
    ) as client:
        app.state.recognizer_client = client
        yield


app = FastAPI(title="OCR Gateway", version="0.1.0", lifespan=lifespan)


# Зависимости вынесены в функции, чтобы в тестах их можно было подменить
def get_settings(request: Request) -> Settings:
    return request.app.state.settings


def get_recognizer_client(request: Request) -> httpx.AsyncClient:
    return request.app.state.recognizer_client


@app.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    return HealthResponse(status="ok")


@app.post("/api/recognize", response_model=RecognitionResponse)
async def recognize(
    file: UploadFile = File(...),
    settings: Settings = Depends(get_settings),
    client: httpx.AsyncClient = Depends(get_recognizer_client),
) -> RecognitionResponse:
    if file.content_type not in ALLOWED_CONTENT_TYPES:
        raise HTTPException(
            status_code=415,
            detail="Неподдерживаемый формат файла. Подойдут JPEG, PNG, TIFF, BMP или WebP.",
        )

    # Читаем на один байт больше лимита: так узнаём, что файл слишком большой,
    # не загружая его целиком в память
    data = await file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"Файл слишком большой. Максимум {settings.max_upload_mb} МБ.",
        )
    if not data:
        raise HTTPException(status_code=400, detail="Файл пустой")

    try:
        response = await client.post(
            "/recognize", files={"file": (file.filename or "image", data, file.content_type)}
        )
    except httpx.TimeoutException:
        raise HTTPException(status_code=504, detail="Сервис распознавания не ответил вовремя")
    except httpx.RequestError:
        raise HTTPException(status_code=503, detail="Сервис распознавания недоступен")

    # Ошибки в данных (например, битая картинка) передаём пользователю как есть
    if response.status_code in (400, 422):
        try:
            detail = response.json()["detail"]
        except (ValueError, KeyError, TypeError):
            detail = "Не удалось обработать изображение"
        raise HTTPException(status_code=422, detail=detail)
    if response.status_code != 200:
        raise HTTPException(status_code=502, detail="Сервис распознавания вернул ошибку")

    try:
        return RecognitionResponse.model_validate(response.json())
    except ValueError:
        raise HTTPException(status_code=502, detail="Сервис распознавания вернул некорректный ответ")
