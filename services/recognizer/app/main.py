"""HTTP API сервиса распознавания.

Эндпоинты:
    GET  /health     — проверка, что сервис жив (используется docker-compose)
    POST /recognize  — принимает изображение (multipart, поле "file"), возвращает текст
"""

import io
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, HTTPException, Request, UploadFile
from PIL import Image, UnidentifiedImageError

from .config import load_settings
from .recognizers import Recognizer, build_recognizer
from .schemas import HealthResponse, RecognitionResponse


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Распознаватель создаётся один раз при старте сервиса.
    # Для настоящей модели это важно: веса грузятся долго, делать это
    # на каждый запрос нельзя.
    settings = load_settings()
    app.state.recognizer = build_recognizer(settings.backend, settings.stub_text)
    yield


app = FastAPI(title="OCR Recognizer", version="0.1.0", lifespan=lifespan)


def get_recognizer(request: Request) -> Recognizer:
    return request.app.state.recognizer


@app.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    recognizer = get_recognizer(request)
    return HealthResponse(status="ok", model=recognizer.name, stub=recognizer.is_stub)


# Обычная функция (def, а не async def): FastAPI выполнит её в отдельном потоке.
# Инференс модели нагружает процессор, и так он не будет блокировать сервер.
@app.post("/recognize", response_model=RecognitionResponse)
def recognize(request: Request, file: UploadFile = File(...)) -> RecognitionResponse:
    data = file.file.read()
    if not data:
        raise HTTPException(status_code=400, detail="Файл пустой")

    try:
        image = Image.open(io.BytesIO(data))
        image.load()  # декодируем сразу, чтобы битый файл отловился здесь
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError):
        raise HTTPException(status_code=422, detail="Не удалось прочитать изображение")

    recognizer = get_recognizer(request)
    text = recognizer.predict(image)
    return RecognitionResponse(text=text, model=recognizer.name, stub=recognizer.is_stub)
