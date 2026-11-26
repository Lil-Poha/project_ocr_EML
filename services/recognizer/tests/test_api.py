"""Тесты API recognizer. Запуск из папки services/recognizer: python -m pytest"""

import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app
from app.recognizers import DEFAULT_STUB_TEXT


@pytest.fixture
def client():
    # with запускает lifespan, то есть создаёт распознаватель, как при старте сервиса
    with TestClient(app) as c:
        yield c


def make_png() -> bytes:
    buf = io.BytesIO()
    Image.new("L", (200, 64), color=255).save(buf, format="PNG")
    return buf.getvalue()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok", "model": "stub", "stub": True}


def test_recognize_returns_stub_text(client):
    response = client.post("/recognize", files={"file": ("line.png", make_png(), "image/png")})
    assert response.status_code == 200
    assert response.json() == {"text": DEFAULT_STUB_TEXT, "model": "stub", "stub": True}


def test_recognize_rejects_non_image(client):
    response = client.post("/recognize", files={"file": ("note.txt", b"hello", "text/plain")})
    assert response.status_code == 422


def test_recognize_rejects_empty_file(client):
    response = client.post("/recognize", files={"file": ("empty.png", b"", "image/png")})
    assert response.status_code == 400


def test_recognize_requires_file(client):
    response = client.post("/recognize")
    assert response.status_code == 422


def test_recognize_rejects_excessive_pixel_count(client, monkeypatch):
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", 100)
    response = client.post("/recognize", files={"file": ("line.png", make_png(), "image/png")})
    assert response.status_code == 422
