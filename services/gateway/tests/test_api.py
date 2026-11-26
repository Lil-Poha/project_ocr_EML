"""Тесты API gateway. Запуск из папки services/gateway: python -m pytest

Настоящий recognizer в тестах не нужен: вместо него подставляется
httpx.MockTransport, который отвечает заранее заданным образом.
"""

import httpx
import pytest
from fastapi.testclient import TestClient

from app.config import Settings
from app.main import app, get_recognizer_client, get_settings

STUB_RESPONSE = {"text": "Тестовый текст", "model": "stub", "stub": True}
PNG_FILE = {"file": ("line.png", b"\x89PNG fake bytes", "image/png")}


def make_client(handler, settings: Settings = Settings()) -> TestClient:
    """Собрать тестовый клиент, где recognizer отвечает функцией handler."""
    mock_client = httpx.AsyncClient(
        transport=httpx.MockTransport(handler), base_url="http://recognizer"
    )
    app.dependency_overrides[get_recognizer_client] = lambda: mock_client
    app.dependency_overrides[get_settings] = lambda: settings
    return TestClient(app)


@pytest.fixture(autouse=True)
def reset_overrides():
    yield
    app.dependency_overrides.clear()


def test_health():
    client = make_client(lambda request: httpx.Response(200))
    assert client.get("/health").json() == {"status": "ok"}


def test_recognize_forwards_file_to_recognizer():
    received = {}

    def handler(request: httpx.Request) -> httpx.Response:
        received["path"] = request.url.path
        received["body"] = request.content
        return httpx.Response(200, json=STUB_RESPONSE)

    response = make_client(handler).post("/api/recognize", files=PNG_FILE)

    assert response.status_code == 200
    assert response.json() == STUB_RESPONSE
    assert received["path"] == "/recognize"
    assert b"\x89PNG fake bytes" in received["body"]


def test_rejects_unsupported_type():
    client = make_client(lambda request: httpx.Response(200, json=STUB_RESPONSE))
    response = client.post("/api/recognize", files={"file": ("a.txt", b"hi", "text/plain")})
    assert response.status_code == 415


def test_rejects_too_large_file():
    client = make_client(
        lambda request: httpx.Response(200, json=STUB_RESPONSE),
        settings=Settings(max_upload_mb=0),  # лимит 0 МБ: любой файл слишком большой
    )
    response = client.post("/api/recognize", files=PNG_FILE)
    assert response.status_code == 413


def test_passes_through_bad_image_error():
    client = make_client(
        lambda request: httpx.Response(422, json={"detail": "Не удалось прочитать изображение"})
    )
    response = client.post("/api/recognize", files=PNG_FILE)
    assert response.status_code == 422
    assert response.json()["detail"] == "Не удалось прочитать изображение"


def test_recognizer_unavailable():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused")

    response = make_client(handler).post("/api/recognize", files=PNG_FILE)
    assert response.status_code == 503


def test_recognizer_timeout():
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("timeout")

    response = make_client(handler).post("/api/recognize", files=PNG_FILE)
    assert response.status_code == 504


def test_recognizer_internal_error():
    response = make_client(lambda request: httpx.Response(500)).post(
        "/api/recognize", files=PNG_FILE
    )
    assert response.status_code == 502


@pytest.mark.parametrize("body", [b"not json", b"{}", b"[]", b'{"text": null}'])
def test_recognizer_invalid_success_response(body):
    response = make_client(lambda request: httpx.Response(200, content=body)).post(
        "/api/recognize", files=PNG_FILE
    )
    assert response.status_code == 502
    assert response.json()["detail"] == "Сервис распознавания вернул некорректный ответ"
