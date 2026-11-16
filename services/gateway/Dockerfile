FROM python:3.11-slim

# Не создавать .pyc и сразу выводить логи (без буферизации)
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /srv

# Сначала только зависимости: этот слой кэшируется и не пересобирается,
# пока не изменится requirements.txt
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY app ./app

# Запускаем не от root — так безопаснее
RUN useradd --create-home --uid 1000 appuser
USER appuser

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
