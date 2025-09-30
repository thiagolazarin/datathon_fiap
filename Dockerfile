# ---- base python ----
FROM python:3.11.9-slim

ENV PIP_NO_CACHE_DIR=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# deps nativos para psycopg2 / lightgbm
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc libpq-dev libgomp1 \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# libs
COPY requirements-api.txt /app/requirements.txt
RUN pip install -r requirements.txt

# código (tudo)
COPY . /app

# garante import de /app
ENV PYTHONPATH=/app

EXPOSE 8000

# sobe a API
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
