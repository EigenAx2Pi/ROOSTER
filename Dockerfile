# syntax=docker/dockerfile:1.7

FROM python:3.12-slim AS base
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

FROM base AS builder
WORKDIR /build
COPY pyproject.toml requirements.txt ./
RUN pip install --prefix=/install -r requirements.txt scikit-learn>=1.4.0

FROM base AS runtime
RUN useradd --create-home --shell /bin/bash rooster
WORKDIR /app
COPY --from=builder /install /usr/local
COPY app/ ./app/
COPY scripts/ ./scripts/
COPY static/ ./static/
COPY pyproject.toml ./
RUN mkdir -p uploads outputs && chown -R rooster:rooster /app
USER rooster
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/').read()" || exit 1
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
