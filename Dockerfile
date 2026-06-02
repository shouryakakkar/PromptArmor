# ── Stage 1: builder ──────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Stage 2: runtime ──────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

WORKDIR /app

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy app source
COPY proxy/       ./proxy/
COPY training/    ./training/
COPY data/        ./data/
COPY dashboard/   ./dashboard/

# Pre-download the sentence-transformer model so it's baked into the image
# (avoids cold-start delay on first request in production)
ENV USE_TF=0
ENV TRANSFORMERS_NO_ADVISORY_WARNINGS=1
ENV HF_HUB_DISABLE_SYMLINKS_WARNING=1

RUN python -c "\
import os; os.environ['USE_TF']='0'; \
from sentence_transformers import SentenceTransformer; \
SentenceTransformer('all-MiniLM-L6-v2'); \
print('Model cached.')"

# Runtime environment defaults (override with platform env vars)
ENV LOG_LEVEL=INFO
ENV BLOCK_THRESHOLD=0.75
ENV FLAG_THRESHOLD=0.5
ENV DATABASE_PATH=/data/proxy.db

# Data volume for SQLite persistence
VOLUME ["/data"]

EXPOSE 8000

# Use a single worker in the free tier; increase for paid plans
CMD ["uvicorn", "proxy.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
