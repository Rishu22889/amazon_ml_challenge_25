# Build & deploy with:  gcloud run deploy --source .   (run from repo root)
FROM python:3.11-slim

# lightgbm needs libgomp1 at runtime
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY backend/requirements.txt ./backend/requirements.txt

# CPU-only torch/torchvision — the default PyPI build pulls ~2GB of
# unused CUDA packages, which is the #1 cause of OOM kills on machines
# without a GPU (Cloud Run included).
RUN pip install --no-cache-dir torch==2.3.1 torchvision==0.18.1 \
      --index-url https://download.pytorch.org/whl/cpu \
    && pip install --no-cache-dir -r backend/requirements.txt \
    && python -c "import nltk; nltk.download('stopwords'); nltk.download('punkt'); nltk.download('punkt_tab')"

COPY backend/ ./backend/
COPY models/ ./models/

WORKDIR /app/backend

ENV PYTHONUNBUFFERED=1

# Cloud Run injects $PORT (defaults to 8080) — shell form so it expands.
CMD exec gunicorn -w 1 -b 0.0.0.0:${PORT:-8080} app:app --timeout 300