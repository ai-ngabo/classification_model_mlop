# Streamlit dashboard for the Corn Leaf Disease classifier.

FROM python:3.11-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

COPY ui/requirements.txt /app/ui/requirements.txt
RUN pip install --upgrade pip && pip install -r /app/ui/requirements.txt

COPY ui/ /app/ui/
# The UI reads sample images + dataset stats from data/train for the
# visualizations tab; copy it in if present (safe if empty).
COPY data/ /app/data/

# API location is injected at runtime (compose sets it to the api service;
# Cloud Run sets it to the deployed API URL).
ENV API_URL=http://localhost:8000 \
    PORT=8501

EXPOSE 8501

# Streamlit must bind 0.0.0.0 and the platform port; disable usage stats.
CMD exec streamlit run ui/app.py \
    --server.port=${PORT} \
    --server.address=0.0.0.0 \
    --server.headless=true \
    --browser.gatherUsageStats=false