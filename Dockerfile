# Dockerfile — buat deploy backend NeuroCheck ke Hugging Face Spaces (SDK: docker).
#
# HF Spaces (Docker) SELALU expose port 7860 secara default -- jangan diubah
# kecuali kamu juga ubah "app_port" di README_HF_SPACE.md.

FROM python:3.11-slim

# opencv butuh library sistem ini biar gak error "libGL.so.1 not found"
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ ./src/

# HF Spaces jalanin container sebagai user non-root secara default di beberapa
# runtime -- pastikan folder tempat SQLite lokal & cache HF ditulis bisa diakses.
RUN mkdir -p /app/data /app/outputs/checkpoints /app/outputs/reports /app/outputs/figures \
    && chmod -R 777 /app/data /app/outputs

ENV HF_HOME=/app/.cache/huggingface
RUN mkdir -p /app/.cache && chmod -R 777 /app/.cache

WORKDIR /app/src

EXPOSE 7860

# Render kasih tahu port lewat env var $PORT (beda-beda tiap kali deploy);
# HF Spaces & lokal pakai 7860 sebagai default kalau $PORT gak diset.
CMD uvicorn app:app --host 0.0.0.0 --port ${PORT:-7860}
