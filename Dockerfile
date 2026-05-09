FROM python:3.11-slim AS base

# OS deps for opencv-python-headless (libGL not strictly needed for headless,
# but ffmpeg libs are pulled in for video decoding)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install deps first for layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code (model weights mounted as volume in compose)
COPY app/ ./app/
COPY training/configs/ ./training/configs/

# Non-root user
RUN useradd -u 1000 -m appuser && \
    mkdir -p /app/data && chown -R appuser:appuser /app
USER appuser

ENV LABOR_INTEL_LOG_LEVEL=INFO \
    PYTHONUNBUFFERED=1

EXPOSE 8000

# Single uvicorn worker — the inference engine holds GPU state and ByteTrack
# state; multi-worker would break track continuity.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
