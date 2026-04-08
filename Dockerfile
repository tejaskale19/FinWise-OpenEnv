# FinWise OpenEnv — Dockerfile
# Builds a containerized FastAPI server for HF Spaces deployment

FROM python:3.11-slim

# Metadata
LABEL name="finwise-openenv"
LABEL description="Indian Portfolio Advisory OpenEnv Environment"
LABEL version="1.0.0"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (layer caching)
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY app.py .
COPY inference.py .
COPY openenv.yaml .
COPY env.py .
COPY models.py .
COPY tasks.py .
COPY graders.py .
COPY finwise_env/ ./finwise_env/

# Expose port (HF Spaces uses 7860)
EXPOSE 7860

# Health check — validates /reset responds
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -sf -X POST http://localhost:7860/reset \
        -H "Content-Type: application/json" \
        -d '{}' || exit 1

# Environment defaults
ENV PORT=7860
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Run FastAPI server
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
