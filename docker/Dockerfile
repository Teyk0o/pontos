# Stage 1: Base image with PyTorch
FROM python:3.12-slim AS base

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    wget \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Stage 2: Dependencies
FROM base AS dependencies

# Copy requirements first (cache optimization)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Stage 3: Application
FROM dependencies AS application

# Copy application code
COPY pontos/ ./pontos/
COPY models/ ./models/
COPY setup.py .
COPY README.md .
COPY .env.example .

# Install pontos
RUN pip install --no-cache-dir -e .

# Create data and output directories
RUN mkdir -p data runs /app/.config/Ultralytics && \
    chmod -R 777 /app/.config

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV DEVICE=cpu
ENV YOLO_CONFIG_DIR=/app/.config/Ultralytics

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import pontos; print('OK')" || exit 1

# Entry point
ENTRYPOINT ["pontos"]
CMD ["--help"]
