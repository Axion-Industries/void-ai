FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy project files
COPY . .

# Create required directories
RUN mkdir -p logs out data/void && \
    chmod -R 755 logs out data/void

# Set environment variables
ENV MODEL_PATH=/app/out/model.pt \
    VOCAB_PATH=/app/data/void/vocab.pkl \
    META_PATH=/app/data/void/meta.pkl \
    PYTHONUNBUFFERED=1 \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PYTHONDONTWRITEBYTECODE=1

# Initialize model files
RUN python init_model.py

# Verify setup
RUN python verify_setup.py
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Expose the port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Run the application with proper worker configuration
CMD exec gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 1 --timeout 300 --access-logfile - --error-logfile - --log-level debug --capture-output --worker-class sync --preload chat_api:app
