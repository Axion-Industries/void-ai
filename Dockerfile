FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first to leverage Docker cache
COPY requirements.txt requirements-dev.txt ./
RUN pip install --no-cache-dir -r requirements.txt -r requirements-dev.txt

# Copy project files
COPY . .

# Create log directory
RUN mkdir -p logs

# Verify setup
RUN python verify_setup.py

# Set environment variables
ENV PORT=10000
ENV PYTHONUNBUFFERED=1
ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8

# Expose the port
EXPOSE 10000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:10000/health || exit 1

# Run the application with proper worker configuration
CMD ["gunicorn", \
    "--bind", "0.0.0.0:10000", \
    "--workers", "4", \
    "--threads", "2", \
    "--timeout", "120", \
    "--access-logfile", "logs/access.log", \
    "--error-logfile", "logs/error.log", \
    "--log-level", "info", \
    "--worker-class", "sync", \
    "chat_api:app"]
