# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    POETRY_NO_INTERACTION=1 \
    POETRY_VENV_IN_PROJECT=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Poetry
RUN pip install poetry

# Copy poetry configuration files and README
COPY pyproject.toml poetry.lock* README.md ./

# Configure poetry and install dependencies
RUN poetry config virtualenvs.create false \
    && poetry install --only=main --no-root --no-interaction --no-ansi \
    && rm -rf /tmp/poetry_cache

# Copy application code
COPY . .

# Set Python path to include src directory
ENV PYTHONPATH="/app/src:$PYTHONPATH"

# Create necessary directories
RUN mkdir -p data/preprocessed/price \
    && mkdir -p data/preprocessed/fear_greed \
    && mkdir -p predictions \
    && mkdir -p models \
    && mkdir -p reports

# Expose port
EXPOSE 5000

# Set environment variables for Flask
ENV FLASK_APP=wsgi.py
ENV FLASK_RUN_HOST=0.0.0.0
ENV FLASK_RUN_PORT=5000
ENV FLASK_ENV=production

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:5000/ || exit 1

# Run the application
CMD ["python", "wsgi.py"]


#/metrics
#//health
#docker-compose up -d crypto-predictor

#rebuild:
#docker build -t crypto-prediction:latest .

#docker pull ghcr.io/your-username/projektiis2:latest
#docker tag ghcr.io/your-username/projektiis2:latest crypto-prediction:latest

#REBUILD
#docker-compose build
#docker-compose up -d