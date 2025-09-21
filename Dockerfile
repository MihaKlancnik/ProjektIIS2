# Use Python 3.11 slim image
FROM python:3.11-slim

ENV POETRY_VERSION=1.5.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_CACHE_DIR=/var/cache/pypoetry

# install system build deps commonly required by scientific/crypto packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make git curl wget \
    libffi-dev libssl-dev libbz2-dev liblzma-dev libsqlite3-dev \
    libatlas-base-dev libopenblas-dev pkg-config && \
    rm -rf /var/lib/apt/lists/*

# install poetry
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app
COPY pyproject.toml poetry.lock* /app/

# try a normal install; if it fails, the verbose retry helps debugging
RUN poetry config virtualenvs.create false && \
    poetry install --only=main --no-root --no-interaction --no-ansi || \
    poetry install --only=main --no-root --no-interaction --no-ansi --verbose

# Copy application code
COPY . .

# Set Python path to include src directory
# Define a build-time ARG so BuildKit won't warn about an undefined $PYTHONPATH
ARG PYTHONPATH
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

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
RUN python src/data/cleanup.py
CMD ["python", "wsgi.py"]


#/metrics
#//health
#docker-compose up -d crypto-predictor

#rebuild:
#docker build -t crypto-prediction:latest .

#docker pull ghcr.io/mihaklancnik/projektiis2:latest
#docker tag ghcr.io/your-username/projektiis2:latest crypto-prediction:latest

#REBUILD
#docker-compose build
#docker-compose up -d



#dob nov kontejner dol
#docker pull ghcr.io/mihaklancnik/projektiis2:latest

#zazen nov kontejner
#docker-compose up -d

#ugasnes
#docker-compose down


#lokalno:
#docker-compose up -d

#1. Dockerfile → Builds the container image
#2. docker-compose.yml → Defines how to run it
#3. wsgi.py → Starts your Flask app inside
#4. .dockerignore → Keeps builds clean
#5. GitHub Actions → Builds automatically when you push code
#6. Deploy scripts → Easy one-command deployment
#7. .env.example → Shows you what to configure