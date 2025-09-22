# Use Python 3.11 slim image (trixie). Works fine with OpenBLAS/LAPACK.
FROM python:3.11-slim

ENV POETRY_VERSION=1.5.1 \
    POETRY_VIRTUALENVS_CREATE=false \
    POETRY_NO_INTERACTION=1 \
    PIP_NO_CACHE_DIR=1 \
    POETRY_CACHE_DIR=/var/cache/pypoetry

# System build deps (no ATLAS; add gfortran + lapack for SciPy/NumPy)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential gcc g++ make git curl wget \
    libffi-dev libssl-dev libbz2-dev liblzma-dev libsqlite3-dev \
    gfortran libopenblas-dev liblapack-dev pkg-config \
 && rm -rf /var/lib/apt/lists/*

# Poetry
RUN pip install "poetry==$POETRY_VERSION"

WORKDIR /app
COPY pyproject.toml poetry.lock* /app/

# Install only main deps first to leverage layer caching
RUN poetry config virtualenvs.create false && \
    poetry install --only=main --no-root --no-interaction --no-ansi || \
    poetry install --only=main --no-root --no-interaction --no-ansi --verbose

# Copy the rest
COPY . .

# Keep PYTHONPATH, but avoid BuildKit warning by defining ARG first
ARG PYTHONPATH
ENV PYTHONPATH="/app/src:${PYTHONPATH}"

# Create necessary directories
RUN mkdir -p data/preprocessed/price \
    data/preprocessed/fear_greed \
    predictions models reports

# Expose and Flask env
EXPOSE 5000
ENV FLASK_APP=wsgi.py \
    FLASK_RUN_HOST=0.0.0.0 \
    FLASK_RUN_PORT=5000 \
    FLASK_ENV=production

# Health check (curl is installed above)
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
  CMD curl -f http://localhost:5000/ || exit 1

# ⚠️ Optional: this runs at build-time. If it hits the network or needs secrets,
# it will make your builds flaky. Prefer doing this at runtime or in CI instead.
# RUN python src/data/cleanup.py

CMD ["python", "wsgi.py"]
