# syntax=docker/dockerfile:1

# Build stage
FROM python:3.11-slim as builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies
COPY pyproject.toml README.md ./
COPY value_ledger/ ./value_ledger/

# Install package
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .


# Production stage
FROM python:3.11-slim as production

WORKDIR /app

# Create non-root user for security
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy virtual environment from builder
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Copy application code
COPY --chown=appuser:appgroup value_ledger/ ./value_ledger/

# Create data directory for ledger storage
RUN mkdir -p /data && chown appuser:appgroup /data
VOLUME ["/data"]

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LEDGER_PATH=/data/ledger.jsonl

# Switch to non-root user
USER appuser

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from value_ledger import ValueLedger; print('healthy')" || exit 1

# Default command - run CLI help
ENTRYPOINT ["python", "-m", "value_ledger.cli"]
CMD ["--help"]


# Development stage with dev dependencies
FROM python:3.11-slim as development

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies including dev
COPY pyproject.toml README.md ./
COPY value_ledger/ ./value_ledger/
COPY tests/ ./tests/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -e ".[dev]"

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser && \
    chown -R appuser:appgroup /app

USER appuser

ENV PYTHONUNBUFFERED=1

CMD ["pytest", "tests/", "-v"]


# Embeddings stage - includes ML dependencies for novelty scoring
FROM python:3.11-slim as embeddings

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create virtual environment
RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Install Python dependencies with embeddings
COPY pyproject.toml README.md ./
COPY value_ledger/ ./value_ledger/

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir ".[embeddings]"

# Create non-root user
RUN groupadd --gid 1000 appgroup && \
    useradd --uid 1000 --gid appgroup --shell /bin/bash --create-home appuser

# Copy application
COPY --chown=appuser:appgroup value_ledger/ ./value_ledger/

# Create data and model cache directories
RUN mkdir -p /data /home/appuser/.cache && \
    chown -R appuser:appgroup /data /home/appuser/.cache

VOLUME ["/data", "/home/appuser/.cache"]

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LEDGER_PATH=/data/ledger.jsonl \
    TRANSFORMERS_CACHE=/home/appuser/.cache/huggingface

USER appuser

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "from value_ledger import ValueLedger; print('healthy')" || exit 1

ENTRYPOINT ["python", "-m", "value_ledger.cli"]
CMD ["--help"]
