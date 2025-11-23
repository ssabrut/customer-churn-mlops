# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.10-slim AS builder

WORKDIR /app

# Install build dependencies (compilers and development headers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

# Copy requirements first to leverage Docker caching
COPY requirements.txt .

# Create a virtual environment and install dependencies
RUN uv venv /opt/venv && \
    uv pip install --no-cache -r requirements.txt --python /opt/venv

# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.10-slim AS runtime

WORKDIR /app

# Set environment variables
# PYTHONDONTWRITEBYTECODE: Prevents Python from writing .pyc files
# PYTHONUNBUFFERED: Ensures logs are flushed immediately
# PATH: Adds our virtual env to the path automatically
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PATH="/opt/venv/bin:$PATH"

# Install ONLY runtime dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq5 \
    jq \
    && rm -rf /var/lib/apt/lists/*

# Create a non-root user for security
RUN groupadd -g 1000 appuser && \
    useradd -u 1000 -g appuser -s /bin/bash -m appuser

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Copy application code
COPY --chown=appuser:appuser . .

# Ensure the entrypoint script is executable
RUN chmod +x ./entrypoint/feast.sh

# Switch to the non-root user
USER appuser

# Expose the port (informational)
EXPOSE 8000

# Set the entrypoint
ENTRYPOINT ["./entrypoint/feast.sh"]

# Default command
CMD ["uvicorn", "core.main:app", "--host", "0.0.0.0", "--port", "8000"]