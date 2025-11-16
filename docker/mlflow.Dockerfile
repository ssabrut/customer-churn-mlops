# =============================================================================
# STAGE 1: The "builder" stage
# =============================================================================
FROM python:3.9.19-slim-bullseye AS builder

# Set environment variables for the virtual environment
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

# 1. Install build-time system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    python3-venv \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Create the virtual environment
RUN python -m venv $VENV_PATH

# 4. Install Python dependencies
COPY requirements.txt .

# - Install packages into the venv.
RUN pip install --no-cache-dir -r requirements.txt


# =============================================================================
# STAGE 2: The "final" stage
# =============================================================================
FROM python:3.9.19-slim-bullseye

# Set environment variables for the venv (must be set again)
ENV VENV_PATH=/opt/venv
ENV PATH="$VENV_PATH/bin:$PATH"

# 1. Install *runtime* system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libpq5 \
    # 2. Clean up apt cache
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 3. Create a non-root user and group
RUN useradd -r -m -d /home/appuser -s /bin/sh appuser

# 4. Create and set permissions for the app directory
WORKDIR /app
RUN chown appuser:appuser /app

# 5. Switch to the non-root user
USER appuser

# 6. Copy the virtual environment from the builder stage
COPY --from=builder --chown=appuser:appuser $VENV_PATH $VENV_PATH

# 7. Copy the application code
COPY --chown=appuser:appuser . .