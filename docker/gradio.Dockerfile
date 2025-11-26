# ==========================================
# Stage 1: Builder
# ==========================================
FROM python:3.10-slim AS builder

# Set environment variables to reduce python buffering and pip clutter
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Install system dependencies required for building Python packages
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Create a virtual environment to isolate dependencies
RUN python -m venv /opt/venv
# Add venv to path so we can use it directly
ENV PATH="/opt/venv/bin:$PATH"

# Copy only requirements first to leverage Docker Layer Caching
COPY requirements.txt .

# Install dependencies into the virtual environment
RUN pip install -r requirements.txt

# ==========================================
# Stage 2: Runtime
# ==========================================
FROM python:3.10-slim AS runner

WORKDIR /app

# Create a non-root user for security
RUN groupadd -r appgroup && useradd -r -g appgroup appuser

# Install only runtime system dependencies (if any specific libs are needed like libgomp1 for torch)
RUN apt-get update && rm -rf /var/lib/apt/lists/*

# Copy the virtual environment from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Set environment variables
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    # GRADIO_SERVER_NAME is crucial for Docker networking
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT="7860"

# Copy the application code
COPY . .

# Change ownership of the app directory to the non-root user
RUN chown -R appuser:appgroup /app

# Switch to non-root user
USER appuser

EXPOSE 7860

# CMD using the venv python
CMD ["python", "entrypoint/gradio_app.py"]