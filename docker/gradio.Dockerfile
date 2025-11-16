FROM python:3.10-slim

# Set working directory
WORKDIR /app

# 1. Install *runtime* system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    libgomp1 \
    libjpeg62-turbo \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 2. Create a non-root user and group
RUN useradd -r -m -d /home/appuser -s /bin/sh appuser

# 3. Install Python dependencies
COPY --chown=appuser:appuser docker/gradio_requirements.txt .

RUN pip install --no-cache-dir -r gradio_requirements.txt

# 4. Copy application code
COPY --chown=appuser:appuser entrypoint/ entrypoint/

# 5. Switch to the non-root user
USER appuser

# Expose the port
EXPOSE 7860

# Run the application
CMD ["python", "entrypoint/gradio_app.py"]