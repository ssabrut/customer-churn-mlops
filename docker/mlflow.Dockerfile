FROM python:3.10-slim

# Install system dependencies (required for psycopg2)
RUN apt-get update \
    && apt-get -y install libpq-dev gcc \
    && rm -rf /var/lib/apt/lists/*

# Install python packages
RUN pip install mlflow psycopg2 boto3 minio gunicorn