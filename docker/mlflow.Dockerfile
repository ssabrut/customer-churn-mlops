FROM python:3.9.19-slim-bullseye

RUN apt-get update \
    && apt-get -y install libpq-dev gcc \ 
    && pip install mlflow psycopg2 boto3 minio