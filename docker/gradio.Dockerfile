FROM python:3.10-slim

WORKDIR /app
COPY entrypoint/ entrypoint/

RUN pip install --no-cache-dir gradio
EXPOSE 7860

CMD ["python", "entrypoint/gradio_app.py"]