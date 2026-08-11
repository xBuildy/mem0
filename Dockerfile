FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir fastapi uvicorn httpx qdrant-client

COPY server.py .

EXPOSE 8012

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8012"]
