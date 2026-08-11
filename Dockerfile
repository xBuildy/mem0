FROM python:3.11-slim

WORKDIR /app

RUN pip install --no-cache-dir mem0ai fastapi uvicorn

COPY server.py .

EXPOSE 8012

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8012"]
