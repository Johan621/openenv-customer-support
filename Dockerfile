FROM python:3.10-slim

WORKDIR /app
ENV PYTHONUNBUFFERED=1

COPY pyproject.toml /app/pyproject.toml

RUN pip install --no-cache-dir -U pip setuptools wheel && \
    pip install --no-cache-dir -e .

COPY . /app

EXPOSE 7860

CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "7860"]