FROM python:3.10.14-slim

WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_PROGRESS_BAR=off

COPY . /app

RUN python -m pip install -e .

EXPOSE 7860

CMD ["sh", "-lc", "python -m uvicorn server.app:app --host 0.0.0.0 --port ${PORT:-7860} --no-access-log --log-level warning"]
