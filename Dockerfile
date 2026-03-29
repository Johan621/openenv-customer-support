# syntax=docker/dockerfile:1
FROM python:3.11-slim

# --- Labels ---
LABEL maintainer="Johan621" \
      description="Customer Support Triage RL Environment" \
      version="0.1.0"

# --- System dependencies ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# --- Working directory ---
WORKDIR /app

# --- Copy dependency files first (for layer caching) ---
COPY pyproject.toml ./

# --- Install Python dependencies ---
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        "fastapi>=0.111.0" \
        "uvicorn[standard]>=0.29.0" \
        "pydantic>=2.7.0" \
        "websockets>=12.0" \
        "httpx>=0.27.0" \
        "numpy>=1.26.0" \
        "scikit-learn>=1.4.0" \
        "python-dotenv>=1.0.0"

# --- Copy application code ---
COPY . .

# --- Create non-root user ---
RUN adduser --disabled-password --gecos "" appuser \
    && chown -R appuser:appuser /app
USER appuser

# --- Expose port ---
EXPOSE 8000

# --- Health check ---
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# --- Start server ---
CMD ["python", "-m", "uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
