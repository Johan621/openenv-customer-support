#!/bin/bash
set -e

echo "=========================================="
echo "  Customer Support Triage RL Environment"
echo "=========================================="
echo ""

PORT="${PORT:-7860}"
HOST="${HOST:-0.0.0.0}"

echo "Starting FastAPI server on ${HOST}:${PORT} ..."
exec python -m uvicorn server.app:app --host "${HOST}" --port "${PORT}"