#!/bin/bash
set -e

echo "=========================================="
echo "  Customer Support Triage RL Environment"
echo "=========================================="
echo ""
echo "Starting FastAPI server..."

python -m uvicorn server.app:app --host 0.0.0.0 --port 8000

# Run inference if HF_TOKEN is set
if [ -n "$HF_TOKEN" ]; then
    echo ""
    echo "HF_TOKEN detected - running inference script..."
    python inference.py
    INFERENCE_EXIT=$?
    if [ $INFERENCE_EXIT -ne 0 ]; then
        echo "⚠️  Inference script exited with code $INFERENCE_EXIT (non-fatal)"
    fi
else
    echo ""
    echo "ℹ️  HF_TOKEN not set - skipping inference script"
fi