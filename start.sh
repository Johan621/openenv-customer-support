#!/bin/bash
set -e

echo "=========================================="
echo "  Customer Support Triage RL Environment"
echo "=========================================="
echo ""
echo "Starting FastAPI server..."

python -m uvicorn server.app:app --host 0.0.0.0 --port 8000 &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to be ready..."
for i in $(seq 1 30); do
    if curl -sf http://localhost:8000/health > /dev/null 2>&1; then
        echo "✅ Server is ready at http://0.0.0.0:8000"
        echo "   - Web UI:    http://0.0.0.0:8000/web"
        echo "   - API docs:  http://0.0.0.0:8000/docs"
        echo "   - Health:    http://0.0.0.0:8000/health"
        break
    fi
    sleep 1
done

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

# Keep server running
wait $SERVER_PID
