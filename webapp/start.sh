#!/bin/bash

# Quick Start Script for Brain MRI Classification Web Application
# Starts backend and frontend automatically

set -e  # Exit on error

echo "=============================================="
echo "Brain MRI Classifier Web Application"
echo "=============================================="
echo ""

# Colors
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Check if Hadoop is running
echo -e "${BLUE}[INFO]${NC} Checking prerequisites..."

if command -v jps &> /dev/null; then
    if jps | grep -q "NameNode"; then
        echo -e "${GREEN}[OK]${NC} Hadoop is running"
    else
        echo -e "${YELLOW}[WARN]${NC} Hadoop not detected. You may need to start it manually."
        echo "       Run: start-dfs.sh"
    fi
else
    echo -e "${YELLOW}[WARN]${NC} JPS not found. Cannot verify Hadoop status."
fi

# Check for model file
MODEL_PATH="../best_model_stage1.keras"
if [ -f "$MODEL_PATH" ]; then
    echo -e "${GREEN}[OK]${NC} Model file found"
else
    echo -e "${RED}[ERROR]${NC} Model file not found at $MODEL_PATH"
    echo "       Please ensure best_model_stage1.keras exists in the project root."
    exit 1
fi

# Start Backend
echo ""
echo -e "${BLUE}[INFO]${NC} Starting Backend (Flask API)..."
cd backend

# Check if virtual environment exists, create if not
if [ ! -d "venv" ] && [ ! -d "../dtsvenv" ]; then
    echo -e "${BLUE}[INFO]${NC} Creating virtual environment..."
    python3 -m venv venv
    source venv/bin/activate
    pip install -q -r requirements.txt
elif [ -d "venv" ]; then
    source venv/bin/activate
else
    source ../../dtsvenv/bin/activate
fi

# Start Flask in background
python app.py &
BACKEND_PID=$!
echo -e "${GREEN}[OK]${NC} Backend starting (PID: $BACKEND_PID)"

cd ..

# Wait for backend to be ready
echo -e "${BLUE}[INFO]${NC} Waiting for backend to initialize (this may take 20-30 seconds)..."
sleep 5

# Check backend health with retries
MAX_RETRIES=12
RETRY_COUNT=0
while [ $RETRY_COUNT -lt $MAX_RETRIES ]; do
    if curl -s http://localhost:5000/api/health > /dev/null 2>&1; then
        echo -e "${GREEN}[OK]${NC} Backend is responding"
        break
    fi
    RETRY_COUNT=$((RETRY_COUNT + 1))
    if [ $RETRY_COUNT -lt $MAX_RETRIES ]; then
        echo -e "${BLUE}[INFO]${NC} Waiting for backend... ($RETRY_COUNT/$MAX_RETRIES)"
        sleep 5
    fi
done

if [ $RETRY_COUNT -eq $MAX_RETRIES ]; then
    echo -e "${RED}[ERROR]${NC} Backend did not respond in time"
    kill $BACKEND_PID 2>/dev/null
    exit 1
fi

# Start Frontend
echo ""
echo -e "${BLUE}[INFO]${NC} Starting Frontend (React)..."
cd frontend

# Install dependencies if needed
if [ ! -d "node_modules" ]; then
    echo -e "${BLUE}[INFO]${NC} Installing npm dependencies..."
    npm install
fi

# Start React
npm start &
FRONTEND_PID=$!
echo -e "${GREEN}[OK]${NC} Frontend starting (PID: $FRONTEND_PID)"

cd ..

# Summary
echo ""
echo "=============================================="
echo -e "${GREEN}Application Started Successfully!${NC}"
echo "=============================================="
echo ""
echo "Frontend: http://localhost:3000"
echo "Backend:  http://localhost:5000"
echo ""
echo "Process IDs:"
echo "  Backend:  $BACKEND_PID"
echo "  Frontend: $FRONTEND_PID"
echo ""
echo "To stop the application, run:"
echo "  kill $BACKEND_PID $FRONTEND_PID"
echo ""
echo "Or press Ctrl+C in this terminal"
echo ""

# Save PIDs to file for easy cleanup
echo "$BACKEND_PID" > .backend.pid
echo "$FRONTEND_PID" > .frontend.pid

# Wait for user interrupt
trap "echo ''; echo 'Shutting down...'; kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; rm -f .backend.pid .frontend.pid; exit 0" INT TERM
wait
