#!/bin/bash
# Restart script for EXAID backend and frontend servers
# This script kills existing processes and restarts them

echo "🛑 Stopping EXAID servers..."

# Kill backend process (Python/FastAPI on port 8000)
BACKEND_PIDS=$(lsof -ti:8000 2>/dev/null)
if [ ! -z "$BACKEND_PIDS" ]; then
    echo "Killing backend processes..."
    kill -9 $BACKEND_PIDS 2>/dev/null
fi

# Kill frontend process (Next.js on port 3000 or 3001)
FRONTEND_PIDS_3000=$(lsof -ti:3000 2>/dev/null)
FRONTEND_PIDS_3001=$(lsof -ti:3001 2>/dev/null)

if [ ! -z "$FRONTEND_PIDS_3000" ]; then
    echo "Killing frontend process on port 3000..."
    kill -9 $FRONTEND_PIDS_3000 2>/dev/null
fi

if [ ! -z "$FRONTEND_PIDS_3001" ]; then
    echo "Killing frontend process on port 3001..."
    kill -9 $FRONTEND_PIDS_3001 2>/dev/null
fi

# Wait a moment for processes to fully terminate
sleep 2

echo "🚀 Starting EXAID servers..."

# Get the script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
PROJECT_ROOT="$SCRIPT_DIR"

# Start backend in background
echo "Starting backend server..."
cd "$PROJECT_ROOT"
python3 "demos/backend/server.py" > /dev/null 2>&1 &
BACKEND_PID=$!
echo "Backend started (PID: $BACKEND_PID)"

# Wait a moment for backend to start
sleep 3

# Start frontend in background
echo "Starting frontend server..."
cd "$PROJECT_ROOT/demos/frontend"
npm run dev > /dev/null 2>&1 &
FRONTEND_PID=$!
echo "Frontend started (PID: $FRONTEND_PID)"

echo "✅ Servers restarted!"
echo "Backend: http://localhost:8000"
echo "Frontend: http://localhost:3000"

