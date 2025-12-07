#!/bin/bash
# Run the Flask-React Well Log Viewer

cd "$(dirname "$0")"

echo "🛢️ WellLog Viewer Pro - Flask + React"
echo "======================================"

# Check if backend dependencies are installed
echo "📦 Checking backend dependencies..."
cd backend
pip install -r requirements.txt -q

# Start backend in background
echo "🚀 Starting Flask backend on port 5000..."
python app.py &
BACKEND_PID=$!

cd ../frontend

# Check if node_modules exists
if [ ! -d "node_modules" ]; then
    echo "📦 Installing frontend dependencies..."
    npm install
fi

# Start frontend
echo "🚀 Starting React frontend on port 3000..."
npm run dev &
FRONTEND_PID=$!

echo ""
echo "✅ Services started!"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:5000"
echo ""
echo "Press Ctrl+C to stop all services"

# Wait for Ctrl+C
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null; exit" INT
wait
