#!/bin/bash

# Simple script to run the ISL web app

cd "$(dirname "$0")"

echo "🚀 Starting ISL Recognition Web App..."
echo ""

# Activate virtual environment
source venv/bin/activate

# Set Python path
export PYTHONPATH=".:$PYTHONPATH"

# Start the app
echo "Starting server on http://localhost:8080"
echo "Press Ctrl+C to stop"
echo ""

python inference/app.py
