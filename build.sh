#!/bin/bash
set -e

echo "🚀 Starting Render build process..."

# Install Python dependencies
echo "📦 Installing dependencies..."
pip install --upgrade pip
pip install -r requirements.txt

# Download ML model
echo "🤖 Setting up ML model..."
python download_model.py

echo "✅ Build complete!"
