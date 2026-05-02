#!/bin/bash
set -e

echo "🚀 Installing Talos Backend..."

# Check prerequisites
command -v docker >/dev/null 2>&1 || { echo "❌ Docker required"; exit 1; }
command -v docker-compose >/dev/null 2>&1 || { echo "❌ Docker Compose required"; exit 1; }

# Check .env exists
if [ ! -f .env ]; then
    echo "❌ .env file not found. Copy .env.example and configure it."
    exit 1
fi

# Pull and start
echo "📦 Pulling images..."
docker-compose pull

echo "🔧 Starting services..."
docker-compose up -d

echo "⏳ Waiting for database..."
sleep 10

echo "✅ Talos Backend installed successfully!"
echo "📍 API available at http://localhost:8000"
echo "📊 Health check: curl http://localhost:8000/health"