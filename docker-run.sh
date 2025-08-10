#!/bin/bash

# Docker run script for MCP Meme Bot
# This script builds and runs the Docker container with proper configuration

set -e

echo "🐳 Building MCP Meme Bot Docker image..."

# Build the Docker image
docker build -t meme-bot-mcp:latest .

echo "🚀 Starting MCP Meme Bot container..."

# Run the container
docker run -d \
  --name meme-bot-mcp \
  --restart unless-stopped \
  -p 8086:8086 \
  -v "$(pwd)/logs:/app/logs" \
  --health-cmd="curl -f http://localhost:8086/mcp/ || exit 1" \
  --health-interval=30s \
  --health-timeout=10s \
  --health-retries=3 \
  --health-start-period=40s \
  meme-bot-mcp:latest

echo "✅ MCP Meme Bot container started successfully!"
echo "📡 Server will be available at: http://localhost:8086/mcp/"
echo "🔍 Check container status: docker ps"
echo "📋 View logs: docker logs meme-bot-mcp"
echo "🛑 Stop container: docker stop meme-bot-mcp"

# Wait a moment and check if container is running
sleep 5
if docker ps | grep -q meme-bot-mcp; then
    echo "🎉 Container is running healthy!"
    echo "🌐 Use this with Cloudflare tunnel for external access:"
    echo "   docker run -d --name cf-tunnel --network host cloudflare/cloudflared:latest tunnel --url http://localhost:8086"
else
    echo "❌ Container failed to start. Check logs:"
    docker logs meme-bot-mcp
fi
