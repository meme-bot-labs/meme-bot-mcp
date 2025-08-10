#!/bin/bash

# Docker entrypoint script for MCP Meme Bot
# This script handles initialization and graceful startup

set -e

# Default values - Railway sets PORT automatically
# Use IPv6 dual-stack binding for Railway V2 compatibility
MCP_HOST=${MCP_HOST:-$([ -n "$RAILWAY_ENVIRONMENT" ] && echo "::" || echo "0.0.0.0")}
MCP_PORT=${PORT:-${MCP_PORT:-8086}}
AUTH_TOKEN=${AUTH_TOKEN:-"devtoken_shared_for_team"}

echo "🚀 Starting MCP Meme Bot on Railway..."
echo "📡 Host: $MCP_HOST"
echo "🔌 Port: $MCP_PORT"
echo "🚂 Railway Environment: ${RAILWAY_ENVIRONMENT:-local}"
echo "🔐 Auth: ${AUTH_TOKEN:0:8}... (${#AUTH_TOKEN} chars)"

# Validate required environment variables
if [ -z "$AUTH_TOKEN" ]; then
    echo "❌ ERROR: AUTH_TOKEN environment variable is required"
    exit 1
fi

if [ -z "$MY_NUMBER" ]; then
    echo "⚠️  WARNING: MY_NUMBER not set, using default"
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  WARNING: GEMINI_API_KEY not set, some features may not work"
fi

# Create logs directory if it doesn't exist
mkdir -p /app/logs

# Wait for any dependencies (if needed)
# Add health checks for external services here

# Set proper permissions
chmod -R 755 /app

# Function to handle graceful shutdown
cleanup() {
    echo "🛑 Received shutdown signal, stopping gracefully..."
    kill -TERM "$child_pid" 2>/dev/null || true
    wait "$child_pid"
    echo "✅ Shutdown complete"
    exit 0
}

# Set up signal handlers
trap cleanup SIGTERM SIGINT

# Start the MCP server
echo "🎭 Launching MCP Meme Bot server..."

# Check if we're in development mode
if [ "$NODE_ENV" = "development" ] || [ "$PYTHON_ENV" = "development" ]; then
    echo "🔧 Running in development mode"
    python -u mcp-bearer-token/mcp_starter.py &
else
    echo "🏭 Running in production mode"
    python mcp-bearer-token/mcp_starter.py &
fi

child_pid=$!

# Wait for the process to finish
wait "$child_pid"
