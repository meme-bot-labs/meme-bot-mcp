#!/bin/bash

# Railway Deployment Script for MCP Meme Bot
# This script helps deploy to Railway with proper configuration

set -e

echo "🚂 Railway Deployment Script for MCP Meme Bot"
echo "============================================="

# Check if Railway CLI is installed
if ! command -v railway &> /dev/null; then
    echo "❌ Railway CLI not found. Installing..."
    npm install -g @railway/cli
fi

# Check if user is logged in
if ! railway whoami &> /dev/null; then
    echo "🔐 Please login to Railway:"
    railway login
fi

# Check for required environment variables
echo "🔍 Checking environment variables..."

if [ -z "$AUTH_TOKEN" ]; then
    echo "⚠️  AUTH_TOKEN not set. Please set it:"
    echo "   export AUTH_TOKEN=your_secure_token"
    read -p "Enter AUTH_TOKEN: " AUTH_TOKEN
    export AUTH_TOKEN
fi

if [ -z "$MY_NUMBER" ]; then
    echo "⚠️  MY_NUMBER not set. Please set it:"
    read -p "Enter your phone number: " MY_NUMBER
    export MY_NUMBER
fi

if [ -z "$GEMINI_API_KEY" ]; then
    echo "⚠️  GEMINI_API_KEY not set. Please set it:"
    read -p "Enter your Gemini API key: " GEMINI_API_KEY
    export GEMINI_API_KEY
fi

echo "✅ Environment variables configured"

# Initialize Railway project if needed
if [ ! -f ".railway" ]; then
    echo "🚂 Initializing Railway project..."
    railway init
fi

# Set environment variables on Railway
echo "🔧 Setting environment variables on Railway..."
railway variables set AUTH_TOKEN="$AUTH_TOKEN"
railway variables set MY_NUMBER="$MY_NUMBER"  
railway variables set GEMINI_API_KEY="$GEMINI_API_KEY"

# Deploy to Railway
echo "🚀 Deploying to Railway..."
railway up --detach

echo "✅ Deployment initiated!"
echo ""
echo "📊 Monitor deployment:"
echo "   railway logs --follow"
echo ""
echo "🔗 Get your app URL:"
echo "   railway domain"
echo ""
echo "🎯 Connect to MCP:"
echo "   /mcp connect https://your-app.railway.app/mcp Bearer $AUTH_TOKEN"
echo ""
echo "🎉 Deployment complete! Your MCP Meme Bot is live on Railway!"
