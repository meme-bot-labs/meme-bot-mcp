#!/usr/bin/env python3
"""
Simple MCP Meme Bot Server for Railway Deployment
Fixed for FastMCP 2.11.2 API compatibility
"""

import asyncio
import os
import time
from fastmcp import FastMCP
from mcp.types import TextContent

# --- Load environment variables ---
TOKEN = os.environ.get("AUTH_TOKEN", "test_token_123")
MY_NUMBER = os.environ.get("MY_NUMBER", "1234567890") 
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "test_key")

print(f"🔐 Auth Token: {'✅ Set' if TOKEN else '❌ Missing'}")
print(f"📱 Phone Number: {'✅ Set' if MY_NUMBER else '❌ Missing'}")
print(f"🤖 Gemini API: {'✅ Set' if GEMINI_API_KEY else '❌ Missing'}")

# --- Simple MCP Server Setup ---
mcp = FastMCP("Meme Bot")

# --- Health Check Route ---
from starlette.responses import JSONResponse

async def health_check(request):
    """Health check endpoint for Railway"""
    return JSONResponse({
        "status": "healthy",
        "service": "MCP Meme Bot",
        "version": "1.0.0", 
        "timestamp": time.time(),
        "environment": {
            "railway": bool(os.environ.get("RAILWAY_ENVIRONMENT")),
            "port": os.environ.get("PORT", "not_set"),
            "host": os.environ.get("MCP_HOST", "not_set")
        }
    })

async def root_endpoint(request):
    """Root endpoint"""
    return JSONResponse({
        "service": "MCP Meme Bot",
        "status": "running",
        "endpoints": {
            "mcp": "/mcp/",
            "health": "/health"
        }
    })

# Add custom routes
mcp.custom_route("/health", ["GET"], name="health")(health_check)
mcp.custom_route("/", ["GET"], name="root")(root_endpoint)

# --- Required validation tool ---
@mcp.tool
async def validate() -> str:
    """Validation tool required by Puch"""
    return MY_NUMBER

# --- Simple meme tool ---
@mcp.tool
async def create_simple_meme(text: str) -> list[TextContent]:
    """Create a simple text meme"""
    meme_text = f"🎭 MEME: {text.upper()} 🎭"
    return [TextContent(type="text", text=meme_text)]

# --- Server startup ---
async def main():
    # Railway compatibility - use PORT env var
    port = int(os.environ.get("PORT", 8086))
    # Railway V2 requires IPv6 binding
    host = "::" if os.environ.get("RAILWAY_ENVIRONMENT") else "0.0.0.0"
    
    print(f"🚀 Starting MCP server on http://{host}:{port}")
    print(f"🚂 Railway Environment: {os.environ.get('RAILWAY_ENVIRONMENT', 'local')}")
    
    try:
        await mcp.run_async("streamable-http", host=host, port=port)
    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    asyncio.run(main())
