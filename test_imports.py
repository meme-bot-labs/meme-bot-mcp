#!/usr/bin/env python3
"""
Simple script to test if all imports work correctly
This helps debug issues during Docker build
"""

import sys

def test_imports():
    """Test critical imports"""
    try:
        print("Testing basic imports...")
        
        # Test Python standard library
        import asyncio
        import os
        print("✅ Standard library imports OK")
        
        # Test FastMCP
        from fastmcp import FastMCP
        print("✅ FastMCP import OK")
        
        # Test MCP
        from mcp import ErrorData, McpError
        print("✅ MCP imports OK")
        
        # Test FastAPI
        from fastapi import Request, HTTPException
        print("✅ FastAPI imports OK")
        
        # Test other critical dependencies
        import httpx
        import google.generativeai as genai
        from PIL import Image
        print("✅ External dependencies OK")
        
        print("🎉 All imports successful!")
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        return False

if __name__ == "__main__":
    success = test_imports()
    sys.exit(0 if success else 1)
