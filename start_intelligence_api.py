#!/usr/bin/env python3
"""
Zen Intelligence API Startup Script

This script starts the Zen Intelligence API server with proper configuration
and integration with the existing Zen MCP server infrastructure.
"""

import os
import sys
import uvicorn
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def main():
    """Start the Zen Intelligence API server."""
    
    # Set default environment variables if not already set
    os.environ.setdefault("LOG_LEVEL", "INFO")
    os.environ.setdefault("ZEN_API_HOST", "0.0.0.0")
    os.environ.setdefault("ZEN_API_PORT", "8000")
    
    # Import the FastAPI app
    try:
        from zen_intelligence_api.api.main import app
    except ImportError as e:
        print(f"Failed to import Zen Intelligence API: {e}")
        print("Make sure you've installed the requirements:")
        print("  pip install -r requirements-intelligence-api.txt")
        sys.exit(1)
    
    # Get configuration from environment
    host = os.getenv("ZEN_API_HOST", "0.0.0.0")
    port = int(os.getenv("ZEN_API_PORT", "8000"))
    workers = int(os.getenv("ZEN_API_WORKERS", "1"))
    reload = os.getenv("ZEN_API_RELOAD", "true").lower() == "true"
    log_level = os.getenv("LOG_LEVEL", "info").lower()
    
    print("🤖 Starting Zen Intelligence API Server")
    print("=" * 50)
    print(f"Host: {host}")
    print(f"Port: {port}")
    print(f"Workers: {workers}")
    print(f"Reload: {reload}")
    print(f"Log Level: {log_level}")
    print("=" * 50)
    print(f"API Docs: http://{host}:{port}/docs")
    print(f"Health Check: http://{host}:{port}/health")
    print("=" * 50)
    
    # Start the server
    uvicorn.run(
        "zen_intelligence_api.api.main:app",
        host=host,
        port=port,
        workers=workers if not reload else 1,  # Workers only work without reload
        reload=reload,
        log_level=log_level,
        access_log=True
    )


if __name__ == "__main__":
    main()