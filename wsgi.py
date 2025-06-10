#!/usr/bin/env python3
"""
Production entry point for the crypto prediction application.
"""
import os
import sys
from pathlib import Path

# Add the src directory to Python path
src_path = Path(__file__).parent / "src"
sys.path.insert(0, str(src_path))

from app import app

if __name__ == "__main__":
    # Production configuration
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "0.0.0.0")
    debug = os.environ.get("FLASK_ENV") == "development"
    
    app.run(host=host, port=port, debug=debug)
