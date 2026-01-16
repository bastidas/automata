#!/usr/bin/env python3
"""
Acinonyx Backend Server
Uses centralized port configuration from configs.appconfig
"""

import uvicorn
import sys
import os

# Add parent directory to path to import configs
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from configs.appconfig import BACKEND_PORT

if __name__ == "__main__":
    print(f"🐆 Starting Acinonyx Backend Server on port {BACKEND_PORT}...")
    uvicorn.run("query_api:app", host="0.0.0.0", port=BACKEND_PORT, reload=True)