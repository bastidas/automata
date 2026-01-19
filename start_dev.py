#!/usr/bin/env python3
"""
Acinonyx Development Server Launcher
Starts both frontend and backend servers using centralized configuration
"""
from __future__ import annotations

import os
import subprocess
import sys
import time

from configs.appconfig import BACKEND_PORT
from configs.appconfig import FRONTEND_PORT


def start_backend():
    """Start the backend server"""
    print(f'🐆 Starting Acinonyx Backend on port {BACKEND_PORT}...')
    backend_process = subprocess.Popen(
        [sys.executable, 'backend/run_server.py'],
        cwd=os.path.dirname(os.path.abspath(__file__)),
    )
    return backend_process


def start_frontend():
    """Start the frontend server"""
    print(f'⚛️  Starting Acinonyx Frontend on port {FRONTEND_PORT}...')
    frontend_process = subprocess.Popen(
        ['npm', 'run', 'dev'],
        cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), 'frontend'),
    )
    return frontend_process


def main():
    """Main function to start both servers"""
    print('🚀 Starting Acinonyx Development Environment...')
    print(f'   Frontend: http://localhost:{FRONTEND_PORT}')
    print(f'   Backend:  http://localhost:{BACKEND_PORT}')
    print('   Press Ctrl+C to stop both servers')
    print('-' * 50)

    try:
        # Start backend first
        backend_process = start_backend()
        time.sleep(2)  # Give backend time to start

        # Start frontend
        frontend_process = start_frontend()

        # Wait for both processes
        try:
            backend_process.wait()
        except KeyboardInterrupt:
            print('\n🛑 Stopping servers...')
            backend_process.terminate()
            frontend_process.terminate()

            # Wait for graceful shutdown
            try:
                backend_process.wait(timeout=5)
                frontend_process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                backend_process.kill()
                frontend_process.kill()

            print('✅ Servers stopped successfully')

    except Exception as e:
        print(f'❌ Error starting servers: {e}')
        sys.exit(1)


if __name__ == '__main__':
    main()
