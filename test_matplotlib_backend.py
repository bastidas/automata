#!/usr/bin/env python3
"""Test matplotlib backend configuration for GUI-free plotting"""
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

print("🧪 Testing Matplotlib Backend Configuration...")

# Test that matplotlib uses the correct backend
import matplotlib
backend = matplotlib.get_backend()
print(f"Current backend: {backend}")

if backend == 'Agg':
    print("✅ SUCCESS: Using non-GUI 'Agg' backend")
else:
    print(f"⚠️  WARNING: Using '{backend}' backend")

print("Done!")