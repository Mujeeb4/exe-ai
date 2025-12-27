"""
Entry point script for Exe CLI.
Allows running with: python exe.py
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.main import app

if __name__ == "__main__":
    app()
