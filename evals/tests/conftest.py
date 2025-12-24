#!/usr/bin/env python3
"""
Pytest configuration for EXAID evaluation tests.

Ensures proper path setup for test discovery and execution.
"""

import sys
from pathlib import Path

# Add src directory to path for imports
src_dir = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_dir))

# Fixtures and configuration can be added here




