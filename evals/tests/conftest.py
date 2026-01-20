#!/usr/bin/env python3
"""
Pytest configuration for EXAID evaluation tests.

Ensures proper path setup for test discovery and execution.
"""

import sys
from pathlib import Path

# Add repo root to path so evals package imports resolve cleanly
repo_root = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(repo_root))

# Fixtures and configuration can be added here















