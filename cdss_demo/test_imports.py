"""
Test script to verify CDSS imports work correctly.
Run this from the project root: python -m cdss_demo.test_imports
"""
import sys
from pathlib import Path

# Ensure project root is in path
project_root = Path(__file__).parent.parent.resolve()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from exaid import EXAID
    print("✓ EXAID import successful")
    
    from agents.buffer_agent import BufferAgent
    print("✓ BufferAgent import successful")
    
    from cdss_demo.cdss import CDSS
    print("✓ CDSS import successful")
    
    from cdss_demo.agents.orchestrator_agent import OrchestratorAgent
    print("✓ OrchestratorAgent import successful")
    
    print("\nAll imports successful! CDSS should work correctly.")
except ImportError as e:
    print(f"✗ Import error: {e}")
    print(f"Project root: {project_root}")
    print(f"Python path: {sys.path[:3]}")  # Show first 3 entries
    sys.exit(1)

