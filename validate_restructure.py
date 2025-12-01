#!/usr/bin/env python3
"""Automated validation script for EXAID restructure."""
import sys
import subprocess
from pathlib import Path

def test_import(import_statement, description):
    """Test a single import statement."""
    try:
        result = subprocess.run(
            [sys.executable, "-c", import_statement],
            capture_output=True,
            text=True,
            timeout=10
        )
        if result.returncode == 0:
            print(f"✓ {description}")
            return True
        else:
            print(f"✗ {description}")
            print(f"  Error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ {description}")
        print(f"  Exception: {e}")
        return False

def check_file_exists(filepath, description):
    """Check if a file exists."""
    if Path(filepath).exists():
        print(f"✓ {description}")
        return True
    else:
        print(f"✗ {description}: {filepath} not found")
        return False

def check_no_syspath_manipulation():
    """Verify no sys.path manipulation remains."""
    import os
    found = []
    for root, dirs, files in os.walk("."):
        # Skip hidden dirs and venv
        dirs[:] = [d for d in dirs if not d.startswith('.') and d not in ['venv', '.venv', '__pycache__', 'node_modules', '.next']]
        for file in files:
            if file.endswith('.py') and file != 'validate_restructure.py':  # Skip the validation script itself
                filepath = Path(root) / file
                try:
                    content = filepath.read_text(encoding='utf-8')
                    if 'sys.path.insert' in content:
                        found.append(str(filepath))
                except:
                    pass
    if not found:
        print("✓ No sys.path.insert found")
        return True
    else:
        print("✗ sys.path.insert still found in:")
        for f in found:
            print(f"  {f}")
        return False

def main():
    print("="*60)
    print("EXAID Restructure Validation")
    print("="*60)
    
    results = []
    
    print("\n[Phase 1: Core EXAID Imports]")
    results.append(test_import(
        "from exaid_core import EXAID",
        "EXAID core import"
    ))
    results.append(test_import(
        "from exaid_core.schema import AgentSummary",
        "AgentSummary import"
    ))
    results.append(test_import(
        "from exaid_core.token_gate import TokenGate",
        "TokenGate import"
    ))
    results.append(test_import(
        "from exaid_core.buffer_agent import BufferAgent",
        "BufferAgent import"
    ))
    results.append(test_import(
        "from exaid_core.summarizer_agent import SummarizerAgent",
        "SummarizerAgent import"
    ))
    
    print("\n[Phase 2: Demo Imports]")
    results.append(test_import(
        "from demos.cdss_example import CDSS",
        "CDSS demo import"
    ))
    results.append(test_import(
        "from demos.backend.server import app",
        "Backend server import"
    ))
    results.append(test_import(
        "from demos.cdss_example.agents.orchestrator_agent import OrchestratorAgent",
        "Orchestrator agent import"
    ))
    
    print("\n[Phase 3: File Structure]")
    results.append(check_file_exists("exaid_core/exaid.py", "exaid_core/exaid.py exists"))
    results.append(check_file_exists("exaid_core/llm.py", "exaid_core/llm.py exists"))
    results.append(check_file_exists("demos/cdss_example/cdss.py", "demos/cdss_example/cdss.py exists"))
    results.append(check_file_exists("demos/backend/server.py", "demos/backend/server.py exists"))
    
    print("\n[Phase 4: sys.path Verification]")
    results.append(check_no_syspath_manipulation())
    
    print("\n[Phase 5: Old Folders Removed]")
    old_exists = (
        Path("agents").exists() or 
        Path("schema").exists() or 
        Path("callbacks").exists() or
        Path("cdss_demo").exists() or
        Path("web_ui").exists()
    )
    if not old_exists:
        print("✓ Old folders removed")
        results.append(True)
    else:
        print("✗ Old folders still exist")
        if Path("agents").exists():
            print("  - agents/")
        if Path("schema").exists():
            print("  - schema/")
        if Path("callbacks").exists():
            print("  - callbacks/")
        if Path("cdss_demo").exists():
            print("  - cdss_demo/")
        if Path("web_ui").exists():
            print("  - web_ui/")
        results.append(False)
    
    print("\n" + "="*60)
    passed = sum(results)
    total = len(results)
    print(f"Results: {passed}/{total} tests passed")
    print("="*60)
    
    return 0 if all(results) else 1

if __name__ == "__main__":
    sys.exit(main())
