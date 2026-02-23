# AI Agent Coding Guidelines & Safety Protocol

To ensure the privacy and security of the user and the integrity of this public repository, all AI agents MUST strictly adhere to the following rules regarding file paths and personal data.

## 1. Zero Absolute Path Policy (NO DOXING)
*   **NEVER** include the user's absolute file system paths in any code, scripts, or documentation.
*   **NEVER** hardcode paths starting with `C:\Users\...`, `/Users/...`, or any directory outside the repository root.
*   **IDENTIFY THE ROOT**: The project root is `stk-mat2011`. All paths MUST be relative to this root or dynamically derived from the script's location.

## 2. Dynamic Path Resolution
Always use standard library tools to resolve paths relative to the script or the project root.

**Example (Python):**
```python
from pathlib import Path

# Correct: Resolve path relative to the script itself
SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]  # Adjust based on depth
DATA_DIR = PROJECT_ROOT / "code" / "data" / "processed"
```

## 3. Script Cleanup
*   Before finalizing any task, scan all newly created or modified scripts for accidental leaks of the local environment (usernames, local IPs, absolute paths).
*   Remove any temporary diagnostic scripts (`check_data.py`, `test_alignment.py`, etc.) unless specifically asked to keep them as part of the core codebase.

## 4. Public Repository Awareness
Assume this repository is public. Any information you write to a file (README, scripts, logs) might be visible to the world. Protect the user's anonymity and system configuration.

## 5. Non-Interactive Execution
When writing automation or processing scripts, ensure they can run in a headless environment without requiring hardcoded local paths that would fail on another machine.

---
**Violation of these rules is a critical security failure.**
