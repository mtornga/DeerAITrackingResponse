#!/usr/bin/env python3
"""Minimal test runner to debug timeouts."""

import os
import shutil
import subprocess
import sys
from pathlib import Path

def find_claude():
    """Find claude binary."""
    # Check nvm
    nvm_root = Path.home() / ".nvm" / "versions" / "node"
    if nvm_root.exists():
        for cli_path in sorted(nvm_root.glob("*/bin/claude"), reverse=True):
            if cli_path.is_file():
                return str(cli_path)
    # Check PATH
    return shutil.which("claude")

def main():
    claude = find_claude()
    if not claude:
        print("ERROR: claude not found")
        sys.exit(1)

    print(f"Using claude: {claude}")

    repo = Path(__file__).resolve().parents[2]
    skill = repo / "agents" / "night-watchman" / "SKILL_minimal.md"

    prompt = f"Read {skill} and execute it."

    cmd = [
        claude,
        "-p", prompt,
        "--allowedTools", "Read,Bash,Write",
        "--output-format", "text",
    ]

    print(f"Running: {' '.join(cmd[:4])}...")
    print("-" * 40)

    result = subprocess.run(cmd, cwd=repo, timeout=120)

    print("-" * 40)
    print(f"Exit code: {result.returncode}")

if __name__ == "__main__":
    main()
