#!/usr/bin/env python3
"""
Night Watchman Agent Launcher

Invokes Claude Code in headless mode to execute the night-watchman skill.
Scheduled to run at 1am Central Time daily via cron.

Usage:
    python agents/night-watchman/run.py
    python agents/night-watchman/run.py --dry-run
    python agents/night-watchman/run.py --verbose
    python agents/night-watchman/run.py --timeout 600

Options:
    --dry-run       Print what would be executed without running
    --verbose, -v   Enable verbose debug logging
    --timeout SECS  Timeout for Claude Code execution (default: 300)
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

# Global verbose flag (set via --verbose CLI arg)
VERBOSE = False

# Configuration
AGENT_NAME = "CalmEagle"
TIMEZONE = ZoneInfo("America/Chicago")
DEFAULT_TIMEOUT = 420  # 7 minutes


def log_verbose(msg: str) -> None:
    """Print message only if verbose mode is enabled."""
    if VERBOSE:
        timestamp = datetime.now(TIMEZONE).strftime("%H:%M:%S")
        print(f"[DEBUG {timestamp}] {msg}")


def _resolve_command_path(candidate: str) -> Optional[str]:
    """Return an executable path if candidate resolves."""
    if not candidate:
        return None
    expanded = os.path.expanduser(candidate)
    candidate_path = Path(expanded)
    if candidate_path.is_file() and os.access(candidate_path, os.X_OK):
        return str(candidate_path)
    return shutil.which(expanded)


def discover_claude_binary() -> Optional[str]:
    """
    Find the claude CLI, preferring CLAUDE_BIN env, then PATH, then ~/.nvm installs.
    """
    candidates: list[str] = []
    env_candidate = os.environ.get("CLAUDE_BIN")
    if env_candidate:
        candidates.append(env_candidate)
    candidates.append("claude")

    nvm_root = Path.home() / ".nvm" / "versions" / "node"
    if nvm_root.exists():
        for cli_path in sorted(nvm_root.glob("*/bin/claude"), reverse=True):
            candidates.append(str(cli_path))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved = _resolve_command_path(candidate)
        if resolved:
            return resolved
    return None


def get_repo_root() -> Path:
    """Get the repository root directory."""
    script_path = Path(__file__).resolve()
    # agents/night-watchman/run.py -> repo root is 2 levels up
    return script_path.parents[2]


def build_prompt() -> str:
    """Build the prompt for SKILL_v2.md."""
    repo_root = get_repo_root()
    skill_path = repo_root / "agents" / "night-watchman" / "SKILL_v2.md"
    now = datetime.now(TIMEZONE)

    return f"""Execute the night-watchman skill.

**Your Identity**: {AGENT_NAME}
**Project Key**: {repo_root}
**Current Time**: {now.strftime('%Y-%m-%d %H:%M')} CT

**Instructions**: Read and follow the instructions in {skill_path}

Complete all 6 steps. Send Agent Mail progress messages at each step start. Use batched bash/jq commands."""


def run_claude_code(prompt: str, dry_run: bool, timeout_seconds: int) -> int:
    """
    Invoke Claude Code in headless mode with MCP Agent Mail tools.

    Returns the exit code from Claude Code.
    """
    repo_root = get_repo_root()
    claude_binary = discover_claude_binary()
    if not claude_binary:
        print("ERROR: Could not locate the 'claude' CLI. Install it or set CLAUDE_BIN to its path.")
        return 1

    log_verbose(f"Found claude binary: {claude_binary}")

    cmd = [
        claude_binary,
        "-p", prompt,
        "--allowedTools", "Read,Write,Bash,mcp__mcp-agent-mail__*",
        "--output-format", "text",
    ]

    env = os.environ.copy()
    env["AGENT_MAIL_PROJECT"] = str(repo_root)
    env["AGENT_MAIL_AGENT"] = AGENT_NAME

    if dry_run:
        print("DRY RUN - would execute:")
        print(f"  Claude CLI: {claude_binary}")
        print(f"  Working directory: {repo_root}")
        print(f"  Timeout: {timeout_seconds}s")
        print(f"  Environment: AGENT_MAIL_PROJECT={repo_root}, AGENT_MAIL_AGENT={AGENT_NAME}")
        print(f"\nPrompt:\n{prompt}")
        return 0

    print(f"Night Watchman starting at {datetime.now(TIMEZONE)}")
    print(f"Working directory: {repo_root}")
    print(f"Timeout: {timeout_seconds}s ({timeout_seconds // 60} minutes)")
    print("-" * 60)

    log_verbose(f"Full command: {' '.join(cmd)}")

    start_time = time.time()
    log_verbose(f"Subprocess starting at {datetime.now(TIMEZONE).isoformat()}")

    try:
        result = subprocess.run(
            cmd,
            cwd=repo_root,
            env=env,
            capture_output=False,  # Let output stream to console
            text=True,
            timeout=timeout_seconds,
        )
        elapsed = time.time() - start_time
        log_verbose(f"Subprocess completed in {elapsed:.1f}s with exit code {result.returncode}")
        print(f"\nClaude Code exited with code {result.returncode} after {elapsed:.1f}s")
        return result.returncode
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start_time
        print(f"\nERROR: Claude Code timed out after {elapsed:.1f}s (limit: {timeout_seconds}s)")
        log_verbose("TimeoutExpired exception caught")
        return 124  # Standard timeout exit code
    except FileNotFoundError:
        print("ERROR: 'claude' command not found. Is Claude Code installed?")
        log_verbose(f"FileNotFoundError for command: {cmd[0]}")
        return 1
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"ERROR: Failed to run Claude Code after {elapsed:.1f}s: {e}")
        log_verbose(f"Exception type: {type(e).__name__}, details: {e}")
        return 1


def main():
    global VERBOSE

    parser = argparse.ArgumentParser(description="Night Watchman Agent Launcher")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print what would be executed without running",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose debug logging",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_TIMEOUT,
        help=f"Timeout in seconds for Claude Code execution (default: {DEFAULT_TIMEOUT})",
    )
    args = parser.parse_args()

    VERBOSE = args.verbose
    if VERBOSE:
        print("[DEBUG] Verbose logging enabled")

    prompt = build_prompt()
    log_verbose(f"Prompt length: {len(prompt)} chars")

    exit_code = run_claude_code(prompt, args.dry_run, args.timeout)

    print("=" * 60)
    print(f"Night Watchman finished with exit code {exit_code}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
