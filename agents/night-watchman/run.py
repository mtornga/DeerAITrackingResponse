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
    --timeout SECS  Timeout for Claude Code execution (default: 420)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo
import urllib.request
import urllib.error

# Global verbose flag (set via --verbose CLI arg)
VERBOSE = False

# Configuration
AGENT_NAME = "CalmEagle"
TIMEZONE = ZoneInfo("America/Chicago")
DEFAULT_TIMEOUT = 420  # 7 minutes


def log(msg: str, level: str = "INFO") -> None:
    """Always log with timestamp."""
    timestamp = datetime.now(TIMEZONE).strftime("%H:%M:%S.%f")[:-3]
    print(f"[{level} {timestamp}] {msg}", flush=True)


def log_verbose(msg: str) -> None:
    """Print message only if verbose mode is enabled."""
    if VERBOSE:
        log(msg, "DEBUG")


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

    # Check npx cache (for npx @anthropic-ai/claude-code installations)
    npx_cache = Path.home() / ".npm" / "_npx"
    if npx_cache.exists():
        for cli_path in npx_cache.glob("*/node_modules/.bin/claude"):
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


def check_mcp_server() -> bool:
    """Test if MCP Agent Mail server is responding."""
    try:
        req = urllib.request.Request(
            "http://127.0.0.1:8765/mcp/",
            method="POST",
            headers={"Content-Type": "application/json"},
            data=b'{"jsonrpc":"2.0","method":"ping","id":1}'
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            log_verbose(f"MCP server responded: {resp.status}")
            return resp.status == 200
    except urllib.error.URLError as e:
        log(f"MCP server check failed: {e}", "WARN")
        return False
    except Exception as e:
        log(f"MCP server check error: {e}", "WARN")
        return False


def check_claude_settings(repo_root: Path) -> dict:
    """Read and report Claude settings files."""
    settings = {}

    # User-level settings
    user_settings = Path.home() / ".claude" / "settings.json"
    if user_settings.exists():
        try:
            settings["user"] = json.loads(user_settings.read_text())
            mcp_servers = settings["user"].get("mcpServers", {})
            log_verbose(f"User MCP servers: {list(mcp_servers.keys())}")
        except Exception as e:
            log(f"Failed to read user settings: {e}", "WARN")

    # Project-level settings
    project_settings = repo_root / ".claude" / "settings.local.json"
    if project_settings.exists():
        try:
            settings["project"] = json.loads(project_settings.read_text())
            enabled = settings["project"].get("enabledMcpjsonServers", [])
            if enabled:
                log(f"WARNING: Project has enabledMcpjsonServers: {enabled}", "WARN")
        except Exception as e:
            log(f"Failed to read project settings: {e}", "WARN")

    return settings


def test_claude_ping(claude_binary: str, repo_root: Path, timeout: int = 30) -> bool:
    """Quick test that Claude can respond at all."""
    log("Testing Claude API connectivity...")
    try:
        result = subprocess.run(
            [claude_binary, "-p", "Reply with just: PONG", "--output-format", "text"],
            cwd=repo_root,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if "PONG" in result.stdout or result.returncode == 0:
            log(f"Claude ping OK (took <{timeout}s)")
            return True
        else:
            log(f"Claude ping unexpected response: {result.stdout[:100]}", "WARN")
            return True  # Still consider it working if we got a response
    except subprocess.TimeoutExpired:
        log(f"Claude ping TIMEOUT after {timeout}s - API may be slow", "ERROR")
        return False
    except Exception as e:
        log(f"Claude ping failed: {e}", "ERROR")
        return False


def run_claude_code(prompt: str, dry_run: bool, timeout_seconds: int) -> int:
    """
    Invoke Claude Code in headless mode with MCP Agent Mail tools.
    Uses Popen for real-time output capture.

    Returns the exit code from Claude Code.
    """
    repo_root = get_repo_root()

    # Pre-flight checks
    log("=" * 60)
    log("PRE-FLIGHT CHECKS")
    log("=" * 60)

    # Check Claude binary
    claude_binary = discover_claude_binary()
    if not claude_binary:
        log("ERROR: Could not locate the 'claude' CLI", "ERROR")
        return 1
    log(f"Claude binary: {claude_binary}")

    # Check settings for problematic MCP servers
    settings = check_claude_settings(repo_root)

    # Check MCP server
    mcp_ok = check_mcp_server()
    log(f"MCP Agent Mail server: {'OK' if mcp_ok else 'NOT RESPONDING'}")

    # Quick Claude ping test
    if not test_claude_ping(claude_binary, repo_root):
        log("Claude API not responding - aborting", "ERROR")
        return 1

    log("=" * 60)
    log("STARTING NIGHT WATCHMAN")
    log("=" * 60)

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

    log(f"Working directory: {repo_root}")
    log(f"Timeout: {timeout_seconds}s ({timeout_seconds // 60} minutes)")
    log_verbose(f"Full command: {' '.join(cmd)}")

    start_time = time.time()
    output_lines = []
    last_output_time = start_time

    # Use Popen for real-time output
    process = subprocess.Popen(
        cmd,
        cwd=repo_root,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,  # Line buffered
    )

    def read_output():
        """Read output in a separate thread."""
        nonlocal last_output_time
        for line in iter(process.stdout.readline, ''):
            if line:
                output_lines.append(line)
                last_output_time = time.time()
                print(line, end='', flush=True)
        process.stdout.close()

    # Start output reader thread
    reader_thread = threading.Thread(target=read_output, daemon=True)
    reader_thread.start()

    # Monitor with heartbeat
    heartbeat_interval = 30  # Log every 30 seconds
    last_heartbeat = start_time

    try:
        while True:
            # Check if process finished
            ret = process.poll()
            if ret is not None:
                reader_thread.join(timeout=1)
                elapsed = time.time() - start_time
                log(f"Process completed in {elapsed:.1f}s with exit code {ret}")
                return ret

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                log(f"TIMEOUT after {elapsed:.1f}s - killing process", "ERROR")
                process.kill()
                reader_thread.join(timeout=1)

                # Diagnostic dump
                log("=" * 60)
                log("TIMEOUT DIAGNOSTICS")
                log("=" * 60)
                log(f"Total output lines received: {len(output_lines)}")
                log(f"Time since last output: {time.time() - last_output_time:.1f}s")
                if output_lines:
                    log("Last 10 lines of output:")
                    for line in output_lines[-10:]:
                        print(f"  | {line}", end='')
                else:
                    log("NO OUTPUT WAS RECEIVED - Claude may have hung during initialization")

                return 124

            # Heartbeat logging
            if time.time() - last_heartbeat > heartbeat_interval:
                last_heartbeat = time.time()
                silence = time.time() - last_output_time
                log(f"Heartbeat: {elapsed:.0f}s elapsed, {len(output_lines)} lines, {silence:.0f}s since last output")

            time.sleep(0.5)

    except KeyboardInterrupt:
        log("Interrupted by user", "WARN")
        process.kill()
        return 130
    except Exception as e:
        elapsed = time.time() - start_time
        log(f"Exception after {elapsed:.1f}s: {e}", "ERROR")
        process.kill()
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
    parser.add_argument(
        "--skip-ping",
        action="store_true",
        help="Skip the Claude API ping test",
    )
    args = parser.parse_args()

    VERBOSE = args.verbose

    log("=" * 60)
    log("NIGHT WATCHMAN LAUNCHER v2.10")
    log("=" * 60)

    if VERBOSE:
        log("Verbose logging enabled", "DEBUG")

    prompt = build_prompt()
    log_verbose(f"Prompt length: {len(prompt)} chars")

    exit_code = run_claude_code(prompt, args.dry_run, args.timeout)

    log("=" * 60)
    log(f"Night Watchman finished with exit code {exit_code}")
    log("=" * 60)

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
