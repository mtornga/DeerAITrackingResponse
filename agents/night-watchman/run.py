#!/usr/bin/env python3
"""
Night Watchman Agent Runner

Invokes Claude Code in headless mode to execute the night-watchman skill.
Scheduled to run at 1am Central Time daily via cron.

Usage:
    python agents/night-watchman/run.py
    python agents/night-watchman/run.py --dry-run
    python agents/night-watchman/run.py --verbose
    python agents/night-watchman/run.py --timeout 1800 --verbose

Options:
    --dry-run           Print what would be executed without running
    --verbose, -v       Enable verbose debug logging (timing, commands, env vars)
    --timeout SECS      Timeout for Claude Code execution (default: 300 for v2, 3600 for v1)
    --start-time TIME   Override review window start (format: YYYY-MM-DD HH:MM)
    --skip-health-check Skip Agent Mail pre-flight connectivity check
    --skill VERSION     Skill version to use: v1 (full) or v2 (MCP smoke test, default)
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional
from zoneinfo import ZoneInfo

# Global verbose flag (set via --verbose CLI arg)
VERBOSE = False


def log_verbose(msg: str) -> None:
    """Print message only if verbose mode is enabled."""
    if VERBOSE:
        timestamp = datetime.now(TIMEZONE).strftime("%H:%M:%S")
        print(f"[DEBUG {timestamp}] {msg}")

# Configuration
# Agent Mail enforces adjective-noun identities; CalmEagle is the registered Night Watchman handle.
AGENT_NAME = "CalmEagle"
AGENT_MAIL_URL = "http://192.168.68.71:8765/mail"
REVIEW_UI_URL = "http://192.168.68.71:8501/"
THREAD_ID = "NIGHT-WATCHMAN"
TIMEZONE = ZoneInfo("America/Chicago")

# Default review window start: 3pm Central previous day
DEFAULT_REVIEW_START_HOUR = 15


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


def get_review_window_start(agent_mail_url: str) -> datetime:
    """
    Determine the review window start time.

    First checks Agent Mail for Day Watchman's last review.
    Falls back to 3pm Central of the previous day.
    """
    now = datetime.now(TIMEZONE)

    # Try to get Day Watchman's last review from Agent Mail
    day_watchman_timestamp = query_day_watchman_last_review(agent_mail_url)
    if day_watchman_timestamp:
        print(f"Found Day Watchman's last review: {day_watchman_timestamp}")
        return day_watchman_timestamp

    # Default: 3pm Central of the previous day
    yesterday = now - timedelta(days=1)
    default_start = yesterday.replace(
        hour=DEFAULT_REVIEW_START_HOUR,
        minute=0,
        second=0,
        microsecond=0
    )
    print(f"No Day Watchman message found. Using default start: {default_start}")
    return default_start


def query_day_watchman_last_review(agent_mail_url: str) -> Optional[datetime]:
    """
    Query Agent Mail for the Day Watchman's last review timestamp.

    Returns None if no message found or on error.
    """
    try:
        import httpx
    except ImportError:
        print("Warning: httpx not installed, skipping Agent Mail query")
        return None

    try:
        # Search for messages from day-watchman in DAY-WATCHMAN thread
        search_url = f"{agent_mail_url}/search"
        params = {
            "thread_id": "DAY-WATCHMAN",
            "from_agent": "day-watchman",
            "limit": 1,
        }

        with httpx.Client(timeout=10.0) as client:
            response = client.get(search_url, params=params)
            if response.status_code == 200:
                messages = response.json()
                if messages and len(messages) > 0:
                    # Parse timestamp from most recent message
                    msg = messages[0]
                    timestamp_str = msg.get("created_at") or msg.get("timestamp")
                    if timestamp_str:
                        # Parse ISO format timestamp
                        dt = datetime.fromisoformat(timestamp_str.replace("Z", "+00:00"))
                        return dt.astimezone(TIMEZONE)
    except Exception as e:
        print(f"Warning: Could not query Agent Mail: {e}")
        log_verbose(f"Exception type: {type(e).__name__}, details: {e}")

    return None


def check_agent_mail_health(agent_mail_url: str, timeout: float = 5.0) -> bool:
    """
    Pre-flight health check for Agent Mail server.

    Returns True if server is reachable, False otherwise.
    """
    try:
        import httpx
    except ImportError:
        print("Warning: httpx not installed, skipping Agent Mail health check")
        return True  # Proceed anyway if httpx not available

    # Try the base URL or a health endpoint
    base_url = agent_mail_url.rstrip("/mail").rstrip("/")
    health_url = f"{base_url}/health"

    log_verbose(f"Checking Agent Mail health at {health_url}")

    try:
        with httpx.Client(timeout=timeout) as client:
            # Try health endpoint first
            try:
                response = client.get(health_url)
                log_verbose(f"Health endpoint response: {response.status_code}")
                if response.status_code == 200:
                    return True
            except httpx.HTTPError:
                pass

            # Fall back to base mail endpoint
            log_verbose(f"Trying base endpoint: {agent_mail_url}")
            response = client.get(agent_mail_url)
            log_verbose(f"Base endpoint response: {response.status_code}")
            return response.status_code in (200, 404, 405)  # 404/405 means server is up

    except httpx.ConnectError as e:
        print(f"ERROR: Cannot connect to Agent Mail at {base_url}")
        log_verbose(f"Connection error: {e}")
        return False
    except httpx.TimeoutException as e:
        print(f"ERROR: Agent Mail at {base_url} timed out after {timeout}s")
        log_verbose(f"Timeout error: {e}")
        return False
    except Exception as e:
        print(f"ERROR: Agent Mail health check failed: {e}")
        log_verbose(f"Exception type: {type(e).__name__}")
        return False


def build_prompt(review_start: datetime, review_end: datetime, skill_version: str = "v2") -> str:
    """Build the prompt for Claude Code."""
    repo_root = get_repo_root()

    if skill_version == "v2":
        # Simplified MCP smoke test
        skill_path = repo_root / "agents" / "night-watchman" / "SKILL_v2.md"
        return f"""Execute the night-watchman-v2 skill (MCP smoke test).

**Your Identity**: CalmEagle
**Project Key**: {repo_root}
**Current Time**: {review_end.strftime('%Y-%m-%d %H:%M')} CT

**Instructions**: Read and follow the instructions in {skill_path}

Complete all 4 steps: register, send message, check inbox, acknowledge."""
    else:
        # Full v1 skill
        skill_path = repo_root / "agents" / "night-watchman" / "SKILL.md"
        return f"""Execute the night-watchman skill.

**Review Window**: {review_start.strftime('%Y-%m-%d %H:%M')} CT to {review_end.strftime('%Y-%m-%d %H:%M')} CT

**Agent Configuration**:
- Agent Mail URL: {AGENT_MAIL_URL}
- Review UI URL: {REVIEW_UI_URL}
- Thread ID: {THREAD_ID}
- Report Output: runs/logs/watchman/{review_end.strftime('%Y-%m-%d')}.md

**Instructions**: Read and follow the instructions in {skill_path}

Begin your patrol now. Start with the system health check, then pipeline health, then review detections from the review window."""


def run_claude_code(
    prompt: str,
    dry_run: bool = False,
    timeout_seconds: int = 3600,
    skill_version: str = "v2"
) -> int:
    """
    Invoke Claude Code in headless mode.

    Args:
        prompt: The prompt to send to Claude Code
        dry_run: If True, print command without executing
        timeout_seconds: Maximum time to wait for Claude Code (default: 1 hour)
        skill_version: Which skill version to run (v1 or v2)

    Returns the exit code from Claude Code.
    """
    repo_root = get_repo_root()
    claude_binary = discover_claude_binary()
    if not claude_binary:
        print("ERROR: Could not locate the 'claude' CLI. Install it or set CLAUDE_BIN to its path.")
        return 1

    log_verbose(f"Found claude binary: {claude_binary}")

    # Different allowed tools for different skill versions
    if skill_version == "v2":
        # v2 skill - MCP Agent Mail + Bash for system checks + Write for digest
        allowed_tools = "Read,Write,Bash,mcp__mcp-agent-mail__*"
    else:
        # Full v1 skill needs more tools
        allowed_tools = "Read,Glob,Grep,Bash,WebFetch"

    log_verbose(f"Skill version: {skill_version}, allowed tools: {allowed_tools}")

    cmd = [
        claude_binary,
        "-p", prompt,
        "--allowedTools", allowed_tools,
        "--output-format", "text",
    ]

    env = os.environ.copy()
    env_additions = {
        "AGENT_MAIL_PROJECT": str(repo_root),
        "AGENT_MAIL_AGENT": AGENT_NAME,
        "AGENT_MAIL_URL": AGENT_MAIL_URL,
        "REVIEW_UI_URL": REVIEW_UI_URL,
    }
    env.update(env_additions)

    if dry_run:
        print("DRY RUN - would execute:")
        print(f"  Claude CLI: {claude_binary}")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Working directory: {repo_root}")
        print(f"  Timeout: {timeout_seconds}s")
        print(f"  Environment additions:")
        for key, value in env_additions.items():
            print(f"    {key}={value}")
        print(f"\nPrompt:\n{prompt}")
        return 0

    print(f"Starting Night Watchman at {datetime.now(TIMEZONE)}")
    print(f"Working directory: {repo_root}")
    print(f"Timeout: {timeout_seconds}s ({timeout_seconds // 60} minutes)")
    print("-" * 60)

    # Verbose logging of full command and environment
    log_verbose(f"Full command: {' '.join(cmd)}")
    log_verbose(f"Environment additions:")
    for key, value in env_additions.items():
        log_verbose(f"  {key}={value}")

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
        print("The process was killed. Check if Agent Mail or MCP connections are hanging.")
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


def send_completion_notification(
    agent_mail_url: str,
    success: bool,
    review_window: str,
    digest_path: Path,
) -> None:
    """Send a completion notification to Agent Mail."""
    try:
        import httpx
    except ImportError:
        print("Warning: httpx not installed, skipping completion notification")
        return

    status = "COMPLETED" if success else "FAILED"
    message = {
        "from_agent": AGENT_NAME,
        "thread_id": THREAD_ID,
        "subject": f"[{status}] Night Watchman Report - {datetime.now(TIMEZONE).strftime('%Y-%m-%d')}",
        "body": f"""Night Watchman patrol {status.lower()}.

Review Window: {review_window}
Digest: {digest_path}

{'See digest for full report.' if success else 'Check logs for errors.'}""",
        "priority": "normal" if success else "high",
    }

    try:
        send_url = f"{agent_mail_url}/send"
        with httpx.Client(timeout=10.0) as client:
            response = client.post(send_url, json=message)
            if response.status_code in (200, 201):
                print(f"Sent completion notification to Agent Mail")
            else:
                print(f"Warning: Agent Mail returned {response.status_code}")
    except Exception as e:
        print(f"Warning: Could not send completion notification: {e}")


def main():
    global VERBOSE

    parser = argparse.ArgumentParser(description="Night Watchman Agent Runner")
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
        "--start-time",
        type=str,
        help="Override review window start (format: YYYY-MM-DD HH:MM)",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Timeout in seconds for Claude Code execution (default: 3600)",
    )
    parser.add_argument(
        "--skip-health-check",
        action="store_true",
        help="Skip Agent Mail pre-flight health check",
    )
    parser.add_argument(
        "--skill",
        type=str,
        choices=["v1", "v2"],
        default="v2",
        help="Skill version: v1 (full patrol) or v2 (MCP smoke test, default)",
    )
    args = parser.parse_args()

    # Set default timeout based on skill version if not explicitly specified
    if args.timeout == 3600 and args.skill == "v2":
        args.timeout = 300  # 5 minutes for v2 smoke test

    # Set global verbose flag
    VERBOSE = args.verbose
    if VERBOSE:
        print("[DEBUG] Verbose logging enabled")

    # Pre-flight health check for Agent Mail
    if not args.skip_health_check and not args.dry_run:
        print("Checking Agent Mail connectivity...")
        log_verbose(f"Agent Mail URL: {AGENT_MAIL_URL}")
        if not check_agent_mail_health(AGENT_MAIL_URL):
            print("=" * 60)
            print("FATAL: Agent Mail is not reachable. Cannot proceed.")
            print("Options:")
            print("  1. Start the Agent Mail server on 192.168.68.71:8765")
            print("  2. Run with --skip-health-check to bypass this check")
            print("=" * 60)
            sys.exit(1)
        print("Agent Mail: OK")
        log_verbose("Agent Mail health check passed")

    # Determine review window
    now = datetime.now(TIMEZONE)

    if args.start_time:
        review_start = datetime.strptime(args.start_time, "%Y-%m-%d %H:%M")
        review_start = review_start.replace(tzinfo=TIMEZONE)
        print(f"Using override start time: {review_start}")
    else:
        review_start = get_review_window_start(AGENT_MAIL_URL)

    review_end = now
    review_window_str = f"{review_start.strftime('%Y-%m-%d %H:%M')} to {review_end.strftime('%Y-%m-%d %H:%M')} CT"

    print(f"Night Watchman Agent Runner (skill: {args.skill})")
    print(f"Review window: {review_window_str}")
    print("=" * 60)

    # Build prompt and run
    prompt = build_prompt(review_start, review_end, skill_version=args.skill)
    log_verbose(f"Prompt length: {len(prompt)} chars")
    exit_code = run_claude_code(
        prompt,
        dry_run=args.dry_run,
        timeout_seconds=args.timeout,
        skill_version=args.skill
    )

    if not args.dry_run:
        # Send completion notification
        repo_root = get_repo_root()
        digest_path = repo_root / "runs" / "logs" / "watchman" / f"{now.strftime('%Y-%m-%d')}.md"
        send_completion_notification(
            AGENT_MAIL_URL,
            success=(exit_code == 0),
            review_window=review_window_str,
            digest_path=digest_path,
        )

    print("=" * 60)
    print(f"Night Watchman finished with exit code {exit_code}")

    sys.exit(exit_code)


if __name__ == "__main__":
    main()
