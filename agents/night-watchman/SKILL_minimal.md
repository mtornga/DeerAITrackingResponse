---
name: night-watchman-minimal
description: "Minimal patrol - just health check"
allowed-tools: "Bash,Write"
---

# Night Watchman MINIMAL Test

You are CalmEagle. Complete these 2 steps:

## Step 1: Health Check

Run this ONE command:

```bash
echo "=== DISK ===" && df -h / && echo "=== DONE ==="
```

Report: OK if >20% free, WARNING if 10-20%, CRITICAL if <10%.

## Step 2: Write Result

Create file `/tmp/watchman_test.txt` with contents:
```
Patrol complete
Disk: [OK/WARNING/CRITICAL]
Time: [current time]
```

Done. Print "MINIMAL PATROL COMPLETE".
