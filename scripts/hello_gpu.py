#!/usr/bin/env python3
from __future__ import annotations

import sys
from typing import NoReturn


def main() -> int:
    try:
        import torch  # type: ignore[import]
    except Exception as exc:  # pragma: no cover - informational script
        print("hello_gpu: unable to import torch.", file=sys.stderr)
        print(f"  error: {exc}", file=sys.stderr)
        return 1

    cuda_available = torch.cuda.is_available()
    print(f"hello_gpu: torch version: {torch.__version__}")
    print(f"hello_gpu: cuda_available={cuda_available}")

    if cuda_available:
        device = torch.device("cuda:0")
        name = torch.cuda.get_device_name(device)
        capability = torch.cuda.get_device_capability(device)
        print(f"hello_gpu: device={name}, capability={capability[0]}.{capability[1]}")
    else:
        print("hello_gpu: CUDA not available; using CPU only.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

