"""
Project-wide Python startup customizations.

Python automatically imports ``sitecustomize`` (if present on the import
path) after loading the standard ``site`` module. We use this hook to
polyfill APIs that new dependencies expect but which are missing from the
older toolchain pinned in ``constraints.txt`` (e.g., PyTorch 2.2).
"""

from __future__ import annotations

import enum
import sys
import types

try:
    import torch
    import torch.nn as torch_nn
except Exception:  # pragma: no cover - best effort fall back
    torch = None  # type: ignore[assignment]
    torch_nn = None  # type: ignore[assignment]


def _ensure_torch_compiler_shim() -> None:
    """Guarantee that ``torch.compiler.is_compiling`` exists."""
    if torch is None:
        return

    compiler = getattr(torch, "compiler", None)
    if compiler is None:
        class _CompilerShim:
            @staticmethod
            def is_compiling() -> bool:
                return False

        torch.compiler = _CompilerShim()  # type: ignore[attr-defined]
        return

    if not hasattr(compiler, "is_compiling"):
        def _is_compiling() -> bool:
            return False

        compiler.is_compiling = _is_compiling  # type: ignore[attr-defined]


def _ensure_torch_attention_shim() -> None:
    """Provide a minimal torch.nn.attention API for older PyTorch builds."""
    if torch_nn is None:
        return

    if hasattr(torch_nn, "attention"):
        return

    class _SDPBackend(enum.Enum):
        MATH = enum.auto()
        EFFICIENT_ATTENTION = enum.auto()
        FLASH_ATTENTION = enum.auto()

    class _NoOpContext:
        def __enter__(self):
            return None

        def __exit__(self, exc_type, exc, tb):
            return False

    def _sdpa_kernel(_backends):
        return _NoOpContext()

    attention_mod = types.ModuleType("torch.nn.attention")
    attention_mod.sdpa_kernel = _sdpa_kernel  # type: ignore[attr-defined]
    attention_mod.SDPBackend = _SDPBackend  # type: ignore[attr-defined]
    torch_nn.attention = attention_mod  # type: ignore[attr-defined]
    sys.modules["torch.nn.attention"] = attention_mod


_ensure_torch_compiler_shim()
_ensure_torch_attention_shim()
