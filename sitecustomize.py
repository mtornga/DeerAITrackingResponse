"""
Project-wide Python startup customizations.

Python automatically imports ``sitecustomize`` (if present on the import
path) after loading the standard ``site`` module. We use this hook to
polyfill APIs that new dependencies expect but which are missing from the
older toolchain pinned in ``constraints.txt`` (e.g., PyTorch 2.2).
"""

from __future__ import annotations

try:
    import torch
except Exception:  # pragma: no cover - best effort fall back
    torch = None  # type: ignore[assignment]


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


_ensure_torch_compiler_shim()

