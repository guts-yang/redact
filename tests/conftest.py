"""Session-wide test setup.

``CUBLAS_WORKSPACE_CONFIG`` has to be set *before* CUDA initialises, otherwise
``torch.use_deterministic_algorithms(True)`` cannot be satisfied on CUDA >= 10.2
and every subsequent cuBLAS GEMM raises. ``conftest.py`` is imported before any
test module, which makes this the earliest available hook.
"""
import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

import pytest
import torch


@pytest.fixture(autouse=True)
def _restore_deterministic_algorithms():
    """Stop one test's global determinism switch leaking into the rest.

    ``torch.use_deterministic_algorithms`` is process-global and is not reset by
    the helpers that set it, so without this a single determinism test poisons
    every later test that touches a non-deterministic kernel.
    """
    was_enabled = torch.are_deterministic_algorithms_enabled()
    warn_only = getattr(torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False)()
    try:
        yield
    finally:
        torch.use_deterministic_algorithms(was_enabled, warn_only=warn_only)
