"""faith method package — self-contained core (torch + stdlib only)."""
from .core import (FAITH, FAITH, Out)
__all__ = ["FAITH", "FAITH", "Out", "compute_loss", "setup"]

# compute_loss(ctx) / setup(ctx, cfg) are defined in this method's train.py.
# Kept as re-export targets so `python -m methods.faith.train` works standalone.
try:
    from .train import compute_loss, setup  # noqa: F401
except Exception:  # HF deps may be absent on CPU-only machines; core still imports
    compute_loss = None
    setup = None
