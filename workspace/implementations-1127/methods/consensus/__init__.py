"""consensus method package — self-contained core (torch + stdlib only)."""
from .core import (CONSENSUS, CONSENSUS, Out)
__all__ = ["CONSENSUS", "CONSENSUS", "Out", "compute_loss", "setup"]

# compute_loss(ctx) / setup(ctx, cfg) are defined in this method's train.py.
# Kept as re-export targets so `python -m methods.consensus.train` works standalone.
try:
    from .train import compute_loss, setup  # noqa: F401
except Exception:  # HF deps may be absent on CPU-only machines; core still imports
    compute_loss = None
    setup = None
