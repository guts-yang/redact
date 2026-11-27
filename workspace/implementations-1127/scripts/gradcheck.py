"""Check that every method's loss is sensitive to its inputs (no dead gradient)."""
import sys
import torch
sys.path.insert(0, "/Users/chenyliao/Repositories/Machine_Unlearning/redact/workspace/implementations-1127")
from scripts.selftest import make_ctx, METHODS
import importlib

for slug in METHODS:
    m = importlib.import_module("methods." + slug + ".train")
    c = make_ctx()
    if m.setup:
        try:
            m.setup(c, None)
        except Exception as e:
            print(slug, "SETUP ERR", e); continue
    out = m.compute_loss(c)
    g = torch.autograd.grad(out.loss, c.logits, retain_graph=True, allow_unused=True)[0]
    mx = float(g.abs().max()) if g is not None else 0.0
    c2 = make_ctx()
    if m.setup:
        try:
            m.setup(c2, None)
        except Exception:
            pass
    with torch.no_grad():
        c2.logits.add_(0.5)
    out2 = m.compute_loss(c2)
    delta = abs(float(out2.loss) - float(out.loss))
    status = "OK" if (mx > 0 or delta > 1e-9) else "DEAD"
    print(f"{slug:10s} loss={float(out.loss):9.5f} maxgrad_logits={mx:.3e} "
          f"loss_delta_on_perturb={delta:.3e} {status}")
