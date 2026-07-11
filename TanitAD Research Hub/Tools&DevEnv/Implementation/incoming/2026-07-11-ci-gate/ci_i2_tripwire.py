"""I2 batch-consistency tripwire — the CI gate's fast deployment invariant (D-004).

Why this exists
---------------
Deployment is **batch-1 streaming** on Orin. Any batch-statistic layer (BatchNorm,
a stray `x - x.mean(0)`, a batched running-stat) is *silent* in training — the loss
still falls — but makes `encode(frame, batch=1) != encode(frame, batch=B)` at
inference, so the ONNX/TensorRT engine returns different latents than the trained
model. I2 (`i2_batch_consistency`) is the instrument that catches this class forever
(see `tanitad/instruments/checks.py`). `test_instruments.py` already asserts it, but
the full pytest suite is the slow part of a commit; this module is a **~2 s standalone
front-gate** so a batch-statistic regression fails *before* the suite even runs, with a
crisp message, and can be wired into `ci.ps1` / pre-commit without importing pytest.

The core check is factored to take any `encode_fn` so the tripwire *logic* is unit
-testable (a synthetic batch-mean-subtracting encoder must fail it) without building a
broken `WorldModel`.

Usage
-----
    python ci_i2_tripwire.py --stack-dir stack           # -> exit 0/1, prints I2 dev
    python ci_i2_tripwire.py --stack-dir stack --tol 1e-4 --batch 8
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _ensure_stack_on_path(stack_dir: str) -> None:
    """Put the stack root on sys.path so `import tanitad` works from anywhere."""
    p = str(Path(stack_dir).resolve())
    if p not in sys.path:
        sys.path.insert(0, p)


def check_encode_fn(encode_fn, frames, tol: float = 1e-4) -> tuple[bool, float]:
    """Run I2 on an arbitrary encode_fn ([B,...] -> [B,S]). Pure wrapper over the
    stack instrument so the tripwire shares ONE definition of the invariant."""
    import torch  # local import: keep module import cheap / torch-optional for --help

    from tanitad.instruments.checks import i2_batch_consistency

    with torch.no_grad():
        ok, dev = i2_batch_consistency(encode_fn, frames, tol=tol)
    return bool(ok), float(dev)


def run_i2_tripwire(tol: float = 1e-4, batch: int = 8, seed: int = 0
                    ) -> tuple[bool, float]:
    """Build the real WorldModel (smoke config) and check its encoder is batch-1
    consistent. Smoke config keeps this to ~1-2 s on CPU."""
    import torch

    from tanitad.config import smoke_config
    from tanitad.models.fourbrain import WorldModel

    torch.manual_seed(seed)
    model = WorldModel(smoke_config()).eval()
    frames = torch.rand(batch, 1, 64, 64)
    return check_encode_fn(model.encode, frames, tol=tol)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--stack-dir", default="stack",
                    help="repo stack/ dir (must contain the tanitad package)")
    ap.add_argument("--tol", type=float, default=1e-4)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args(argv)

    _ensure_stack_on_path(args.stack_dir)
    try:
        ok, dev = run_i2_tripwire(tol=args.tol, batch=args.batch, seed=args.seed)
    except Exception as exc:  # import/instantiation failure is itself a gate failure
        print(f"I2 TRIPWIRE ERROR: {type(exc).__name__}: {exc}")
        return 2
    if ok:
        print(f"I2 TRIPWIRE OK: encoder batch-1 consistent (dev={dev:.2e} < tol {args.tol:.0e})")
        return 0
    print(f"I2 TRIPWIRE FAILED: encoder violates batch-1 consistency "
          f"(dev={dev:.2e} >= tol {args.tol:.0e}) — a batch-statistic layer is in "
          f"the inference path; deployment is batch-1 streaming (Orin).")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
