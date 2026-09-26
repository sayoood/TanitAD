"""D-REFCV6-EQUALIZE-DROPPED (SPEC_REFCV7 FIX-3) probe: ONE deterministic roll (DDIM draw = 0) of a
pre-fix refcv6 checkpoint through the battery loader, in its own process, on whatever tree
REFCV6_REPO / PYTHONPATH point at.

usage: python eq_as_trained_roll.py --mode {as_trained,skip} <roll_seed.py arguments...>
  as_trained  the loader as shipped (its `trunk_rows_as_trained` override active)
  skip        the REGRESSION ARM: the override replaced by a no-op, i.e. the trunk rebuilt AS DECLARED
Used by test_trunk_equalize_as_trained.py; the verdict lives there.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
torch.randn_like = lambda x, *a, **k: torch.zeros_like(x, *a, **k)      # deterministic: eps = 0

mode = sys.argv[sys.argv.index("--mode") + 1]
del sys.argv[sys.argv.index("--mode"):sys.argv.index("--mode") + 2]
import refcv6_loader as L  # noqa: E402

if mode == "skip":
    L.trunk_rows_as_trained = lambda tr, cfg, config: (cfg, {"status": "SKIPPED by eq_as_trained_roll.py "
                                                                       "(regression arm: trunk as DECLARED)"})
elif mode != "as_trained":
    raise SystemExit(f"--mode {mode}?")
import roll_seed  # noqa: E402

if __name__ == "__main__":
    roll_seed.main()
