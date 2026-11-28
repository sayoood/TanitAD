"""SPEC A4 lever L3: ONE refcv6 roll with the anchored-Gaussian draw set to ZERO (deterministic DDIM).

Same arguments as roll_seed.py. `torch.randn_like` returns zeros for the whole process; on refcv6's
eval path its only caller is the anchored-Gaussian draw (`refc.py:2551`; the other four sites in
stack/tanitad are training-only or other models, `refc.py:3065` is zeros_like outside training).
The A2 wrapper probe used the same patch. Runs in its own process, like roll_seed.py.
"""
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

_CALLS = {"n": 0}


def _zeros_like(x, *a, **k):
    _CALLS["n"] += 1
    return torch.zeros_like(x, *a, **k)


torch.randn_like = _zeros_like

import roll_seed  # noqa: E402  (imports the loader; the patch is already in place)


def main():
    import json
    roll_seed.main()
    print(f"[roll_eps0] torch.randn_like calls answered with zeros: {_CALLS['n']}")
    out = sys.argv[sys.argv.index("--out-json") + 1]
    rec = json.load(open(out, encoding="utf-8"))
    rec["lever"] = "L3 (SPEC A4): anchored-Gaussian draw = 0"
    rec["eps0_randn_like_calls"] = _CALLS["n"]
    json.dump(rec, open(out, "w", encoding="utf-8"), indent=1, default=str)
    if _CALLS["n"] == 0:
        raise SystemExit("[roll_eps0] the patched draw was never called -- L3 would be a no-op; refusing")


if __name__ == "__main__":
    main()
