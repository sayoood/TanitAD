#!/usr/bin/env python3
"""Offline check of a ULP reference BEFORE any scorer uses it (any numpy python).

For every token in the dev-box reference: (1) its hex values must RE-HASH to the seam fingerprint
(else it is not the exported input and the agent would refuse it anyway); (2) the ulp distance of
the Thor AgentInput (``make_ulp_reference.py`` run on Thor) to it, per field. Reports the
distribution, so the bound handed to the agent is read off data -- and states that it was.

    python ulp_reference_check.py --ref ref_devbox.json --thor agentinput_thor.json --seam seam.npz --out x.json
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import ulp_guard as U                                                  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--thor", required=True)
    ap.add_argument("--seam", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    ref = json.load(open(a.ref, encoding="utf-8"))["tokens"]
    thor = json.load(open(a.thor, encoding="utf-8"))["tokens"]
    z = np.load(a.seam, allow_pickle=False)
    sfp = {str(t): str(f) for t, f in zip(z["token"], z["fingerprint"])}
    rehash_ok, rehash_bad, per_tok, per_field = 0, [], {}, {k: 0 for k in U.FIELDS}
    max_abs = 0.0
    for t, r in ref.items():
        if U.fingerprint_from_hex(r) == sfp.get(t):
            rehash_ok += 1
        else:
            rehash_bad.append(t)
        th = thor.get(t)
        if th is None:
            continue
        worst = 0
        for k in U.FIELDS:
            for ra, ta in zip(r[k], th[k]):
                for x, y in zip(ra, ta):
                    fx, fy = float.fromhex(x), float.fromhex(y)
                    d = U.ulp_distance(fx, fy)
                    if d:
                        per_field[k] = max(per_field[k], d)
                        max_abs = max(max_abs, abs(fx - fy))
                    worst = max(worst, d)
        per_tok[t] = worst
    vals = sorted(per_tok.values())
    rep = {"n_reference_tokens": len(ref), "n_rehash_to_seam_fingerprint": rehash_ok,
           "rehash_failures": rehash_bad[:50], "n_compared_with_thor": len(per_tok),
           "max_ulp_overall": (vals[-1] if vals else None),
           "ulp_quantiles": ({q: vals[min(len(vals) - 1, int(q * len(vals)))] for q in (0.5, 0.9, 0.99)} if vals else None),
           "n_tokens_identical": sum(1 for v in vals if v == 0),
           "max_ulp_per_field": per_field, "max_abs_diff_any_value": max_abs}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep))
    return 0 if not rehash_bad else 1


if __name__ == "__main__":
    raise SystemExit(main())
