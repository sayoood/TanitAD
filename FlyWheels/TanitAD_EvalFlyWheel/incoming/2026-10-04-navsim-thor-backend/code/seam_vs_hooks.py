#!/usr/bin/env python3
"""Prove a seam file is the one a banked dev-box run actually scored (NAVSIM venv / any numpy).

A W3 navtest counts.json names its seam by PATH only (no sha256), and a seam can be re-bridged in
place. The wrapper's hooks file, however, recorded the agent poses the scorer received for every
token (``record_poses=True``). This compares those poses with the seam's rows, bit for bit
(float32 -> float), for every token in ``--tokens`` (or every hooked token).

    python seam_vs_hooks.py --seam <seam.npz> --hooks <label>_hooks.json [--tokens sub.json] --out x.json
"""
from __future__ import annotations

import argparse
import hashlib
import json

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", required=True)
    ap.add_argument("--hooks", required=True)
    ap.add_argument("--tokens", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    z = np.load(a.seam, allow_pickle=False)
    rows = {str(t): np.asarray(p, dtype=np.float32) for t, p in zip(z["token"], z["poses"])}
    calls = json.load(open(a.hooks, encoding="utf-8"))["pdm_score_calls"]
    hooked = {c["token"]: c["agent_poses"] for c in calls if "agent_poses" in c}
    want = (json.load(open(a.tokens, encoding="utf-8"))["tokens"] if a.tokens else sorted(hooked))
    n_eq = n_ne = n_missing = 0
    worst = 0.0
    for t in want:
        if t not in hooked or t not in rows:
            n_missing += 1
            continue
        h = np.asarray(hooked[t], dtype=np.float64)
        s = rows[t].astype(np.float64)
        if h.shape == s.shape and np.array_equal(h, s):
            n_eq += 1
        else:
            n_ne += 1
            if h.shape == s.shape:
                worst = max(worst, float(np.abs(h - s).max()))
    h = hashlib.sha256(open(a.seam, "rb").read()).hexdigest()
    rep = {"seam": a.seam, "seam_sha256": h, "hooks": a.hooks, "n_tokens": len(want),
           "n_identical": n_eq, "n_different": n_ne, "n_missing": n_missing,
           "max_abs_diff_when_different": worst,
           "verdict": "SEAM_IS_THE_SCORED_ONE" if n_eq == len(want) and len(want) else "MISMATCH"}
    json.dump(rep, open(a.out, "w", encoding="utf-8"), indent=1)
    print(json.dumps(rep))
    return 0 if rep["verdict"] == "SEAM_IS_THE_SCORED_ONE" else 1


if __name__ == "__main__":
    raise SystemExit(main())
