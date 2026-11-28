"""Recompute the headline numbers from the per-frame dump ALONE — no GPU, no pod, no checkpoint.

This exists so the study is falsifiable by a reader with nothing but this folder. It reads
`artifacts/heldout_frames.npz` (every held-out frame: labels, valid masks, ego channels and every
arm's score) and reproduces, independently of `sc_eval.py`:

  * the base rate, positive count and positive-cluster count per situation (C-POW),
  * every arm's AP and AP/base,
  * the paired above-chance ΔAP against a constant score,

then diffs them against `artifacts/sc_results.json`. Any disagreement is printed and the exit code
is non-zero.

⚠️ The AP and the bootstrap machinery are IMPORTED from `h2c_stats` (→ `taniteval.ci`), the same as
the evaluator — a re-implementation here would test my own arithmetic twice rather than testing the
pipeline once. What this script independently re-derives is the *reduction*: which rows are scored,
how they are grouped and which numbers are compared.

usage:  python sc_recheck.py <artifacts_dir>
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "..",
                                                "2026-07-26-h2-classifier", "scripts")))
from h2c_stats import average_precision  # noqa: E402
from taniteval.ci import _draws, episode_index  # noqa: E402

TOL = 1e-6


def main():
    art = sys.argv[1]
    Z = np.load(os.path.join(art, "heldout_frames.npz"), allow_pickle=True)
    R = json.load(open(os.path.join(art, "sc_results.json")))
    sits = [str(s) for s in Z["situations"]]
    arms = [str(a) for a in Z["arms"]]
    eid_all, Y, V = Z["clip_cluster"], Z["y"], Z["valid"]
    bad = []
    for si, sit in enumerate(sits):
        m = V[:, si].astype(bool)
        y = Y[m, si].astype(float)
        eid = eid_all[m]
        r = R["situations"][sit]
        checks = {
            "n_windows": (int(m.sum()), r["n_windows"]),
            "n_pos": (int(y.sum()), r["n_pos"]),
            "n_pos_clusters": (int(len(np.unique(eid[y > 0]))), r["n_pos_clusters"]),
            "n_clusters": (int(len(np.unique(eid))), r["n_clusters"]),
            "base_rate": (round(float(y.mean()), 9), round(r["base_rate"], 9)),
        }
        for k, (a, b) in checks.items():
            if isinstance(a, float):
                ok = abs(a - b) <= 1e-9
            else:
                ok = a == b
            print(f"  {sit:14s} {k:16s} dump={a} json={b} {'OK' if ok else 'MISMATCH'}")
            if not ok:
                bad.append((sit, k, a, b))
        # AP per arm, and the paired above-chance test
        uniq, idx_by_ep = episode_index(eid)
        draws = [d for d in _draws(uniq, idx_by_ep, 400, 0)]
        const = np.zeros(len(y))
        cb = np.array([average_precision(y[s], const[s]) for s in draws])
        for arm in arms + ["heur_kin"]:
            s_ = (Z["heur_kin"][m, si] if arm == "heur_kin" else Z[arm][m, si]).astype(float)
            ap = average_precision(y, s_)
            ref = r["AP"][arm]["point"]
            ok = abs(ap - ref) <= TOL
            ab = np.array([average_precision(y[s], s_[s]) for s in draws]) - cb
            lo = float(np.quantile(ab, .025))
            refsep = r["above_chance"][arm]["separated"]
            print(f"  {sit:14s} AP[{arm:20s}] dump={ap:.6f} json={ref:.6f} {'OK' if ok else 'MISMATCH'}"
                  f" | above-chance(B=400) lo={lo:+.5f} sep={lo > 0} (json {refsep})")
            if not ok:
                bad.append((sit, f"AP:{arm}", ap, ref))
    print("\nRECHECK:", "ALL AGREE" if not bad else f"{len(bad)} MISMATCHES: {bad}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
