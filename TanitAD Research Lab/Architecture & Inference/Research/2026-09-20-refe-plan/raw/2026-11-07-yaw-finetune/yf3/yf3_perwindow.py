#!/usr/bin/env python3
"""YF3 per-window LATERAL arithmetic (TANITAD venv; imports the SAME adapter + `four_families._seq_geometry` that families6 / `lateral()` use, so the numerators and denominators are the
block's own). families6 prints only POOLED means; a paired log-cluster bootstrap of a difference needs the per-window sums. For a window i:
    curv_abs_sum_i = sum over valid step PAIRS of |kappa_pred - kappa_gt|, curv_n_i = number of valid pairs   -> pooled MAE = sum(curv_abs_sum) / sum(curv_n)
which is `lateral()`'s `_masked(...).mean()` exactly (checked in test_yf3_analyze.py against the banked sft4_B_u400_families.json: curvature_mae_1pm 0.01882).
Same for heading (valid steps), yaw rate (pairs), cross (all H steps), along (all H steps), signed curvature / cross for the biases.
    python yf3_perwindow.py --seam <seam.npz (only the tokens of the set)> --tokens <w3 token json> --out <perwindow.npz>
"""
import argparse
import gzip
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np

ADAPTER = "E:/Projects/TanitAD/taniteval/adapters/navsim.py"
D_TANITEVAL = "E:/Projects/TanitAD/taniteval"
EXPORT_CANDIDATES = [f"{d}:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz" for d in "EFD"]


def load_adapter():
    if D_TANITEVAL not in sys.path:
        sys.path.append(D_TANITEVAL)
    spec = importlib.util.spec_from_file_location("navsim_adapter6", ADAPTER)
    ad = importlib.util.module_from_spec(spec)
    sys.modules["navsim_adapter6"] = ad
    spec.loader.exec_module(ad)
    from taniteval import four_families as ff
    return ad, ff


def export_doc(path=None):
    import os
    for p in ([path] if path else []) + EXPORT_CANDIDATES:
        if p and os.path.exists(p):
            return json.load(gzip.open(p, "rt", encoding="utf-8"))
    raise SystemExit("navtest export not found")


def per_window(seam_npz, toks, doc=None, verify=True):
    """-> dict of arrays aligned to `toks` (as given). `seam_npz` may hold more tokens than `toks`."""
    import torch
    ad, ff = load_adapter()
    doc = doc or export_doc()
    te = doc["tokens"]
    z = np.load(seam_npz, allow_pickle=False)
    idx = {str(t): i for i, t in enumerate(z["token"])}
    miss = [t for t in toks if t not in idx]
    if miss:
        raise SystemExit(f"{len(miss)} tokens absent from the seam, e.g. {miss[:3]}")
    human = np.asarray([te[t]["human_future_poses"] for t in toks], np.float64)
    v0 = [float(np.hypot(*te[t]["ego_statuses"][-1]["ego_velocity"])) for t in toks]
    logs = [te[t]["log_name"] for t in toks]
    pred = np.asarray([z["poses"][idx[t]] for t in toks], np.float64)
    w = ad.scenes_to_win(pred, human, frame="ego", origin_included=False, dt_s=0.5, scene_tokens=list(toks), ego_speed_mps=v0, log_names=logs, verify=verify)
    P_ = torch.as_tensor(w["pred_dense"]).float()
    G_ = torch.as_tensor(w["gt_dense"]).float()
    dt = float(w.get("dt_s", 0.5) or 0.5)
    P, G = ff._seq_geometry(P_, dt), ff._seq_geometry(G_, dt)
    both, bp = P["valid"] & G["valid"], P["pair_valid"] & G["pair_valid"]
    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi

    def S(x, m):                               # float64 sum of the masked entries per window (a NaN under the mask is impossible; outside it is dropped by where)
        return torch.where(m, x.double(), torch.zeros((), dtype=torch.float64)).sum(1).numpy()

    ones = torch.ones_like(P["cross"], dtype=torch.bool)
    out = {"token": np.asarray(toks), "log": np.asarray(logs), "v0": np.asarray(v0),
           "curv_abs_sum": S((P["curvature"] - G["curvature"]).abs(), bp), "curv_signed_sum": S(P["curvature"] - G["curvature"], bp), "curv_n": bp.sum(1).numpy().astype(np.float64),
           "head_abs_sum_rad": S(dh.abs(), both), "head_n": both.sum(1).numpy().astype(np.float64),
           "yaw_abs_sum_radps": S((P["yaw_rate"] - G["yaw_rate"]).abs(), bp), "yaw_n": bp.sum(1).numpy().astype(np.float64),
           "cross_abs_sum": S((P["cross"] - G["cross"]).abs(), ones), "cross_signed_sum": S(P["cross"] - G["cross"], ones), "cross_n": ones.sum(1).numpy().astype(np.float64),
           "along_abs_sum": S((P["along"] - G["along"]).abs(), ones), "along_signed_sum": S(P["along"] - G["along"], ones), "along_n": ones.sum(1).numpy().astype(np.float64)}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seam", required=True)
    ap.add_argument("--tokens", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    toks = sorted(json.load(open(a.tokens, encoding="utf-8"))["tokens"])
    # the adapter's frame-convention guard is a STATISTICAL check ("90 % of the turning windows agree"); on a smoke subset of ~18 tokens it fires on 7 of 8 windows. It is kept ON for any real set (n >= 100) and
    # skipped (and said so) only below that.
    verify = len(toks) >= 100
    if not verify:
        print(f"NOTE: adapter frame verification skipped: n={len(toks)} < 100 (statistical guard, not decision-grade on a smoke subset)")
    pw = per_window(a.seam, toks, verify=verify)
    np.savez(a.out, **pw)
    n = pw["curv_n"].sum()
    print(f"perwindow ok: {len(toks)} windows, curvature MAE {pw['curv_abs_sum'].sum() / n:.6f} over {int(n)} step pairs, heading MAE deg {math.degrees(pw['head_abs_sum_rad'].sum() / pw['head_n'].sum()):.4f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
