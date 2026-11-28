#!/usr/bin/env python3
"""Lateral / longitudinal decomposition, and the 2x2 (2x3) factorial table.

The regression this work addresses was **100 % longitudinal**, so a composite
delta with no axis attached is not admissible. This recomputes, from the
committed per-window dumps and with NO GPU:

* per-arm mean signed along-track and cross-track endpoint error vs the LOGGED
  path, in the reference-pose frame (``pseudosim._cross_and_along``, imported);
* the share of squared endpoint error carried by each axis;
* the factorial: plan SHAPE (steers / straight) x plan SCHEDULE (v1's own /
  v0*t / the true logged distance).

⚠️ ``_cross_and_along`` is IMPORTED from ``taniteval.pseudosim``, not
reimplemented — it is the same function the composite uses, so the axes here are
the axes the score is built on.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

for p in ("/root/TanitAD/stack", "/workspace/_egoin/lib", "/root/taniteval"):
    if p not in sys.path and Path(p).exists():
        sys.path.insert(0, p)

from taniteval.pseudosim import _cross_and_along  # noqa: E402

PW_KEYS = ("traj", "ref_path", "ref_yaw", "v0", "pt_dlat", "pt_dyaw",
           "pt_dlon", "anchor", "ep_i", "eid")


def load(path: Path) -> dict:
    z = np.load(path, allow_pickle=False)
    return {k: (torch.as_tensor(z[k]) if k != "eid" else [str(x) for x in z[k]])
            for k in PW_KEYS}


def axes(pw: dict) -> dict:
    """Endpoint error split onto the two axes the composite is built from."""
    x, y, ref_x, ref_y = _cross_and_along(pw)
    along_err = (x[:, -1] - ref_x[:, -1])        # + = plan travelled TOO FAR
    cross_err = (y[:, -1] - ref_y[:, -1])        # + = plan is left of the log
    tot = (along_err ** 2 + cross_err ** 2)
    return {
        "n_rows": int(x.shape[0]),
        "along_err_mean_m": round(float(along_err.mean()), 4),
        "along_err_absmean_m": round(float(along_err.abs().mean()), 4),
        "along_err_rms_m": round(float((along_err ** 2).mean().sqrt()), 4),
        "along_err_sd_m": round(float(along_err.std()), 4),
        "cross_err_mean_m": round(float(cross_err.mean()), 4),
        "cross_err_absmean_m": round(float(cross_err.abs().mean()), 4),
        "cross_err_rms_m": round(float((cross_err ** 2).mean().sqrt()), 4),
        "longitudinal_share_of_sq_err": round(
            float((along_err ** 2).sum() / tot.sum().clamp_min(1e-9)), 4),
        "plan_end_along_m": round(float(x[:, -1].mean()), 4),
        "logged_end_along_m": round(float(ref_x[:, -1].mean()), 4),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--arms", default="")
    a = ap.parse_args()
    in_dir = Path(a.in_dir)
    want = [s for s in a.arms.split(",") if s] or None

    out: dict = {
        "_what": "lateral / longitudinal decomposition of the plan ENDPOINT "
                 "error, on the same axes the composite is built from",
        "_source": "taniteval.pseudosim._cross_and_along (IMPORTED, not "
                   "reimplemented)",
        "_evidence_class": "MEASURED (ours; artifact = this JSON)",
        "_sign_convention": {
            "along_err": "+ = the plan travelled FURTHER than the human did",
            "cross_err": "+ = the plan ends LEFT of the logged path"},
        "_no_gpu": True, "arms": {},
    }
    for f in sorted(in_dir.glob("pw_*.npz")):
        name = f.name[len("pw_"):-len(".npz")]
        if want and name not in want:
            continue
        out["arms"][name] = axes(load(f))
        n = out["arms"][name]
        print(f"{name:24s} along {n['along_err_mean_m']:+8.3f} "
              f"(|.| {n['along_err_absmean_m']:6.3f}, rms {n['along_err_rms_m']:6.3f})  "
              f"cross {n['cross_err_mean_m']:+7.3f} "
              f"(|.| {n['cross_err_absmean_m']:5.3f})  "
              f"lon_share {n['longitudinal_share_of_sq_err']:.3f}", flush=True)
    Path(a.out).write_text(json.dumps(out, indent=2), encoding="utf-8")
    print(f"[decompose] -> {a.out}")
    print("DECOMPOSE_DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
