#!/usr/bin/env python3
"""SPEC AMENDMENT A2's fixed smoother — and the seams its controls need. Zero training, zero tuning.

    # the arm: refcv4b's own seam, smoothed
    python code/smooth_seam.py --in raw/bridge_navtest/seam_A1_ego_cmd.npz \
        --out raw/bridge_navtest/seam_A1S.npz --arm A1S
    # the controls: the logged human future through the SEAM path, raw and smoothed
    python code/smooth_seam.py --human --tokens raw/A1_sub200_tokens.json \
        --out raw/bridge_navtest/seam_HUMANseam_sub200.npz --arm HUMANseam --no-smooth
    python code/smooth_seam.py --human --tokens raw/A1_sub200_tokens.json \
        --out raw/bridge_navtest/seam_HUMANS_sub200.npz --arm HUMANS

THE SMOOTHER (fixed in SPEC §7b before any smoothed number existed): per plan, a least-squares
CUBIC in time for x(t) and y(t) separately, ANCHORED at the origin (no constant term: p(0) = 0,
the ego's own pose at t0), equal weights over the 8 poses at t = 0.5 … 4.0 s; heading =
atan2(ẏ, ẋ) of the fit, except where the fitted speed is below 0.5 m/s (the four-families
harness's own ``min_ds_mps``), where the input heading is kept. Degree 3 is the lowest that can
represent a lane change. ⛔ There is NO free parameter, and nothing here reads a score.
"""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np

T = np.arange(1, 9, dtype=np.float64) * 0.5            # 0.5 … 4.0 s — NAVSIM's 8 × 0.5 s
V_MIN = 0.5                                             # m/s — four_families min_ds_mps
EXPORT = "D:/Archive/devbox-C/navsim/exp/w3_navtest_v1/inputs/navtest_inputs.json.gz"
BASIS = np.stack([T, T ** 2, T ** 3], axis=1)           # [8, 3], no constant column
DBASIS = np.stack([np.ones_like(T), 2 * T, 3 * T ** 2], axis=1)


def cubic_smooth(poses: np.ndarray, v_min: float = V_MIN) -> np.ndarray:
    """One plan [8, 3] (x, y, heading) -> the anchored cubic fit, same shape."""
    p = np.asarray(poses, np.float64)
    cx, *_ = np.linalg.lstsq(BASIS, p[:, 0], rcond=None)
    cy, *_ = np.linalg.lstsq(BASIS, p[:, 1], rcond=None)
    x, y = BASIS @ cx, BASIS @ cy
    vx, vy = DBASIS @ cx, DBASIS @ cy
    h = np.arctan2(vy, vx)
    slow = np.hypot(vx, vy) < v_min
    h[slow] = p[slow, 2]
    return np.stack([x, y, h], axis=1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="inp", default=None, help="a seam npz to smooth")
    ap.add_argument("--human", action="store_true",
                    help="build the seam from the export's logged human future instead")
    ap.add_argument("--tokens", default=None, help="restrict to these tokens (JSON)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--no-smooth", action="store_true",
                    help="pass the poses through UNCHANGED (the seam-path control)")
    a = ap.parse_args(argv)
    keep = None
    if a.tokens:
        d = json.load(open(a.tokens, encoding="utf-8"))
        keep = set(d["tokens"] if isinstance(d, dict) else d)
    if a.human:
        doc = json.load(gzip.open(EXPORT, "rt", encoding="utf-8"))["tokens"]
        toks = sorted(t for t in doc if keep is None or t in keep)
        fps = [doc[t]["fingerprint"] for t in toks]
        raw = np.asarray([doc[t]["human_future_poses"] for t in toks], np.float64)
        source = "logged human future (export: human_future_poses)"
    else:
        z = np.load(a.inp, allow_pickle=False)
        toks_all = [str(t) for t in z["token"].tolist()]
        sel = [i for i, t in enumerate(toks_all) if keep is None or t in keep]
        toks = [toks_all[i] for i in sel]
        fps = [str(z["fingerprint"][i]) for i in sel]
        raw = np.asarray(z["poses"], np.float64)[sel]
        source = f"{a.inp}"
    if not toks:
        raise SystemExit("⛔ no tokens selected")
    out = raw.copy() if a.no_smooth else np.stack([cubic_smooth(p) for p in raw])
    if not np.isfinite(out).all():
        raise SystemExit("⛔ non-finite poses after smoothing")
    delta = np.abs(out - raw)
    np.savez(a.out, token=np.asarray(toks), fingerprint=np.asarray(fps),
             source=np.asarray([a.arm] * len(toks)), poses=out.astype(np.float32),
             sampling=np.asarray([8, 0.5]), arm=np.asarray(a.arm),
             n_requested=np.asarray(len(toks)))
    man = {"arm": a.arm, "n": len(toks), "input": source, "smoothed": not a.no_smooth,
           "smoother": (None if a.no_smooth else
                        "SPEC §7b: anchored LSQ cubic in t for x and y; heading atan2(ẏ,ẋ); "
                        f"input heading kept below {V_MIN} m/s"),
           "max_abs_change": {"x_m": float(delta[..., 0].max()), "y_m": float(delta[..., 1].max()),
                              "heading_rad": float(delta[..., 2].max())},
           "median_abs_change": {"x_m": float(np.median(delta[..., 0])),
                                 "y_m": float(np.median(delta[..., 1])),
                                 "heading_rad": float(np.median(delta[..., 2]))}}
    Path(a.out).with_suffix(".manifest.json").write_text(json.dumps(man, indent=1),
                                                         encoding="utf-8")
    print(json.dumps(man, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
