"""P1g - THE LANE-CHANGE **DEMAND** THE CORPUS ACTUALLY PLACES  (model-free)

RULE ZERO: the label and vocabulary defects are a diagnosis, not a product.
The cheapest experiment that decides whether fixing them is worth a retrain is
to measure how much lane-change behaviour the ground truth CONTAINS -- read
straight off the recorded ego poses, with no label pipeline and no model in the
path.  This is the SUPPLY/DEMAND complement of P1c: P1c showed every candidate's
heading is monotone, so a window whose GT heading is NON-MONOTONE while it
displaces a lane width is a window the vocabulary provably cannot serve.

DEFINITIONS, DECLARED BEFORE ANY NUMBER.  For the GT path over 0..T s in the
ego frame at t0 (x, y, yaw taken from the recorded poses, T = 6.0 s):
  lat_T      = y(T)                          lateral offset, metres
  dyaw_T     = wrap(yaw(T) - yaw(0))         net heading change
  dyaw_peak  = the largest |wrap(yaw(t)-yaw(0))| over the band
  RETURNS    = |dyaw_T| <= 10 deg AND dyaw_peak >= 3 deg
               (the heading LEAVES and COMES BACK -- non-monotone)
  LANE CHANGE (strict)  : |lat_T| in [2.5, 5.0) m  AND RETURNS
  LANE CHANGE (relaxed) : |lat_T| >= 2.5 m         AND RETURNS
  TURN (the CONTROL box, must read non-zero) : |dyaw_T| >= 30 deg

CONTROLS
  C-poses : the window's v0 must equal poses[ws + offset, 3] exactly, or the
            window->pose mapping is wrong and every row here is void.
  C-turn  : the TURN box must read non-zero.
  C-cover : windows without a full 6 s of recorded future are EXCLUDED and
            counted, never silently truncated.
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

import _env  # noqa: F401

sys.stdout.reconfigure(encoding="utf-8", errors="replace")

EPS = r"C:\Users\Admin\refav1_eval_full\eps"
HZ = 10.0
T_S = 6.0
LANE_LO, LANE_HI = 2.5, 5.0
RET_NET_DEG = 10.0
RET_PEAK_DEG = 3.0
TURN_DEG = 30.0


def wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-json", required=True)
    ap.add_argument("--dump", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    with open(a.arm_json, encoding="utf-8") as fh:
        man = json.load(fh)["refcv3"]["manifest"]
    clip_of = {int(e["file_index"]): e["clip_id"] for e in man["episodes"]}
    off = int(man["corpus"]["frames"]["provider_to_raw_frame_offset"])
    n_fut = int(round(T_S * HZ))

    rows = []
    v0_err = 0.0
    n_all = n_short = 0
    for fi in sorted(clip_of):
        z = np.load(os.path.join(a.dump, f"ep{fi:03d}.npz"), allow_pickle=True)
        ws = z["ws"].astype(int)
        v0 = z["v0"].astype(np.float64)
        p = torch.load(os.path.join(EPS, clip_of[fi] + ".v2ep.pt"),
                       map_location="cpu",
                       weights_only=False)["poses"].numpy().astype(np.float64)
        v0_err = max(v0_err, float(np.abs(p[ws + off, 3] - v0).max()))
        for i, w in enumerate(ws):
            n_all += 1
            j0 = w + off
            if j0 + n_fut >= len(p):
                n_short += 1
                continue
            seg = p[j0:j0 + n_fut + 1]
            c, s = np.cos(-seg[0, 2]), np.sin(-seg[0, 2])
            dx, dy = seg[:, 0] - seg[0, 0], seg[:, 1] - seg[0, 1]
            lat = s * dx + c * dy
            yaw = wrap(seg[:, 2] - seg[0, 2])
            rows.append((fi, w, v0[i], lat[-1], np.degrees(yaw[-1]),
                         np.degrees(np.abs(yaw).max()),
                         np.degrees(np.abs(yaw)[np.argmax(np.abs(lat))])))
    R = np.array(rows, dtype=np.float64)
    fi_, ws_, v0_, latT, dyawT, dyawPk, _ = R.T
    print(f"[C-poses] max|dump v0 - poses[ws+{off}, 3]| = {v0_err:.3e}")
    assert v0_err < 1e-4
    print(f"[C-cover] windows with a full {T_S} s of recorded future: "
          f"{len(R)} / {n_all}   (excluded, too short: {n_short})")

    returns = (np.abs(dyawT) <= RET_NET_DEG) & (dyawPk >= RET_PEAK_DEG)
    lc = (np.abs(latT) >= LANE_LO) & (np.abs(latT) < LANE_HI) & returns
    lcr = (np.abs(latT) >= LANE_LO) & returns
    turn = np.abs(dyawT) >= TURN_DEG
    print(f"\nGT DEMAND over {len(R)} windows / {len(set(fi_.tolist()))} clips")
    print(f"  heading LEAVES AND RETURNS (non-monotone)        : "
          f"{int(returns.sum())} / {len(R)}  ({100.0*returns.mean():.2f} %)")
    print(f"  LANE CHANGE strict  |lat| in [{LANE_LO},{LANE_HI}) + RETURNS : "
          f"{int(lc.sum())} / {len(R)}  ({100.0*lc.mean():.2f} %)")
    print(f"  LANE CHANGE relaxed |lat| >= {LANE_LO} + RETURNS            : "
          f"{int(lcr.sum())} / {len(R)}  ({100.0*lcr.mean():.2f} %)")
    print(f"  TURN box |dyaw| >= {TURN_DEG} deg [CONTROL, must be non-zero]: "
          f"{int(turn.sum())} / {len(R)}  ({100.0*turn.mean():.2f} %)")
    cl = sorted({int(f) for f in fi_[lcr]})
    print(f"  clips containing >= 1 relaxed lane-change window : "
          f"{len(cl)} / {len(set(fi_.tolist()))}")
    for f in cl[:15]:
        k = int((fi_[lcr] == f).sum())
        print(f"    {clip_of[f][:8]}  {k} windows")
    out = {"_tier": "model-free GT property (no checkpoint, no forward pass)",
           "_evidence_class": "MEASURED (ours)",
           "definitions": {"T_s": T_S, "lane_band_m": [LANE_LO, LANE_HI],
                           "returns_net_deg": RET_NET_DEG,
                           "returns_peak_deg": RET_PEAK_DEG,
                           "turn_deg": TURN_DEG},
           "n_windows_scored": int(len(R)), "n_windows_all": int(n_all),
           "n_windows_excluded_short": int(n_short),
           "n_clips": int(len(set(fi_.tolist()))),
           "heading_returns": int(returns.sum()),
           "lane_change_strict": int(lc.sum()),
           "lane_change_relaxed": int(lcr.sum()),
           "turn_control": int(turn.sum()),
           "clips_with_lane_change": [clip_of[f] for f in cl],
           "v0_roundtrip_max_abs": v0_err}
    with open(a.out, "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
