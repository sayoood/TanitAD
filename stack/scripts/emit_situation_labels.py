"""Emit per-window LANE CHANGE and INTERSECTION labels over an episode cache.

PI direction 2026-07-29: *"the classifier should detect the situation of lane change not
necessarily based on objects — only on situational labels, so you don't need objects. You need the
detection of lane changes and an intersection label."*

⇒ This produces exactly those two labels and nothing else. No object features, no `obstacle.offline`,
no detections. Roundabout is computed but NOT emitted (26 held-out clusters = UNPOWERED; the PI
deferred it).

THE DISCIPLINE THAT MAKES THESE LABELS USABLE (PI, same directive):
  **future ego motion is legitimate for GENERATING and VALIDATING labels, and illegitimate as a
  model INPUT.** The detectors here read the whole pose track including the future — that is what
  makes them ground truth. A classifier trained on these must take the CAMERA (plus the ego's own
  instantaneous state, which a vehicle genuinely has at inference) and must never be fed the pose
  track these labels came from.

The target is ANTICIPATION, not recognition: y(t) = 1 iff an onset falls in (t, t+lead]; frames
INSIDE an ongoing situation are masked out of scoring, so a classifier cannot score by noticing a
manoeuvre already under way.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import numpy as np
import torch

from tanitad.data.situations import (HZ, anticipation_target, detect_intersection,
                                     detect_lane_change, kinematics)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True, help="episode cache dir (ep_*.pt or *.v2ep.pt)")
    ap.add_argument("--out", required=True, help="output .npz")
    ap.add_argument("--lead-s", type=float, default=3.0)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.cache, "ep_*.pt")) or
                   glob.glob(os.path.join(a.cache, "*.v2ep.pt")))
    if a.limit:
        files = files[:a.limit]
    print(f"[situ] {len(files)} episodes in {a.cache}", flush=True)

    eid, tt, y_lc, y_ix, v_lc, v_ix = [], [], [], [], [], []
    n_lc_ep = n_ix_ep = 0
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        P = d["poses"].numpy() if torch.is_tensor(d["poses"]) else np.asarray(d["poses"])
        K = kinematics(P.astype(np.float64))
        T = P.shape[0]
        lc = detect_lane_change(K)
        ix, _turns, _x = detect_intersection(K)
        n_lc_ep += bool(lc)
        n_ix_ep += bool(ix)
        ylc, vlc = anticipation_target(T, lc, lead_s=a.lead_s)
        yix, vix = anticipation_target(T, ix, lead_s=a.lead_s)
        eid.append(np.full(T, i, np.int32)); tt.append(np.arange(T, dtype=np.int32))
        y_lc.append(ylc); y_ix.append(yix); v_lc.append(vlc); v_ix.append(vix)
        if (i + 1) % 100 == 0:
            print(f"[situ] {i+1}/{len(files)}", flush=True)

    cat = lambda xs: np.concatenate(xs)
    Y_LC, Y_IX, V_LC, V_IX = cat(y_lc), cat(y_ix), cat(v_lc), cat(v_ix)
    np.savez_compressed(a.out, episode=cat(eid), t=cat(tt),
                        y_lane_change=Y_LC, valid_lane_change=V_LC,
                        y_intersection=Y_IX, valid_intersection=V_IX,
                        lead_s=a.lead_s, hz=HZ)
    summary = {
        "cache": a.cache, "episodes": len(files), "frames": int(Y_LC.size),
        "lead_s": a.lead_s,
        "lane_change": {"episodes_with": n_lc_ep,
                        "ep_pct": round(100 * n_lc_ep / max(1, len(files)), 2),
                        "positive_frames": int(Y_LC[V_LC].sum()),
                        "scorable_frames": int(V_LC.sum()),
                        "base_rate": round(float(Y_LC[V_LC].mean()), 5) if V_LC.any() else None},
        "intersection": {"episodes_with": n_ix_ep,
                         "ep_pct": round(100 * n_ix_ep / max(1, len(files)), 2),
                         "positive_frames": int(Y_IX[V_IX].sum()),
                         "scorable_frames": int(V_IX.sum()),
                         "base_rate": round(float(Y_IX[V_IX].mean()), 5) if V_IX.any() else None},
        "_read": ("ANTICIPATION targets: y(t)=1 iff an onset falls in (t, t+lead]; frames inside an "
                  "ongoing situation are masked via valid_*. Labels are GENERATED from the full "
                  "pose track (future included) — that is legitimate for ground truth and "
                  "ILLEGITIMATE as a model input."),
    }
    print(json.dumps(summary, indent=1), flush=True)
    with open(os.path.splitext(a.out)[0] + "_summary.json", "w") as fh:
        json.dump(summary, fh, indent=1)
    print(f"[situ] -> {a.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
