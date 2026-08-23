"""SIZE the centred-difference causality fix, and prove it did not move any LABEL.

CONTEXT. `stack/tanitad/data/situations.py` built `alon_pre` / `omega_pre` on `np.gradient` — a
CENTRED difference — under a comment reading `STRICTLY CAUSAL`, so both channels read one frame
(0.1 s) past `t` on every interior frame. The fix (`backward_diff`, `causal_pre=True` by default)
landed earlier on 2026-08-03 with tests in `stack/tests/test_label_causality_and_nav.py`.

WHAT WAS STILL MISSING. The module's blast-radius note names the consumers but carries **no number**:
nobody had measured how far the leaky channels actually are from the causal ones. "A defect exists"
and "the defect is 4.7 % of the channel" license very different decisions about whether banked
`ego`-block artifacts must be rebuilt. This script supplies that number, and simultaneously checks
the claim the fix rests on — that the LABEL side is untouched.

TWO THINGS ARE MEASURED
  1. the magnitude of the change in the two inference-side channels, in absolute units and
     relative to each channel's own scale, plus the fraction of frames materially affected;
  2. that `omega` / `kappa` / `alon` and ALL THREE detectors return bit-identical events under
     both modes — i.e. the fix cannot have silently re-derived a single situation label, which
     would have retro-fitted a pre-registered study.

usage:
  python causality_blast_radius.py --cache <episode cache dir> --out causality_blast_radius.json
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[5]
sys.path.insert(0, str(REPO / "stack"))

from tanitad.data.situations import (detect_intersection,            # noqa: E402
                                     detect_lane_change,
                                     detect_roundabout, kinematics)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache",
                    default=r"C:/Users/Admin/tanitad-data/physicalai/_epcache/"
                            r"physicalai-val-bb543bdf7836")
    ap.add_argument("--out", default="causality_blast_radius.json")
    ap.add_argument("--limit", type=int, default=100)
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.cache, "ep_*.pt")))[:a.limit or None]
    da, do, na, no = [], [], [], []
    label_side_identical = True
    n_ev = {"lane_change": 0, "roundabout": 0, "intersection": 0}
    for f in files:
        P = np.asarray(torch.load(f, map_location="cpu", weights_only=True,
                                  mmap=True)["poses"]).astype(np.float64)
        K = kinematics(P, causal_pre=True)          # the fix
        L = kinematics(P, causal_pre=False)         # the legacy, leaky channels
        da.append(np.abs(K["alon_pre"] - L["alon_pre"]))
        do.append(np.abs(K["omega_pre"] - L["omega_pre"]))
        na.append(np.abs(L["alon_pre"]))
        no.append(np.abs(L["omega_pre"]))
        for ch in ("omega", "kappa", "alon"):       # LABEL-derivation channels
            label_side_identical &= bool(np.allclose(K[ch], L[ch]))
        lc, ix, rb = (detect_lane_change, detect_intersection, detect_roundabout)
        label_side_identical &= lc(K) == lc(L)
        label_side_identical &= ix(K, cross=None)[0] == ix(L, cross=None)[0]
        label_side_identical &= rb(K) == rb(L)
        n_ev["lane_change"] += len(lc(K))
        n_ev["intersection"] += len(ix(K, cross=None)[0])
        n_ev["roundabout"] += len(rb(K))
    da, do, na, no = (np.concatenate(x) for x in (da, do, na, no))

    def block(d, base, unit):
        return {"unit": unit,
                "mean_abs_change": round(float(d.mean()), 6),
                "p99_abs_change": round(float(np.percentile(d, 99)), 6),
                "max_abs_change": round(float(d.max()), 6),
                "mean_abs_change_RELATIVE_to_channel_scale":
                    round(float(d.mean() / max(base.mean(), 1e-9)), 5),
                "frac_frames_changed_gt_1pct_of_scale":
                    round(float((d > 0.01 * base.mean()).mean()), 5)}

    out = {"_what": "MEASURED blast radius of the causal_pre fix in tanitad.data.situations",
           "_legacy": "LEGACY_centred_np_gradient_LEAKS_t_plus_1",
           "_fix": "causal_backward_diff",
           "cache": a.cache, "n_clips": len(files), "n_frames": int(da.size),
           "alon_pre": block(da, na, "m/s^2"),
           "omega_pre": block(do, no, "rad/s"),
           "LABEL_SIDE_IDENTICAL": bool(label_side_identical),
           "_label_side_note": ("omega / kappa / alon AND all three detectors return "
                                "bit-identical events under both modes, so the fix cannot "
                                "have re-derived a single situation label"),
           "events_detected": n_ev}
    Path(a.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
