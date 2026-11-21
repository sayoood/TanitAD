#!/usr/bin/env python3
"""PREDICT the lead coverage of the canonical val40 windows — a falsifiable number the eval-host
run will confirm or refute.

The exact `t0` of a val40 window needs the episode's poses (eval host only) or the camera
timestamps parquet (inside a ~2 GB camera zip). Neither is here. But the episode grid is
`t = a + b*i`, and both parameters are measurable on the 500 R0 clips where the camera clock IS
local:  `b` from the registration fit, and `a` as an offset from the egomotion start.

So: estimate `a` from that measured offset, build the windows, and report coverage — WITH a
sensitivity sweep over t0 (+-0.25 s, ~2x the offset's own spread) so the number carries its own
error bar instead of pretending to be exact.

⛔ This is an ESTIMATE, labelled as one. It is not a result and must not be quoted as the val40
distance-keeping number — that comes from `score_val40_lead.py` on the eval host.
"""
from __future__ import annotations

import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(r"G:/Meine Ablage/SayBouBase/raw/Projects/TanitAD")
sys.path.insert(0, str(REPO / "stack"))
sys.path.insert(0, str(REPO / "taniteval"))
sys.path.insert(0, str(REPO / "stack" / "scripts"))
from lead_state_gate import VEHICLE_CLASSES, quaternion_yaw       # noqa: E402
from taniteval.lead_source import (LEAD, NO_LABEL, NO_LEAD,       # noqa: E402
                                   lead_block, window_last_indices)

ROOT = Path(r"C:/Users/Admin/tanitad-data/physicalai")
HERE = Path(__file__).parent
WP_REL_S = np.array([0.5, 1.0, 1.5, 2.0])
SHIFTS = (-0.25, -0.1, 0.0, 0.1, 0.25)


def main():
    reg = json.loads((HERE / "registration_per_clip.json").read_text())
    off = np.array([r["a_true_s"] - r["ego_t0_s"] for r in reg])
    b = np.array([r["b_fit_s"] for r in reg])
    prior = {"n_clips": len(reg),
             "a_minus_ego_t0_s": {"median": round(float(np.median(off)), 5),
                                  "p05": round(float(np.percentile(off, 5)), 5),
                                  "p95": round(float(np.percentile(off, 95)), 5),
                                  "std": round(float(off.std()), 5)},
             "grid_dt_s": {"median": round(float(np.median(b)), 6),
                           "min": round(float(b.min()), 6), "max": round(float(b.max()), 6)}}
    print(json.dumps(prior, indent=1))
    a_off, b_med = float(np.median(off)), float(np.median(b))

    rows = json.loads((HERE / "val40_map.json").read_text())
    out = {"_what": "ESTIMATED lead coverage of the canonical val40 windows",
           "_evidence_class": "ESTIMATED — t0 is reconstructed from a measured prior, not read "
                              "from the episode. The eval-host run supersedes this.",
           "_prior": prior, "per_shift": {}, "per_clip": []}
    for shift in SHIFTS:
        tally = {LEAD: 0, NO_LEAD: 0, NO_LABEL: 0}
        n_ep = 0
        detail = []
        for r in rows:
            cid, ch, T = r["clip_id"], int(r["chunk"]), int(r["T"])
            ezp = ROOT / "labels" / "egomotion" / f"egomotion.chunk_{ch:04d}.zip"
            ozp = ROOT / "labels" / "obstacle.offline" / f"obstacle.offline.chunk_{ch:04d}.zip"
            if not ezp.exists():
                continue
            with zipfile.ZipFile(ezp) as ez:
                m = next((x for x in ez.namelist() if x.endswith(".parquet")
                          and x.split("/")[-1].startswith(cid)), None)
                if m is None:
                    continue
                ego_df = pd.read_parquet(io.BytesIO(ez.read(m)))
            t = ego_df["timestamp"].to_numpy(np.float64) / 1e6
            o = np.argsort(t)
            g = lambda c: ego_df[c].to_numpy(np.float64)[o]        # noqa: E731
            ego = {"t": t[o], "x": g("x"), "y": g("y"),
                   "yaw": np.unwrap(quaternion_yaw(g("qx"), g("qy"), g("qz"), g("qw"))),
                   "v": np.hypot(g("vx"), g("vy"))}
            obs = None
            if ozp.exists():
                with zipfile.ZipFile(ozp) as oz:
                    mo = next((x for x in oz.namelist() if x.endswith(".parquet")
                               and x.split("/")[-1].startswith(cid)), None)
                    if mo is not None:
                        df = pd.read_parquet(io.BytesIO(oz.read(mo)))
                        obs = {"t": df["timestamp_us"].to_numpy(np.float64) / 1e6,
                               "track": df["track_id"].astype(str).to_numpy(object),
                               "center_x": df["center_x"].to_numpy(np.float64),
                               "center_y": df["center_y"].to_numpy(np.float64),
                               "size_x": df["size_x"].to_numpy(np.float64),
                               "is_vehicle": df["label_class"].astype(str)
                               .isin(VEHICLE_CLASSES).to_numpy()}
            last = window_last_indices(T)
            t0s = float(ego["t"][0]) + a_off + shift + b_med * last
            blk = lead_block(t0s, WP_REL_S, obs, ego)
            for k, v in blk["counts"].items():
                tally[k] += int(v)
            n_ep += 1
            if shift == 0.0:
                detail.append({"file": r["file"], "chunk": ch, "n_windows": int(last.size),
                               "has_obstacle": bool(obs is not None),
                               **{k: int(v) for k, v in blk["counts"].items()}})
        n = sum(tally.values())
        out["per_shift"][f"{shift:+.2f}s"] = {
            "n_episodes": n_ep, "n_windows": n, **tally,
            "lead_rate_over_labelled": round(tally[LEAD] / max(n - tally[NO_LABEL], 1), 4),
            "lead_rate_over_all": round(tally[LEAD] / max(n, 1), 4)}
        if shift == 0.0:
            out["per_clip"] = detail
        print(f"shift {shift:+.2f}s -> {json.dumps(out['per_shift'][f'{shift:+.2f}s'])}")
    (HERE / "val40_coverage.json").write_text(json.dumps(out, indent=1))
    print("wrote val40_coverage.json")


if __name__ == "__main__":
    main()
