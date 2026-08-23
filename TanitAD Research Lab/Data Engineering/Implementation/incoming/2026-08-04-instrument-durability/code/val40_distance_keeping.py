#!/usr/bin/env python3
"""Turn the LONGITUDINAL family ON for val40 on Thor, and reproduce D-LEAD-1's SIGN.

`val40_lead_run.py` proves the lead BLOCK reproduces (881 windows, 270 LEAD). This proves the
block is actually CONSUMABLE: `four_families.longitudinal(..., lead=)` must stop returning
`distance_keeping: {"status": "UNAVAILABLE"}` and return real headway / time-gap / min-TTC.

⭐ THE REPRODUCTION. D-LEAD-1 (the pre-registered control that admitted the metric) measured
GT - CV on a different surface (2,417 clips / 14,027 windows): min-TTC +1.7474 s, headway
+0.9769 m, time-gap +0.1641 s — all three SEPARATED and positively signed (the human keeps more
distance than a hold-v0 policy that never brakes). Re-running GT vs CV on the val40 LEAD windows
must reproduce that SIGN. It is a different surface, so the MAGNITUDES are not expected to match
and are not claimed to.

⛔ CV here is the hold-`v0` baseline, not an arm. No arm is being scored.
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, "/home/nvidia/TanitAD/taniteval")

from taniteval import four_families  # noqa: E402
from taniteval.lead_source import (LEAD, RegistrationError, lead_block,  # noqa: E402
                                   register_poses_to_time, window_last_indices)
from val40_lead_run import ego_dict, obs_dict  # noqa: E402


def ego_path_in_window(ego: dict, t0: float, ts: np.ndarray) -> np.ndarray:
    """TRUE ego positions at ``ts`` in the window-origin frame at ``t0``. (K, 2)."""
    et = ego["t"]
    x0, y0 = float(np.interp(t0, et, ego["x"])), float(np.interp(t0, et, ego["y"]))
    yaw0 = float(np.interp(t0, et, ego["yaw"]))
    dx = np.interp(ts, et, ego["x"]) - x0
    dy = np.interp(ts, et, ego["y"]) - y0
    c, s = np.cos(yaw0), np.sin(yaw0)
    return np.stack([dx * c + dy * s, -dx * s + dy * c], axis=1)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--episodes", default="/home/nvidia/valdata/physicalai-val-0c5f7dac3b11")
    ap.add_argument("--manifest", default="/home/nvidia/leadwork/manifest_EVALPOD_val40.json")
    ap.add_argument("--bundle", default="/home/nvidia/leadwork/val40_lead_bundle.zip")
    ap.add_argument("--index", default="/home/nvidia/leadwork/val40_lead_index.json")
    ap.add_argument("--out", default="/home/nvidia/leadwork/val40_distance_keeping.json")
    a = ap.parse_args()

    ts_rel = np.array([0.5, 1.0, 1.5, 2.0])
    DT = 0.5
    man = {e["file"]: e for e in json.loads(Path(a.manifest).read_text())["episodes"]}
    idx = json.loads(Path(a.index).read_text())
    zf = zipfile.ZipFile(a.bundle)

    GT, CV, LEADS, LENS, SPD, EID = [], [], [], [], [], []
    for ep in sorted(man):
        clip = idx[ep]["clip_id"]
        me, mo = f"egomotion/{clip}.parquet", f"obstacle/{clip}.parquet"
        if me not in zf.namelist():
            continue
        d = torch.load(Path(a.episodes) / ep, map_location="cpu", weights_only=False)
        poses = d["poses"].numpy().astype(np.float64)
        ego = ego_dict(pd.read_parquet(io.BytesIO(zf.read(me))))
        try:
            reg = register_poses_to_time(poses[:, :2], ego["t"], ego["x"], ego["y"])
        except RegistrationError:
            continue
        widx = window_last_indices(int(poses.shape[0]))
        t0s = reg["t_s"][widx]
        obs = obs_dict(pd.read_parquet(io.BytesIO(zf.read(mo)))) if mo in zf.namelist() else None
        blk = lead_block(t0s, ts_rel, obs, ego)
        keep = np.flatnonzero(blk["state"] == LEAD)
        for i in keep:
            t0 = float(t0s[i])
            GT.append(ego_path_in_window(ego, t0, t0 + ts_rel))
            v0 = float(blk["speeds"][i])
            CV.append(np.stack([v0 * ts_rel, np.zeros_like(ts_rel)], axis=1))
            LEADS.append(blk["leads"][i])
            LENS.append(blk["lead_lens"][i])
            SPD.append(v0)
            EID.append(ep)

    n = len(GT)
    print(f"LEAD windows assembled: {n}", flush=True)
    gt = torch.tensor(np.stack(GT), dtype=torch.float32)
    cv = torch.tensor(np.stack(CV), dtype=torch.float32)
    lead = {"leads": np.stack(LEADS), "lead_lens": np.array(LENS),
            "speeds": np.array(SPD), "eid": np.array(EID, dtype=object)}

    res = {}
    for name, pred in (("GT", gt), ("CV_hold_v0", cv)):
        # ⭐ the load-bearing call: lead= must flip distance_keeping OFF of UNAVAILABLE
        out = four_families.longitudinal(pred, gt, DT, lead)
        dk = out["distance_keeping"]
        res[name] = {"status": dk.get("status", "COMPUTED"),
                     "mean_headway_min_m": dk.get("mean_headway_min_m"),
                     "mean_time_gap_min_s": dk.get("mean_time_gap_min_s"),
                     "mean_min_ttc_s": dk.get("mean_min_ttc_s"),
                     "n": dk.get("n"), "n_closing": dk.get("n_closing"),
                     "n_time_gap": dk.get("n_time_gap"), "dt_s": dk.get("dt_s"),
                     "gap_convention": dk.get("gap_convention"),
                     "admitted_by": dk.get("admitted_by"),
                     "ttc_cap_s": dk.get("ttc_cap_s")}
    # what the family reports with NO lead — the before/after that proves the wiring
    res["NO_LEAD_ARG_control"] = four_families.longitudinal(
        gt, gt, DT)["distance_keeping"].get("status")

    def g(name, k):
        v = res[name].get(k)
        return None if v is None else float(v)

    res["D_LEAD_1_sign_reproduction"] = {
        "surface": f"val40 LEAD windows (n={n}) — NOT D-LEAD-1's 14,027-window surface",
        "note": "SIGN is the reproduction target; magnitudes differ by surface and are not claimed",
        "delta_GT_minus_CV": {
            k: (None if (g("GT", k) is None or g("CV_hold_v0", k) is None)
                else round(g("GT", k) - g("CV_hold_v0", k), 4))
            for k in ("mean_headway_min_m", "mean_time_gap_min_s", "mean_min_ttc_s")},
        "D_LEAD_1_reference_deltas": {"mean_headway_min_m": 0.9769,
                                      "mean_time_gap_min_s": 0.1641,
                                      "mean_min_ttc_s": 1.7474},
    }
    d = res["D_LEAD_1_sign_reproduction"]["delta_GT_minus_CV"]
    res["D_LEAD_1_sign_reproduction"]["all_three_positive"] = all(
        v is not None and v > 0 for v in d.values())
    res["n_lead_windows"] = n
    Path(a.out).write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
