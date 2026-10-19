"""force_lon_ka_floor_r8.py -- BANKED DATA ONLY (no model, no roll): how big is the draw-to-draw noise of the mean ADE on the force_lon window set?

KA of the force_lon experiment compares CPU native against the banked GPU `os` and has a bar of |mean ADE diff| <= 0.05 m on the subset.  Two
independent sampler draws of the SAME model on the SAME windows already exist on disk (inference seeds 0 and 1 of the EMA-30k battery), so the
noise floor of that very statistic can be read BEFORE the roll finishes: if two GPU draws differ by more than 0.05 m on this window set, a 0.05 m
bar is not attainable by a faithful re-roll and KA must be read against the floor, not against the bar.

    python -X utf8 force_lon_ka_floor_r8.py        # (numpy only) writes <PKG>/raw/force_lon/KA_NOISE_FLOOR_BANKED.json
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import force_lon_reroll_r8 as F  # noqa: E402


def load_panel_np(root: str, seed: int) -> dict:
    """Torch-free read of an S6 panel: {"clip|ws": (os [6,2], g [6,2], episode file index)} -- numpy + json only."""
    import glob
    pdir = os.path.join(root, f"s6_s{seed}")
    man = json.load(open(os.path.join(pdir, "manifest.json"), encoding="utf-8"))
    cid = {int(e["file_index"]): str(e["clip_id"]) for e in man["episodes"]}
    out = {}
    for f in sorted(glob.glob(os.path.join(pdir, "ep*.npz"))):
        fi = int(os.path.basename(f)[2:5])
        z = np.load(f)
        for j, w in enumerate(z["ws"].tolist()):
            out[f"{cid[fi]}|{int(w)}"] = (z["os"][j].astype(np.float64), z["g"][j].astype(np.float64), fi)
    return out


def main() -> int:
    P0, P1 = load_panel_np(F.ARCH, 0), load_panel_np(F.ARCH, 1)
    plan = F.jload(F.SCR / "plan_raw.json")["entries"]
    import s6_lon_decomp as SD                                      # torch-free at import
    out = {"tool": "force_lon_ka_floor_r8.py", "tier": F.TIER, "banked_tree": F.ARCH,
           "what": "GPU inference-seed 1 minus seed 0 of the SAME EMA-30k model on the force_lon window set (banked; no new compute; numpy only). "
                   "This is the draw-to-draw floor of the KA statistic (mean ADE difference over the window set). ADE = mean over the 6 S6 instants "
                   "of the L2 distance to GT (numpy; identical to the battery's `ade_m`, which s6_lon_decomp asserts).", "sets": {}}
    for setname in ("FB", "CT", "ALL"):
        keys = [f"{e['clip']}|{e['ws']}" for e in plan if setname == "ALL" or e["set"] == setname]
        a0 = np.asarray([np.linalg.norm(P0[k][0] - P0[k][1], axis=-1).mean() for k in keys])
        a1 = np.asarray([np.linalg.norm(P1[k][0] - P1[k][1], axis=-1).mean() for k in keys])
        assert all(np.array_equal(P0[k][1], P1[k][1]) for k in keys), "seed-0 and seed-1 GT differ"
        d = a1 - a0
        eid = np.asarray([P0[k][2] for k in keys])
        mc = F.mean_ci(d, eid, SD)
        flips = np.abs(d) > 1.0
        out["sets"][setname] = {
            "n_windows": int(len(keys)), "n_episodes": int(len(set(eid.tolist()))),
            "mean_ADE_s0": float(a0.mean()), "mean_ADE_s1": float(a1.mean()),
            "mean_diff_s1_minus_s0": mc, "abs_mean_diff_exceeds_0p05": bool(abs(mc["mean"]) > 0.05),
            "per_window_diff_sd": float(d.std(ddof=1)), "per_window_diff_rms": float(np.sqrt((d ** 2).mean())),
            "iid_se_of_mean_diff": float(d.std(ddof=1) / np.sqrt(len(d))),
            "frac_windows_with_abs_diff_gt_1m (mode flips)": float(flips.mean()),
            "frac_windows_with_abs_diff_gt_0p25m": float((np.abs(d) > 0.25).mean()),
            "mean_abs_diff_excluding_flips": float(np.abs(d[~flips]).mean()) if (~flips).any() else None}
    F.jdump(out, F.OUT / "KA_NOISE_FLOOR_BANKED.json")
    print(json.dumps({k: {kk: v[kk] for kk in ("n_windows", "mean_diff_s1_minus_s0", "iid_se_of_mean_diff", "per_window_diff_rms",
                                                "frac_windows_with_abs_diff_gt_1m (mode flips)")} for k, v in out["sets"].items()}, indent=1))
    print("ZZFLKAFLOORDONEZZ")
    return 0


if __name__ == "__main__":
    sys.exit(main())
