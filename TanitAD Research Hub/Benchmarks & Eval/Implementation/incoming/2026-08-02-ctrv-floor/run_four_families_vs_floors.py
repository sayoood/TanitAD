"""The four metric families (CLAUDE.md, BINDING 2026-08-02) for MODEL **and every FLOOR**.

The binding rule says an eval reporting ADE alone is incomplete. This driver applies
`taniteval/four_families.py` not just to the arm but to each trivial floor, because the whole point
of 2026-08-02's finding is that the *comparator* decides what the families say: a straight-line floor
cannot have a curvature error, so "our lateral family beats the floor" is uninformative until the
floor can turn.

⚠️ **CADENCE.** `four_families` hardcodes ``DT_S = 0.1`` because it is written for the dense 10 Hz
path. The banked pre-2026-07-25 dumps carry only the **sparse 4-waypoint view at 0.5 s**, so this
driver sets ``ff.DT_S = 0.5`` and says so in the output. Speeds/accelerations are therefore computed
over 0.5 s steps — correct for this surface, and NOT comparable to a dense-path run of the same
module. Re-run on `pred_dense`/`gt_dense` when a dump carries them.

    cd /workspace && PYTHONPATH=/workspace/TanitAD/stack:/workspace/TanitAD/taniteval \\
      python3 run_four_families_vs_floors.py --arms flagship-30k refc-xl-30k
"""
from __future__ import annotations

import argparse
import math
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import four_families as ff  # noqa: E402  (shipped standalone; torch-only)
from ctrv_floor import build_floors, hold_v0_waypoints, verify_alignment  # noqa: E402

from taniteval import ci as _ci  # noqa: E402

ff.DT_S = 0.5                    # THE SPARSE 4-WAYPOINT CADENCE — see module docstring

LONG_KEYS = ("speed_mae_mps", "speed_bias_mps", "along_mae_m", "along_final_bias_m")
LAT_KEYS = ("heading_mae_deg", "yaw_rate_mae_degps", "curvature_mae_1pm", "cross_mae_m",
            "cross_bias_m")
AGREE_TOL = 1e-3        # a per-window form that disagrees with the module is NOT published


def per_window_families(pred, gt):
    """Per-window forms of the module's family scalars, so they can be bootstrapped.

    ⛔ These are a REIMPLEMENTATION and are only admissible where they reproduce
    ``four_families.longitudinal`` / ``.lateral`` — which `main()` checks per metric against
    ``AGREE_TOL`` and REFUSES on mismatch. (First cut of this file returned heading in RADIANS and
    silently dropped curvature and yaw-rate by mis-keying ``_seq_geometry``; that is exactly the
    C63 failure — an imported metric published before its precondition was measured.)
    """
    P, G = ff._seq_geometry(pred), ff._seq_geometry(gt)
    both = P["valid"] & G["valid"]
    both_pair = P["pair_valid"] & G["pair_valid"]
    deg = 180.0 / math.pi

    def _rowmean(x, m):
        n = m.sum(1)
        s = torch.where(m, x, torch.zeros_like(x)).sum(1)
        return torch.where(n > 0, s / n.clamp(min=1), torch.full_like(s, float("nan")))

    dh = P["heading"] - G["heading"]
    dh = (dh + math.pi) % (2 * math.pi) - math.pi
    ct = P["cross"] - G["cross"]
    return {
        "speed_mae_mps": (P["speed"] - G["speed"]).abs().mean(1),
        "speed_bias_mps": (P["speed"] - G["speed"]).mean(1),
        "along_mae_m": (P["along"] - G["along"]).abs().mean(1),
        "along_final_bias_m": (P["along"] - G["along"])[:, -1],
        "heading_mae_deg": _rowmean(dh.abs() * deg, both),
        "yaw_rate_mae_degps": _rowmean((P["yaw_rate"] - G["yaw_rate"]).abs() * deg, both_pair),
        "curvature_mae_1pm": _rowmean((P["curvature"] - G["curvature"]).abs(), both_pair),
        "cross_mae_m": ct.abs().mean(1),
        "cross_bias_m": ct.mean(1),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--val-dir", default="/workspace/val40cache")
    ap.add_argument("--results-dir", default="/workspace/TanitAD/taniteval/results")
    ap.add_argument("--arms", nargs="+", default=["flagship-30k"])
    ap.add_argument("--out", default="/workspace/four_families_vs_floors.json")
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(a.val_dir, "ep_*.pt")))
    eps = []
    for i, f in enumerate(files):
        d = torch.load(f, map_location="cpu", weights_only=False)
        eps.append((i, d["poses"].float(),
                    int(min(d["frames_u8"].shape[0], d["actions"].shape[0],
                            d["poses"].shape[0]))))
    built = build_floors(eps)

    out = {"block": "bench-eval/four-families-vs-floors",
           "binding_rule": "CLAUDE.md 2026-08-02 — four families, ADDED to ADE, never pooled",
           "cadence_s": ff.DT_S,
           "cadence_note": ("sparse 4-waypoint view at 0.5 s; four_families' default DT_S=0.1 is "
                            "for the dense 10 Hz path and is NOT what these dumps carry"),
           "estimator": {"delta": "paired_episode_cluster_bootstrap", "n_boot": 2000,
                         "seed": 0, "orientation": "floor - model; positive = model wins"},
           "arms": {}}

    for arm in a.arms:
        win = torch.load(Path(a.results_dir) / f"windows_{arm}.pt",
                         map_location="cpu", weights_only=False)
        al = verify_alignment(built, win)
        if not al["aligned"]:
            out["arms"][arm] = {"refused": "window alignment failed", **al}
            continue
        gt, eid = win["gt"].float(), list(win["eid"])
        srcs = {"model": win["pred"].float(), "cv": built["cv"].float(),
                "holdv0": hold_v0_waypoints(win["speed"].float(), n=gt.shape[1]),
                "ctrv_gated": built["ctrv_gated"].float()}
        pw = {k: per_window_families(v, gt) for k, v in srcs.items()}
        # ⛔ AUTHORITATIVE point estimates come from the module itself, never from the
        # reimplementation above. The reimplementation only supplies the paired interval, and
        # only for metrics where it reproduces the module.
        mod = {s: {**ff.longitudinal(v, gt), **ff.lateral(v, gt)}
               for s, v in srcs.items()}

        row = {"n_windows": int(gt.shape[0]), "n_episodes": len(set(eid)),
               "LONGITUDINAL": {"point": {}, "vs_floor_paired": {},
                                "distance_keeping": mod["model"]["distance_keeping"]},
               "LATERAL": {"point": {}, "vs_floor_paired": {},
                           "coverage": {k: mod["model"][k] for k in
                                        ("n_steps_heading", "n_steps_curvature",
                                         "excluded_below_min_ds", "min_ds_m")}},
               "TACTICAL": ff.tactical(win), "STRATEGIC": ff.strategic(win),
               "agreement_vs_module": {}, "refused_metrics": {}}
        for fam, keys in (("LONGITUDINAL", LONG_KEYS), ("LATERAL", LAT_KEYS)):
            for k in keys:
                if k not in pw["model"] or mod["model"].get(k) is None:
                    continue
                diff = max(abs(float(np.nanmean(np.asarray(pw[s][k], dtype=float)))
                               - float(mod[s][k])) for s in srcs)
                row["agreement_vs_module"][k] = round(diff, 8)
                row[fam]["point"][k] = {s: mod[s][k] for s in srcs}
                if diff > AGREE_TOL:
                    row["refused_metrics"][k] = (
                        f"per-window form disagrees with four_families by {diff:.2e} "
                        f"(> {AGREE_TOL}); point estimate published, INTERVAL REFUSED")
                    continue
                row[fam]["vs_floor_paired"][k] = {
                    s: _ci.paired_episode_cluster_bootstrap(
                        np.asarray(pw[s][k], dtype=float),
                        np.asarray(pw["model"][k], dtype=float), eid,
                        n_boot=2000, seed=0)
                    for s in ("cv", "holdv0", "ctrv_gated")}
        out["arms"][arm] = row
        print(f"[{arm}] LONG speed_mae "
              f"model {row['LONGITUDINAL']['point']['speed_mae_mps']['model']} "
              f"ctrv {row['LONGITUDINAL']['point']['speed_mae_mps']['ctrv_gated']} · "
              f"LAT heading "
              f"model {row['LATERAL']['point'].get('heading_mae_deg', {}).get('model')} "
              f"ctrv {row['LATERAL']['point'].get('heading_mae_deg', {}).get('ctrv_gated')}",
              flush=True)

    out["families_unavailable"] = ["TACTICAL", "STRATEGIC", "LONGITUDINAL.distance_keeping"]
    out["work_items"] = ("a family reported UNAVAILABLE is a WORK ITEM, not a pass "
                         "(CLAUDE.md binding rule 5)")
    Path(a.out).write_text(json.dumps(out, indent=1))
    print("[done]", a.out, flush=True)


if __name__ == "__main__":
    main()
