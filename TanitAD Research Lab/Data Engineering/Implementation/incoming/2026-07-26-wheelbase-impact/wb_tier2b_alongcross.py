#!/usr/bin/env python3
"""TIER 2b — the along/cross (longitudinal/lateral) split of the Tier-2 deltas.

`taniteval.lateral.block` on the eval pod SKIPS: that pod's `rollout.collect`
predates the 2026-07-25 dense-path upgrade, so `pred_dense/gt_dense` do not
exist and the module refuses (correctly) to fit a compounding law to 4 knots.
`lateral.paired_cross_track` does return, but the pod copy reports
`step=4, horizon_s=0.4` against a 4-knot sparse surface, i.e. its horizon
bookkeeping does not match the surface it was handed — not quotable.

So the split is computed here EXPLICITLY on the sparse surface, where it is
unambiguous: `rollout.collect` returns waypoints already in the EGO frame of
the window's last pose (`driving_diagnostic.gt_ego_waypoints`), x forward,
y left. Therefore at knot j:
      along_err = |pred[:, j, 0] - gt[:, j, 0]|      (longitudinal)
      cross_err = |pred[:, j, 1] - gt[:, j, 1]|      (lateral)
No frenet fit, no dense path, no assumption beyond the frame convention that
`assert_axis_convention` already pins.

Paired CIs: `taniteval.ci.paired_episode_cluster_bootstrap`, B=2000, over the
40 val episodes. Never `overlapping_holdout_se`.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/root/taniteval")
sys.path.insert(0, "/root/TanitAD/stack")
sys.path.insert(0, "/root/TanitAD/stack/scripts")

from taniteval import ci, data, loaders, rollout  # noqa: E402
from taniteval.registry import MODELS  # noqa: E402

import wb_tier2_input_eval as t2  # noqa: E402  (arm definitions, one source)

VAL = "/root/valdata/physicalai-val-0c5f7dac3b11"
KNOTS_S = (0.5, 1.0, 1.5, 2.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="flagship-30k")
    ap.add_argument("--wheelbase-json", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--dump", default="/root/wb_tier2_windows.pt")
    ap.add_argument("--n-boot", type=int, default=2000)
    a = ap.parse_args()

    wbmap = json.loads(Path(a.wheelbase_json).read_text())
    e = [m for m in MODELS if m["key"] == a.model][0]
    L = loaders.load(e, "cuda")
    files = list(data.list_val_episodes(VAL, 40))
    eps = data.load_frames(files)
    wb_by_idx = [float(wbmap[Path(f).name]) for f in files]

    ARMS = ("shipped", "corrected", "zeroed", "x1.5", "global_mean",
            "dispersion", "harness_2p7")
    P, G, eid = {}, None, None
    for mode in ARMS:
        win = rollout.collect(L["model"], L["step_readout"],
                              t2.patch_steer(eps, mode, wb_by_idx), "cuda",
                              speed_input=bool(e.get("speed_input")))
        P[mode] = win["pred"].numpy().astype(np.float64)     # [N,4,2] ego frame
        if G is None:
            G, eid = win["gt"].numpy().astype(np.float64), win["eid"]
        print(f"[t2b] {mode} done", flush=True)
    torch.save({"pred": {k: torch.tensor(v) for k, v in P.items()},
                "gt": torch.tensor(G), "eid": eid,
                "wb_by_idx": wb_by_idx}, a.dump)

    res = {"_meta": {
        "evidence_class": "MEASURED", "model": a.model,
        "n_windows": int(G.shape[0]), "n_episodes": len(set(eid)),
        "frame": "ego frame of the window's last pose; x forward (along), y left (cross)",
        "estimator": "paired_episode_cluster_bootstrap", "n_boot": a.n_boot,
        "knots_s": list(KNOTS_S)}}

    def comp(pred, j):
        return (np.abs(pred[:, j, 0] - G[:, j, 0]),
                np.abs(pred[:, j, 1] - G[:, j, 1]))

    # cumulative (ADE-style) along/cross over knots 0..j
    def cum(pred, j):
        al = np.abs(pred[:, :j + 1, 0] - G[:, :j + 1, 0]).mean(axis=1)
        cr = np.abs(pred[:, :j + 1, 1] - G[:, :j + 1, 1]).mean(axis=1)
        return al, cr

    res["levels"] = {}
    for mode in ARMS:
        al2, cr2 = comp(P[mode], 3)                    # the 2 s knot
        cal, ccr = cum(P[mode], 3)
        res["levels"][mode] = {
            "along_mae@2s": round(float(al2.mean()), 6),
            "cross_mae@2s": round(float(cr2.mean()), 6),
            "along_cum_0_2s": round(float(cal.mean()), 6),
            "cross_cum_0_2s": round(float(ccr.mean()), 6),
            "sq_energy_share_longitudinal@2s": round(float(
                (al2 ** 2).sum() / ((al2 ** 2).sum() + (cr2 ** 2).sum())), 4),
        }
    res["paired_vs_shipped"] = {}
    for mode in ARMS[1:]:
        blk = {}
        for lab, f in (("along@2s", lambda p: comp(p, 3)[0]),
                       ("cross@2s", lambda p: comp(p, 3)[1]),
                       ("along_cum_0_2s", lambda p: cum(p, 3)[0]),
                       ("cross_cum_0_2s", lambda p: cum(p, 3)[1])):
            blk[lab] = ci.paired_episode_cluster_bootstrap(
                f(P[mode]), f(P["shipped"]), eid, n_boot=a.n_boot, reduce="mean")
        res["paired_vs_shipped"][mode] = blk
        print(f"[t2b] {mode:12s} d_along@2s={blk['along@2s']['delta']:+.5f} "
              f"{ 'SEP' if blk['along@2s']['separated'] else '   '} | "
              f"d_cross@2s={blk['cross@2s']['delta']:+.5f} "
              f"{'SEP' if blk['cross@2s']['separated'] else ''}", flush=True)

    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    print("[t2b] wrote", a.out)


if __name__ == "__main__":
    main()
