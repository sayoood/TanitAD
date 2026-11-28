#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""opcount_steps.json (R7 / R6, b1 / b2) -> per-component kernel / FLOP / byte / sync tables and the
R7 - R6 difference, plus the per-sample slope (b2 - b1) and a linear b16 projection.

Components are scope-name groups (rules below, first match wins); every scope is listed raw too, so
a grouping error is visible. COUNTS ONLY -- no time is claimed from a CPU run."""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

RULES = [
    (r"^fn:conflict", "conflict_probe"),
    (r"^fn:trunk_s8_tap", "trunk_s8_tap_pass"),
    (r"^(fn:trunk_main|mod:core\.encoder)", "trunk_main"),
    (r"^(mod:_map_hires|fn:map_near_geometry)", "map10_branch"),
    (r"^fn:map_per_class", "map10_per_class_signal"),
    (r"^fn:(map_hires_ce|map_hires_loss_row)", "map10_loss"),
    (r"^fn:(match_cost_build|hungarian_cpu|match_slots)", "slot_matching"),
    (r"^fn:det_", "box_detection_census"),
    (r"^fn:(slot_set_loss|box3d_set_loss|refined_slot_losses|presence_term|ignore_presence_weight)",
     "slot_losses"),
    (r"^mod:_perception", "perception_branch(box_decoder,bev_pool,0.5m_map_if_refcv6)"),
    (r"^mod:core\.decoder", "planner_decoder"),
    (r"^mod:core", "core_other"),
    (r"^fn:(grad_reach|log_row|grad_probe_row)", "logging_grad_reach"),
    (r"^fn:clip_grad", "clip_grad"),
    (r"^fn:h2d_frames", "frames_to_device"),
    (r"^phase:fwd_loss", "loss_glue(compute_losses_v3 body)"),
    (r"^mod:", "other_modules"),
    (r"^<top>", "outside_scopes(optimizer,zero_grad,loop)"),
    (r"^<untagged>", "untagged_bwd(AccumulateGrad etc.)"),
]
KEYS = ("kernels", "flops", "bytes", "sync_item", "sync_cpu", "sync_tolist", "sync_numpy")


def group_of(scope: str) -> tuple[str, str]:
    base, phase = scope, "fwd"
    for suf in ("@bwd", "@recompute"):
        if scope.endswith(suf):
            base, phase = scope[: -len(suf)], suf[1:]
    for rx, g in RULES:
        if re.search(rx, base):
            return g, phase
    return "unmatched", phase


def load(p: Path):
    rows = json.loads(p.read_text(encoding="utf-8"))
    return {int(r["step"]): r["tally"] for r in rows}


def by_group(tally: dict) -> dict:
    out = defaultdict(lambda: defaultdict(float))
    for sc, t in tally.items():
        g, ph = group_of(sc)
        for k in KEYS:
            v = float(t.get(k, 0.0))
            out[g][f"{k}_{ph}"] += v
            out[g][f"{k}_all"] += v
    return {g: dict(v) for g, v in out.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--step", type=int, default=1, help="steady step to read (0 = first step)")
    a = ap.parse_args()
    root = Path(a.root)
    res = {"step": a.step, "rules": RULES, "runs": {}}
    G = {}
    for job in ("R7_opcount_b1", "R6_opcount_b1", "R7_opcount_b2", "R6_opcount_b2"):
        f = root / job / "opcount_steps.json"
        if not f.exists():
            continue
        st = load(f)
        if a.step not in st:
            continue
        G[job] = by_group(st[a.step])
        res["runs"][job] = {"groups": G[job],
                            "scopes_top40_by_kernels": sorted(
                                ((sc, t.get("kernels", 0), t.get("flops", 0), t.get("bytes", 0),
                                  t.get("sync_item", 0) + t.get("sync_cpu", 0)
                                  + t.get("sync_tolist", 0) + t.get("sync_numpy", 0))
                                 for sc, t in st[a.step].items()), key=lambda x: -x[1])[:40]}
    diffs = {}
    for b in (1, 2):
        r7, r6 = G.get(f"R7_opcount_b{b}"), G.get(f"R6_opcount_b{b}")
        if not (r7 and r6):
            continue
        d = {}
        for g in set(r7) | set(r6):
            d[g] = {k: r7.get(g, {}).get(k, 0.0) - r6.get(g, {}).get(k, 0.0)
                    for k in set(r7.get(g, {})) | set(r6.get(g, {}))}
        diffs[f"b{b}"] = d
    res["diff_R7_minus_R6"] = diffs
    if "b1" in diffs and "b2" in diffs:
        proj = {}
        for g in set(diffs["b1"]) | set(diffs["b2"]):
            proj[g] = {}
            for k in ("kernels_all", "flops_all", "bytes_all", "sync_item_all", "sync_cpu_all"):
                v1 = diffs["b1"].get(g, {}).get(k, 0.0)
                v2 = diffs["b2"].get(g, {}).get(k, 0.0)
                proj[g][k + "_b16_linear"] = v1 + 15 * (v2 - v1)
        res["projection_b16_linear"] = proj
    Path(a.out).write_text(json.dumps(res, indent=1), encoding="utf-8")
    for job, g in G.items():
        print("==", job)
        for name, v in sorted(g.items(), key=lambda x: -x[1].get("kernels_all", 0)):
            syncs = sum(v.get(f"{k}_all", 0) for k in ("sync_item", "sync_cpu", "sync_tolist", "sync_numpy"))
            print(f"  {name:55s} k_fwd {v.get('kernels_fwd',0):7.0f} k_bwd {v.get('kernels_bwd',0):7.0f} "
                  f"k_rec {v.get('kernels_recompute',0):6.0f} GF {v.get('flops_all',0)/1e9:9.1f} "
                  f"GB {v.get('bytes_all',0)/1e9:7.2f} syncs {syncs:5.0f}")


if __name__ == "__main__":
    main()
