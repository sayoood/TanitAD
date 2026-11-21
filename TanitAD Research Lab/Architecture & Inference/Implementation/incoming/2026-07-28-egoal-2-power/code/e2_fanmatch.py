#!/usr/bin/env python3
"""E-GOAL-2 gate F-A -- does the 600-episode fan CONTAIN the committed one?

⚠️ WHY THIS GATE EXISTS. Everything in §6 rests on the 600-episode decode being
the same decode that produced `taniteval/results/fan_refc-xl-30k.pt`. The
published 40 episodes are `val600[0:40]` element-for-element by pose-sha256
(S0/B, verified across two hosts), `list_val_episodes` returns a SORTED listing,
and the decode is deterministic -- so the first 881 rows of the 600-episode
dump must reproduce the committed fan EXACTLY.

⛔ This gate can fail, and if it does the n = 600 numbers are VOID, not
"approximately right". Three streams in two days shipped stable, plausible,
wrong numbers out of their own scoring code.

Compared: `fan` (all 881x256x4x2 values), `gt`, `cv`, `sel`, `logits`, `v0`,
`speed`, and the episode-id blocking. Reported as max absolute difference per
field, not as a summary statistic that a wrong join could pass.

Run (dev box, CPU):
    python e2_fanmatch.py --big <fan600.pt> --out ../raw/e2_fanmatch.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
STREAM = HERE.parent
REPO = STREAM.parents[4]
REF = REPO / "taniteval" / "results" / "fan_refc-xl-30k.pt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", required=True)
    ap.add_argument("--ref", default=str(REF))
    ap.add_argument("--out", default=str(STREAM / "raw" / "e2_fanmatch.json"))
    a = ap.parse_args()

    r = torch.load(a.ref, map_location="cpu", weights_only=False)
    b = torch.load(a.big, map_location="cpu", weights_only=False)
    W = r["fan"].shape[0]

    r_eid = np.asarray([str(x) for x in r["eid"]])
    b_eid = np.asarray([str(x) for x in b["eid"]])
    res = {"_stream": "2026-07-28-egoal-2-power", "_gate": "F-A prefix identity",
           "ref": {"path": a.ref, "windows": int(W),
                   "episodes": int(len(np.unique(r_eid))),
                   "ckpt": str(r.get("ckpt")), "step": int(r.get("ckpt_step", -1)),
                   "steps": int(r.get("steps", -1)),
                   "nav_mode": str(r.get("nav_mode"))},
           "big": {"path": a.big, "windows": int(b["fan"].shape[0]),
                   "episodes": int(len(np.unique(b_eid))),
                   "ckpt": str(b.get("ckpt")), "step": int(b.get("ckpt_step", -1)),
                   "steps": int(b.get("steps", -1)),
                   "nav_mode": str(b.get("nav_mode")),
                   "wall_s": b.get("wall_s")}}

    res["decode_config_identical"] = {
        k: bool(str(r.get(k)) == str(b.get(k)))
        for k in ("ckpt", "ckpt_step", "steps", "nav_mode", "n_anchors",
                  "wp_steps", "vtarget_source")}

    # -- the episode blocking must line up before any value is compared -----
    same_eid = bool(np.array_equal(r_eid, b_eid[:W]))
    res["episode_blocking_identical"] = same_eid
    res["first_windows_per_episode"] = {
        "ref": [int((r_eid == e).sum()) for e in dict.fromkeys(r_eid)][:5],
        "big": [int((b_eid[:W] == e).sum()) for e in dict.fromkeys(b_eid[:W])][:5]}

    fields = ("fan", "gt", "cv", "logits", "v0", "speed", "sel", "head_deg",
              "a_gt", "v_target")
    diffs = {}
    for k in fields:
        if k not in r or k not in b:
            diffs[k] = None
            continue
        x = r[k].numpy().astype(np.float64)
        y = b[k][:W].numpy().astype(np.float64)
        if x.shape != y.shape:
            diffs[k] = {"shape_mismatch": [list(x.shape), list(y.shape)]}
            continue
        diffs[k] = {"max_abs_diff": float(np.nanmax(np.abs(x - y))),
                    "n_exact": int(np.sum(x == y)), "n": int(x.size)}
    res["per_field"] = diffs

    hard = [k for k in ("fan", "gt", "sel", "v0", "logits")
            if isinstance(diffs.get(k), dict)
            and diffs[k].get("max_abs_diff", 1.0) != 0.0]
    res["PASSES"] = bool(same_eid and not hard
                         and all(res["decode_config_identical"].values()))
    res["failing_fields"] = hard
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: v for k, v in res.items() if k != "per_field"},
                     indent=1))
    print("per-field max|diff|:",
          json.dumps({k: (v.get("max_abs_diff") if isinstance(v, dict) else v)
                      for k, v in diffs.items()}))
    print("F-A", "PASS" if res["PASSES"] else "FAIL")


if __name__ == "__main__":
    main()
